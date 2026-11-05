import colorsys
import os
import random
import sys
import re
import warnings
import importlib.util
from collections import Counter, defaultdict, namedtuple
from pathlib import Path
import networkx as nx
from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tqdm import tqdm
import pickle
import json
from copy import deepcopy
try:
    from .utils import create_structure
except ImportError:
    # Script-mode compatibility when repograph/ is inserted directly in sys.path.
    from utils import create_structure
from tree_sitter_languages import get_language, get_parser
import argparse
import subprocess
import shutil
import time
from typing import List, Dict, Any, Optional, Tuple, Iterable

try:
    from .git_cache_utils import (
        _git_clone_env,
        _git_clone_http_opts,
        _github_https_clone_url,
        _git_clone_no_checkout,
        ensure_git_worktree,
        safe_version_dirname,
        write_repo_versions_manifest,
    )
except ImportError:
    from git_cache_utils import (
        _git_clone_env,
        _git_clone_http_opts,
        _github_https_clone_url,
        _git_clone_no_checkout,
        ensure_git_worktree,
        safe_version_dirname,
        write_repo_versions_manifest,
    )

_ROOT_PKG = Path(__file__).resolve().parent.parent
if str(_ROOT_PKG) not in sys.path:
    sys.path.insert(0, str(_ROOT_PKG))
from agent.repograph.java_path_filters import is_localizable_java_source_path

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)

_WARNED_NON_UTF8_SOURCES = set()


def _read_source_text(path) -> str:
    """Read legacy Java source without aborting a repository-wide graph build.

    Defects4J contains historical source files encoded as Windows-1252 or
    ISO-8859-1. Java syntax is ASCII-compatible in all three encodings, so a
    conservative fallback preserves the source structure and line boundaries.
    """
    source_path = Path(path)
    raw = source_path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as utf8_error:
        selected = "latin-1"
        try:
            text = raw.decode("cp1252")
            selected = "cp1252"
        except UnicodeDecodeError:
            # Latin-1 maps every byte and is the final lossless byte fallback.
            text = raw.decode("latin-1")
        warning_key = str(source_path)
        if warning_key not in _WARNED_NON_UTF8_SOURCES:
            _WARNED_NON_UTF8_SOURCES.add(warning_key)
            warnings.warn(
                f"Non-UTF-8 Java source decoded as {selected}: {source_path} "
                f"({utf8_error})",
                RuntimeWarning,
            )
        return text


# 【新增-注释属性】扩展 Tag 以支持 comment；保留原 7 字段兼容
# 原: rel_fname fname line name kind category info
# Tag = namedtuple("Tag", "rel_fname fname line name kind category info comment".split())

# Tag 定义（已包含 annotations）
Tag = namedtuple("Tag", "rel_fname fname line name kind category info comment annotations".split())


def _line_to_str(line_val):
    if isinstance(line_val, (list, tuple)):
        return "-".join([str(x) for x in line_val])
    return str(line_val)


def make_node_id(rel_fname: str, name: str, kind: str, category: str, line) -> str:
    # Stable unique id per symbol occurrence. Keeps compatibility with current fields.
    return f"{rel_fname}::{name}::{kind}::{category}::{_line_to_str(line)}"

class CodeGraph:

    warned_files = set()

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=None,
        enable_inheritance_edges: bool = True,
        structure_only_files=None,
    ):
        self.io = io
        self.verbose = verbose

        if not root:
            root = os.getcwd()
        self.root = root

        self.max_map_tokens = map_tokens
        self.max_context_window = max_context_window

        # self.token_count = main_model.token_count
        self.repo_content_prefix = repo_content_prefix
        self.enable_inheritance_edges = enable_inheritance_edges
        if structure_only_files is None:
            self.structure = create_structure(self.root)
        else:
            self.structure = create_structure(self.root, only_files=structure_only_files)

    def get_code_graph(self, other_files, mentioned_fnames=None):
        """
        始终返回 (tags, G)。不再在边界条件下返回 None，避免调用方 ``tags, G = ...`` 解包崩溃。
        若无可索引文件或图谱被关闭，返回 ([], empty MultiDiGraph)。
        """
        if self.max_map_tokens <= 0:
            return [], nx.MultiDiGraph()
        if not other_files:
            return [], nx.MultiDiGraph()
        if not mentioned_fnames:
            mentioned_fnames = set()

        max_map_tokens = self.max_map_tokens

        # With no files in the chat, give a bigger view of the entire repo
        MUL = 16
        padding = 4096
        if max_map_tokens and self.max_context_window:
            target = min(max_map_tokens * MUL, self.max_context_window - padding)
        else:
            target = 0

        tags = self.get_tag_files(other_files, mentioned_fnames)
        if tags is None:
            return [], nx.MultiDiGraph()
        code_graph = self.tag_to_graph(tags)

        return tags, code_graph

    def get_tag_files(self, other_files, mentioned_fnames=None):
        try:
            tags = self.get_ranked_tags(other_files, mentioned_fnames)
            return tags
        except RecursionError:
            self.io.tool_error("Disabling code graph, git repo too large?")
            self.max_map_tokens = 0
            return

    def tag_to_graph(self, tags):
        G = nx.MultiDiGraph()
        tag_node_ids = {}
        for tag in tags:
            node_id = make_node_id(tag.rel_fname, tag.name, tag.kind, tag.category, tag.line)
            tag_node_ids[id(tag)] = node_id
            G.add_node(
                node_id,
                symbol_name=tag.name,
                category=tag.category,
                info=tag.info,
                fname=tag.fname,
                rel_fname=tag.rel_fname,
                line=tag.line,
                kind=tag.kind,
                comment=getattr(tag, "comment", ""),
                annotations=getattr(tag, "annotations", [])   # ← 确保传入
            )
        # contains、invokes、继承边保持你原来的逻辑（可后续优化）
        def_nodes_by_name = {}
        def_func_nodes_by_file_name = {}
        for tag in tags:
            if tag.kind == "def":
                nid = tag_node_ids[id(tag)]
                def_nodes_by_name.setdefault(tag.name, []).append(nid)
                if tag.category == "function":
                    key = (tag.rel_fname, tag.name)
                    def_func_nodes_by_file_name.setdefault(key, []).append(nid)

        for tag in tags:
            if tag.category == 'class' and tag.info:
                class_nid = tag_node_ids[id(tag)]
                for f in tag.info.split('\n'):
                    fn = f.strip()
                    if fn:
                        # Prefer method defs in same file for contains relation.
                        targets = def_func_nodes_by_file_name.get((tag.rel_fname, fn), [])
                        if not targets:
                            targets = def_nodes_by_name.get(fn, [])
                        for t in targets:
                            G.add_edge(class_nid, t, edge_type="contains")
        # invokes 边（原双循环可优化，这里先保留）
        tags_ref = [tag for tag in tags if tag.kind == 'ref']
        for tag_ref in tags_ref:
            src = tag_node_ids[id(tag_ref)]
            for dst in def_nodes_by_name.get(tag_ref.name, []):
                G.add_edge(src, dst, edge_type="invokes")

        if self.enable_inheritance_edges:
            self._add_inheritance_edges(G)
        return G

    def _add_inheritance_edges(self, G):
        """【新增】添加继承边：extends、implements，便于分析类层次"""
        class_def_nodes_by_name = {}
        for node_id, nd in G.nodes(data=True):
            if nd.get("category") == "class" and nd.get("kind") == "def":
                class_def_nodes_by_name.setdefault(nd.get("symbol_name"), []).append(node_id)
        for node_name in list(G.nodes()):
            nd = G.nodes[node_name]
            if nd.get("category") != "class" or nd.get("kind") != "def":
                continue
            rel_fname = self.get_rel_fname(nd["fname"])
            ref_fname_lst = rel_fname.replace("\\", "/").split("/")
            s = self.structure
            for part in ref_fname_lst:
                if part not in s:
                    s = None
                    break
                s = s[part]
            if not s or not isinstance(s, dict):
                continue
            for cls in s.get("classes", []):
                if cls.get("name") != nd.get("symbol_name"):
                    continue
                superclass = cls.get("superclass")
                if superclass:
                    for super_node_id in class_def_nodes_by_name.get(superclass, []):
                        G.add_edge(node_name, super_node_id, edge_type="extends")
                for iface in cls.get("interfaces", []):
                    if iface:
                        for iface_node_id in class_def_nodes_by_name.get(iface, []):
                            G.add_edge(node_name, iface_node_id, edge_type="implements")
                break

    def get_rel_fname(self, fname):
        return os.path.relpath(fname, self.root)

    def split_path(self, path):
        path = os.path.relpath(path, self.root)
        return [path + ":"]

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.io.tool_error(f"File not found error: {fname}")

    # 【修改2】get_class_functions 原为 Python ast 遍历，Java 版由 structure 提供类方法信息，此方法保留作占位
    def get_class_functions(self, tree, class_name):
        return []

    def get_func_block(self, first_line, code_block):
        first_line_escaped = re.escape(first_line)
        pattern = re.compile(rf'({first_line_escaped}.*?)(?=(^\S|\Z))', re.DOTALL | re.MULTILINE)
        match = pattern.search(code_block)

        return match.group(0) if match else None

    # 【修改3】std_proj_funcs 改为 Java 版：分析 import 并过滤 JDK/第三方库方法
    # Python 版用 ast + exec + inspect；Java 无法动态导入，改为解析 import + 常用 JDK 方法名列表
    def std_java_funcs(self, code, fname):
        """分析 Java 文件的 import，返回应过滤的标准库/第三方库方法名集合"""
        std_libs = []
        std_funcs = set()
        # 常用 JDK 方法名（作为 reference 时通常来自标准库）
        JDK_COMMON_METHODS = {
            "println", "print", "printf", "length", "size", "add", "get", "put", "set",
            "equals", "hashCode", "toString", "substring", "trim", "split", "toLowerCase",
            "toUpperCase", "charAt", "indexOf", "contains", "isEmpty", "clear", "remove",
            "iterator", "next", "hasNext", "parseInt", "parseDouble", "valueOf",
            "append", "insert", "replace", "format", "close", "flush", "read", "write",
            "main", "run", "start", "join", "sleep", "wait", "notify", "notifyAll",
            "compareTo", "compare", "sort", "reverse", "copyOf", "asList", "of",
        }
        std_funcs.update(JDK_COMMON_METHODS)
        # 解析 import 语句
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('import '):
                rest = line[7:].rstrip(';').strip()
                if rest.startswith(('java.', 'javax.', 'org.junit', 'org.mockito', 'junit.')):
                    std_libs.append(rest.split('.')[0])
        return std_funcs, std_libs
                    

    def get_tags(self, fname, rel_fname):
        # Check if the file is in the cache and if the modification time has not changed
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []
        # miss!
        data = list(self.get_tags_raw(fname, rel_fname))
        return data
    
    def get_tags_raw(self, fname, rel_fname):
        ref_fname_lst = rel_fname.split('/')
        s = deepcopy(self.structure)
        for fname_part in ref_fname_lst:
            if fname_part not in s:
                return []
            s = s[fname_part]
        # 【修改4】structure 由 utils copy 的 create_structure 提供，已为 Java 解析结果
        structure_classes = {item['name']: item for item in s.get('classes', [])}
        structure_class_methods = {}
        for cls in s.get('classes', []):
            for item in cls.get('methods', []):
                structure_class_methods[item['name']] = item

        structure_all_funcs = {**structure_class_methods}
        lang = filename_to_lang(fname)
        if not lang:
            return
        # 【修改5】tree-sitter 对 .java 返回 "java"，get_language/get_parser 均支持
        language = get_language(lang)
        parser = get_parser(lang)

        # 【修改6】tree-sitter 查询从 Python 语法改为 Java 语法
        # Python: class_definition, function_definition, call (function/attribute)
        # Java: class_declaration, method_declaration, constructor_declaration, method_invocation
        query_scm = """
            (class_declaration
            name: (identifier) @name.definition.class) @definition.class

            (method_declaration
            name: (identifier) @name.definition.function) @definition.function

            (constructor_declaration
            name: (identifier) @name.definition.function) @definition.function

            (method_invocation
            name: (identifier) @name.reference.call) @reference.call
            """
        # if not query_scm.exists():
        #     return
        # query_scm = query_scm.read_text()

        code = _read_source_text(fname)
        codelines = code.splitlines(keepends=True)

        # 【修改7】移除 Python 专用预处理，仅保留 BOM 处理（Java 源码通用）
        code = code.replace('\ufeff', '')

        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))

        # 【修改8】使用 Java 版标准库/第三方库过滤
        try:
            std_funcs, std_libs = self.std_java_funcs(code, fname)
        except Exception:
            std_funcs, std_libs = set(), []

        # 【修改9】Java 无 builtins 模块，使用常见 Java 关键字/类型名过滤
        java_builtins = {
            "String", "Integer", "Double", "Boolean", "Object", "List", "Map", "Set",
            "System", "Math", "Arrays", "Collections", "Optional", "void", "null",
            "true", "false", "this", "super", "new", "if", "else", "for", "while",
        }

        # Run the tags queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)
        captures = list(captures)

        saw = set()
        for node, tag in captures:
            if tag.startswith("name.definition."):
                kind = "def"
            elif tag.startswith("name.reference."):
                kind = "ref"
            else:
                continue

            saw.add(kind)
            cur_cdl = codelines[node.start_point[0]]
            # 【修改10】Java 类声明含 "class "，方法为 method_declaration/constructor_declaration
            category = 'class' if 'class ' in cur_cdl else 'function'
            tag_name = node.text.decode("utf-8")

            # 【修改11】仅保留项目内定义，过滤 JDK/第三方库
            if tag_name in std_funcs:
                continue
            elif tag_name in std_libs:
                continue
            elif tag_name in java_builtins:
                continue

            # 【注解数据来源】annotations 来自 utils.parse_java_file() 写入 structure 的
            # classes[].annotations / methods[].annotations。此前仅用「从声明行向上扫文本」会漏掉
            # tree-sitter 把声明起点算在 @Override 行上的情况；已在 utils 中改为 modifiers AST ∪ 文本扫描。
            if category == 'class':
                cls_info = structure_classes.get(tag_name, {})
                class_functions = [item['name'] for item in cls_info.get('methods', [])]
                line_nums = [cls_info.get('start_line', 0), cls_info.get('end_line', 0)] if kind == 'def' else [node.start_point[0], node.end_point[0]]

                result = Tag(
                    rel_fname=rel_fname,
                    fname=fname,
                    name=tag_name,
                    kind=kind,
                    category=category,
                    info='\n'.join(class_functions),
                    line=line_nums,
                    comment=cls_info.get('comment', ''),
                    annotations=cls_info.get('annotations', []),
                )

            elif category == 'function':
                func_info = structure_all_funcs.get(tag_name, {})
                if kind == 'def':
                    line_nums = [func_info.get('start_line', node.start_point[0]), func_info.get('end_line', node.end_point[0])]
                else:
                    line_nums = [node.start_point[0], node.end_point[0]]

                result = Tag(
                    rel_fname=rel_fname,
                    fname=fname,
                    name=tag_name,
                    kind=kind,
                    category=category,
                    info=func_info.get('text', [''])[0] if isinstance(func_info.get('text'), list) else func_info.get('text', ''),
                    line=line_nums,
                    comment=func_info.get('comment', ''),
                    annotations=func_info.get('annotations', []),
                )

            yield result

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # We saw defs, without any refs
        # Some tags files only provide defs (cpp, for example)
        # Use pygments to backfill refs

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except ClassNotFound:
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
                category='function',
                info='none',
                comment='',
                annotations=[]
            )

    def get_ranked_tags(self, other_fnames, mentioned_fnames):
        # defines = defaultdict(set)
        # references = defaultdict(list)
        # definitions = defaultdict(set)
        
        tags_of_files = list()

        personalization = dict()

        fnames = set(other_fnames)
        # chat_rel_fnames = set()

        fnames = sorted(fnames)

        # Default personalization for unspecified files is 1/num_nodes
        # https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank
        personalize = 10 / len(fnames)

        for fname in tqdm(fnames):
            if not Path(fname).is_file():
                if fname not in self.warned_files:
                    if Path(fname).exists():
                        self.io.tool_error(
                            f"Code graph can't include {fname}, it is not a normal file"
                        )
                    else:
                        self.io.tool_error(f"Code graph can't include {fname}, it no longer exists")

                self.warned_files.add(fname)
                continue

            # dump(fname)
            rel_fname = self.get_rel_fname(fname)

            # if fname in chat_fnames:
            #     personalization[rel_fname] = personalize
            #     chat_rel_fnames.add(rel_fname)

            if fname in mentioned_fnames:
                personalization[rel_fname] = personalize
            
            tags = list(self.get_tags(fname, rel_fname))

            tags_of_files.extend(tags)

            if tags is None:
                continue

        return tags_of_files
    

    def render_tree(self, abs_fname, rel_fname, lois):
        key = (rel_fname, tuple(sorted(lois)))

        if key in self.tree_cache:
            return self.tree_cache[key]

        # code = self.io.read_text(abs_fname) or ""
        code = _read_source_text(abs_fname) or ""

        if not code.endswith("\n"):
            code += "\n"

        context = TreeContext(
            rel_fname,
            code,
            color=False,
            line_number=False,
            child_context=False,
            last_line=False,
            margin=0,
            mark_lois=False,
            loi_pad=0,
            # header_max=30,
            show_top_of_file_parent_scope=False,
        )

        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        self.tree_cache[key] = res
        return res

    def to_tree(self, tags, chat_rel_fnames):
        if not tags:
            return ""

        tags = [tag for tag in tags if tag[0] not in chat_rel_fnames]
        tags = sorted(tags)

        cur_fname = None
        cur_abs_fname = None
        lois = None
        output = ""

        # add a bogus tag at the end so we trip the this_fname != cur_fname...
        dummy_tag = (None,)
        for tag in tags + [dummy_tag]:
            this_rel_fname = tag[0]

            # ... here ... to output the final real entry in the list
            if this_rel_fname != cur_fname:
                if lois is not None:
                    output += "\n"
                    output += cur_fname + ":\n"
                    output += self.render_tree(cur_abs_fname, cur_fname, lois)
                    lois = None
                elif cur_fname:
                    output += "\n" + cur_fname + "\n"
                if type(tag) is Tag:
                    lois = []
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None:
                lois.append(tag.line)

        # truncate long lines, in case we get minified js or something else crazy
        output = "\n".join([line[:100] for line in output.splitlines()]) + "\n"

        return output


    def find_src_files(self, directory):
        if not os.path.isdir(directory):
            return [directory]

        src_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                full = os.path.join(root, file)
                rel = os.path.relpath(full, self.root).replace("\\", "/")
                if not is_localizable_java_source_path(rel):
                    continue
                src_files.append(full)
        return src_files
    

    def find_files(self, dir):
        """
        收集待建图的 .java 文件路径。

        ``dir`` 可为：
          - 单一路径字符串（仓库根或子目录）；
          - 路径列表（如 worktree 根列表）。

        若误传单个 str，必须包在列表里；此处对 str/PathLike 做归一化，避免 ``for c in path_str``
        按字符迭代导致永远找不到 .java。
        """
        if isinstance(dir, (str, os.PathLike)):
            roots: List[str] = [os.path.abspath(os.path.normpath(str(dir)))]
        else:
            roots = [os.path.abspath(os.path.normpath(str(x))) for x in dir]

        chat_fnames: List[str] = []

        for fname in roots:
            if Path(fname).is_dir():
                chat_fnames += self.find_src_files(fname)
            else:
                chat_fnames.append(fname)
        
        chat_fnames_new = []
        for item in chat_fnames:
            # 【修改14】仅保留 .java 文件，过滤其他语言
            rel = os.path.relpath(item, self.root).replace("\\", "/")
            if not is_localizable_java_source_path(rel):
                continue
            chat_fnames_new.append(item)

        return chat_fnames_new
    

def get_random_color():
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    res = f"#{r:02x}{g:02x}{b:02x}"
    return res


# ==============================
# Incremental repograph generation
# ==============================


def _run_git(repo_dir: str, args: List[str]) -> str:
    """Run a git command and return stdout."""
    cmd = ["git", "-C", repo_dir] + args
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.stdout


def _git_try_fetch_reset_clean(repo_dir: str, sha: str) -> Tuple[bool, str]:
    """One attempt: fetch sha, reset --hard, clean. net_relaxed on fetch/reset for promisor/TLS."""
    env = _git_clone_env()

    def run_parts(parts: List[str]) -> Tuple[int, str]:
        p = subprocess.run(
            parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        return p.returncode, p.stdout

    fetch_cmd = ["git", *_git_clone_http_opts(), "-C", repo_dir, "fetch", "--no-tags", "origin", sha]
    rc, out = run_parts(fetch_cmd)
    if rc != 0:
        return False, f"fetch rc={rc}\n{out}"
    reset_cmd = ["git", *_git_clone_http_opts(), "-C", repo_dir, "reset", "--hard", "-q", sha]
    rc, out = run_parts(reset_cmd)
    if rc != 0:
        return False, f"reset rc={rc}\n{out}"
    clean_cmd = ["git", "-C", repo_dir, "clean", "-fdq"]
    rc, out = run_parts(clean_cmd)
    if rc != 0:
        return False, f"clean rc={rc}\n{out}"
    return True, ""


def _git_checkout_commit(repo_dir: str, sha: str) -> None:
    # Fetch commit (no --depth 1): shallow single-commit fetches break `git diff old_sha new_sha`
    # merge-base for incremental Multi-SWE-bench builds.
    delays_sec = (0, 8, 24, 60)
    logs: List[str] = []
    for attempt, delay in enumerate(delays_sec, start=1):
        if delay:
            time.sleep(delay)
        ok, frag = _git_try_fetch_reset_clean(repo_dir, sha)
        if ok:
            return
        logs.append(f"checkout attempt {attempt}/{len(delays_sec)}:\n{frag.strip()}")
    raise RuntimeError(
        "git fetch/reset failed after retries (unstable network or broken repo cache). "
        "If this repo was cloned with an older script using --filter=blob:none, delete its folder under "
        f"repo-cache and rerun.\n{repo_dir}\n---\n" + "\n---\n".join(logs)
    )


def _git_diff_changed_java_files(repo_dir: str, old_sha: str, new_sha: str) -> List[str]:
    """
    Return changed Java files relative to repo_dir.
    Uses `git diff --name-status` to support add/modify/delete.
    """
    out = _run_git(repo_dir, ["diff", "--name-status", old_sha, new_sha, "--", "*.java"])
    changed: List[str] = []
    # Each line: <status>\t<path> (or 'Rxx\told\tnew')
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) == 1:
            continue
        status = parts[0]
        if status.startswith("R") and len(parts) >= 3:
            # rename: old -> new, we use new path
            path = parts[-1]
        else:
            path = parts[1] if len(parts) >= 2 else ""
        if path and path.endswith(".java"):
            changed.append(path)
    return changed


def _read_tags_json(tags_json_path: str) -> List[Dict[str, Any]]:
    tags: List[Dict[str, Any]] = []
    if not os.path.exists(tags_json_path):
        return tags
    with open(tags_json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tags.append(json.loads(line))
    return tags


def _write_tags_json(tags: List[Dict[str, Any]], tags_json_path: str) -> None:
    Path(tags_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(tags_json_path, "w", encoding="utf-8") as f:
        for tag in tags:
            f.write(json.dumps(tag, ensure_ascii=False) + "\n")


def _filter_tags_by_changed_files(
    old_tags: List[Dict[str, Any]],
    repo_name: str,
    changed_rel_files: List[str],
) -> List[Dict[str, Any]]:
    """
    Remove all tags that belong to changed Java files.
    We compare by suffix on `rel_fname` because rel_fname may include repo prefix.
    """
    changed_suffixes = [p.replace("\\", "/").lstrip("/") for p in changed_rel_files]
    changed_suffixes_set = set(changed_suffixes)
    kept: List[Dict[str, Any]] = []
    for t in old_tags:
        rel_fname = (t.get("rel_fname") or "").replace("\\", "/")
        # Heuristic: if rel_fname ends with changed file path, drop.
        drop = False
        for suf in changed_suffixes_set:
            if suf and rel_fname.endswith(suf):
                drop = True
                break
        if not drop:
            kept.append(t)
    return kept


def _extract_tags_for_changed_files(
    code_graph: "CodeGraph",
    abs_changed_java_files: List[str],
) -> List[Dict[str, Any]]:
    new_tags: List[Dict[str, Any]] = []
    for abs_fp in abs_changed_java_files:
        if not abs_fp or not str(abs_fp).endswith(".java"):
            continue
        if not os.path.exists(abs_fp):
            # Deleted file: we only need to remove old tags, no need to re-parse.
            continue
        rel_fname = code_graph.get_rel_fname(abs_fp)
        for tag in code_graph.get_tags_raw(abs_fp, rel_fname):
            # The get_tags_raw yields Tag objects; convert to dict.
            new_tags.append(
                {
                    "fname": tag.fname,
                    "rel_fname": tag.rel_fname,
                    "line": tag.line,
                    "name": tag.name,
                    "kind": tag.kind,
                    "category": tag.category,
                    "info": tag.info,
                    "comment": getattr(tag, "comment", ""),
                    "annotations": getattr(tag, "annotations", []),
                }
            )
    return new_tags


def update_graph(old_graph, diff_patch, *, old_tags_json: str, new_tags_json: str, output_graph_json: str, **kwargs):
    """
    Incrementally build next-version repograph by updating tags.json
    and then rebuilding graph.pkl from merged tags.

    Parameters:
      old_graph: previous version networkx graph (kept for API compatibility)
      diff_patch: git diff --name-status output OR a list of changed file paths

    Required kwargs:
      repo_dir: checked-out repo clone dir (for resolving abs changed files)
      repo_work_root: parent root used by rel_fname prefix in legacy paths
      code_graph_root: CodeGraph root (worktree path when using per-version worktrees; else repo_work_root)
      repo_name: repo folder name
      prev_sha: previous commit sha
      new_sha: new commit sha
      code_graph_for_tags_kwargs: dict (enable_inheritance_edges/structure_only_files)
    """
    repo_dir = kwargs.get("repo_dir")
    repo_work_root = kwargs.get("repo_work_root")
    repo_name = kwargs.get("repo_name")
    code_graph_root = kwargs.get("code_graph_root") or repo_work_root

    if isinstance(diff_patch, list):
        changed_rel_files = diff_patch
    else:
        # parse name-status lines
        changed_rel_files = []
        out = str(diff_patch or "")
        for line in out.splitlines():
            parts = line.strip().split("\t")
            if not parts or len(parts) < 2:
                continue
            status = parts[0]
            if status.startswith("R") and len(parts) >= 3:
                path = parts[-1]
            else:
                path = parts[1]
            if path and path.endswith(".java"):
                changed_rel_files.append(path)

    changed_rel_files = sorted(set(changed_rel_files))
    changed_rel_files = [
        p for p in changed_rel_files if is_localizable_java_source_path(p)
    ]
    abs_changed_java_files = [os.path.join(repo_dir, p) for p in changed_rel_files if p.endswith(".java")]

    old_tags = _read_tags_json(old_tags_json)
    kept_old_tags = _filter_tags_by_changed_files(old_tags, repo_name, changed_rel_files)

    # Build tags only for changed files (graph root must match full-build root: worktree vs shared).
    code_graph_for_changed = CodeGraph(
        root=code_graph_root,
        enable_inheritance_edges=False,
        structure_only_files=set(abs_changed_java_files),
    )
    new_tags = _extract_tags_for_changed_files(code_graph_for_changed, abs_changed_java_files)

    merged_tags = kept_old_tags + new_tags
    _write_tags_json(merged_tags, new_tags_json)

    # =============================
    # Incremental graph.pkl update
    # =============================
    # We update the existing graph by:
    # 1) Removing nodes belonging to changed files (and their incident edges)
    # 2) Adding nodes for changed files
    # 3) Re-adding "contains" edges for changed class tags
    #
    # Note: current node id uses `tag.name` (not unique across files). This can cause
    # collisions even in full rebuild. Here we keep compatibility, and do best-effort
    # file-based node removal using node attributes `fname` / `rel_fname`.

    G = old_graph
    if G is None:
        old_graph_pkl = kwargs.get("old_graph_pkl")
        if old_graph_pkl and os.path.exists(old_graph_pkl):
            with open(old_graph_pkl, "rb") as f:
                G = pickle.load(f)

    if G is None:
        # Full rebuild fallback if no previous graph is available.
        code_graph_for_graph = CodeGraph(
            root=code_graph_root,
            enable_inheritance_edges=False,
            structure_only_files=set(),
        )
        G = code_graph_for_graph.tag_to_graph(
            [
                Tag(
                    t["rel_fname"],
                    t["fname"],
                    t["line"],
                    t["name"],
                    t["kind"],
                    t["category"],
                    t["info"],
                    t.get("comment", ""),
                    t.get("annotations", []),
                )
                for t in merged_tags
            ]
        )
    else:
        # Remove nodes from changed files
        changed_suffixes = {p.replace("\\", "/").lstrip("/") for p in changed_rel_files}
        to_remove = []
        for node_id, data in list(G.nodes(data=True)):
            fname = (data.get("fname") or data.get("file") or "").replace("\\", "/")
            rel_fname = (data.get("rel_fname") or "").replace("\\", "/")
            hit = False
            for suf in changed_suffixes:
                if suf and (fname.endswith(suf) or rel_fname.endswith(suf)):
                    hit = True
                    break
            if hit:
                to_remove.append(node_id)
        for n in to_remove:
            if n in G:
                G.remove_node(n)

        # Add nodes for changed files from newly parsed tags
        new_tag_objs = [
            Tag(
                t["rel_fname"],
                t["fname"],
                t["line"],
                t["name"],
                t["kind"],
                t["category"],
                t["info"],
                t.get("comment", ""),
                t.get("annotations", []),
            )
            for t in new_tags
        ]
        for tag in new_tag_objs:
            node_id = make_node_id(tag.rel_fname, tag.name, tag.kind, tag.category, tag.line)
            G.add_node(
                node_id,
                symbol_name=tag.name,
                category=tag.category,
                info=tag.info,
                fname=tag.fname,
                rel_fname=tag.rel_fname,
                line=tag.line,
                kind=tag.kind,
                comment=getattr(tag, "comment", ""),
                annotations=getattr(tag, "annotations", []),
            )

        # Re-add contains edges for changed class tags
        def_nodes_by_name = {}
        def_func_nodes_by_file_name = {}
        for tag in new_tag_objs:
            nid = make_node_id(tag.rel_fname, tag.name, tag.kind, tag.category, tag.line)
            if tag.kind == "def":
                def_nodes_by_name.setdefault(tag.name, []).append(nid)
                if tag.category == "function":
                    key = (tag.rel_fname, tag.name)
                    def_func_nodes_by_file_name.setdefault(key, []).append(nid)

        for tag in new_tag_objs:
            if tag.category == "class" and tag.info:
                class_nid = make_node_id(tag.rel_fname, tag.name, tag.kind, tag.category, tag.line)
                for f_name in str(tag.info).split("\n"):
                    fn = f_name.strip()
                    if not fn:
                        continue
                    targets = def_func_nodes_by_file_name.get((tag.rel_fname, fn), [])
                    if not targets:
                        targets = def_nodes_by_name.get(fn, [])
                    for dst in targets:
                        G.add_edge(class_nid, dst, edge_type="contains")

        # Optional: update inheritance edges (off by default for speed)
        if kwargs.get("enable_inheritance_edges", False):
            code_graph_for_graph = CodeGraph(
                root=code_graph_root,
                enable_inheritance_edges=True,
                structure_only_files=None,
            )
            code_graph_for_graph._add_inheritance_edges(G)
    with open(output_graph_json, "wb") as f:
        pickle.dump(G, f)

    return G


def _checkout_and_build_first(
    repo_clone_dir: str,
    repo_work_root: str,
    repo_name: str,
    sha: str,
    version_output_dir: str,
    *,
    use_git_checkout: bool = True,
    code_graph_root: str,
    java_scan_roots: List[str],
):
    """
    use_git_checkout: False when using a per-version git worktree (already at correct commit).
    code_graph_root: CodeGraph root for rel_fname (use worktree path for version-isolated layout).
    java_scan_roots: Directories to scan for .java (usually one worktree dir or main clone dir).
    """
    if use_git_checkout:
        _git_checkout_commit(repo_clone_dir, sha)
    code_graph = CodeGraph(root=code_graph_root, enable_inheritance_edges=True)
    for root in java_scan_roots:
        ap = os.path.abspath(os.path.normpath(str(root)))
        if not os.path.isdir(ap):
            raise RuntimeError(
                "Repograph build: java scan root is missing or not a directory: "
                f"{root!r} (resolved {ap!r}). Check git worktree / clone layout."
            )
    chat_fnames_new = code_graph.find_files(java_scan_roots)
    if not chat_fnames_new:
        raise RuntimeError(
            "Repograph build: no non-test .java files found under "
            f"java_scan_roots={java_scan_roots!r}, code_graph_root={code_graph_root!r}. "
            "Common causes: empty worktree after `git worktree add`, wrong paths, "
            "or all sources under paths filtered by is_localizable_java_source_path."
        )
    tags, G = code_graph.get_code_graph(chat_fnames_new)
    if not tags:
        raise RuntimeError(
            "Repograph build: get_code_graph returned no tags despite "
            f"{len(chat_fnames_new)} candidate .java files (first: {chat_fnames_new[0]!r}). "
            "See get_ranked_tags / parse errors above."
        )
    Path(version_output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(version_output_dir, "graph.pkl"), "wb") as f:
        pickle.dump(G, f)
    with open(os.path.join(version_output_dir, "tags.json"), "w", encoding="utf-8") as f:
        for tag in tags:
            f.write(
                json.dumps(
                    {
                        "fname": tag.fname,
                        "rel_fname": tag.rel_fname,
                        "line": tag.line,
                        "name": tag.name,
                        "kind": tag.kind,
                        "category": tag.category,
                        "info": tag.info,
                        "comment": tag.comment,
                        "annotations": getattr(tag, "annotations", []),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _lookup_local_version_map(
    local_version_roots: Optional[Dict[str, Any]], repo_key: str
) -> Optional[Dict[str, str]]:
    """
    JSON top-level key may be ``org/repo`` (dataset key) or ``org__repo`` for convenience.
    Inner keys must match version_id: ``{ref}-{sha[:7]}`` (same as directory names under output_root).
    """
    if not local_version_roots:
        return None
    org, _, repo = repo_key.partition("/")
    if not repo:
        return None
    for k in (repo_key, f"{org}__{repo}"):
        v = local_version_roots.get(k)
        if isinstance(v, dict) and v:
            return {str(vid): str(path) for vid, path in v.items()}
    return None


def build_repographs_from_multiswebench_java_dataset(
    dataset_jsonl_dir: str,
    output_root: str,
    repo_cache_root: str,
    github_clone_prefix: str = "",
    clone_fallback_direct: bool = True,
    use_version_worktrees: bool = True,
    only_repo_key: Optional[str] = None,
    local_version_roots: Optional[Dict[str, Any]] = None,
):
    """
    Pull multi-swe-bench Java history versions from network and build repographs.
    Uses `record.base.sha` and iterates in order of appearance for each (org, repo).

    If use_version_worktrees (default): each dataset version gets a detached git worktree under
    `<cache>/<org>__<repo>/versions/<safe_version_id>/`, plus `repo_versions_manifest.json` mapping
    version_id -> sha/path. Repograph output stays `<output>/<repo>/<ref-sha7>/` (localization unchanged).

    If False (--shared-repo-checkout): legacy single clone with repeated checkout (cannot parallelize).

    If ``local_version_roots`` is set and contains an entry for a repo, that repo skips all git
    clone/worktree/fetch: each version must map to an existing local directory; each version is
    built with a full CodeGraph pass (no git-diff incremental updates).
    """
    dataset_dir = Path(dataset_jsonl_dir)
    out_root = Path(output_root)
    cache_root = Path(repo_cache_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Collect versions per repo.
    versions_by_repo: Dict[str, List[Tuple[str, str]]] = {}  # repo_key -> [(sha, ref)]
    dataset_files = sorted(dataset_dir.glob("*_dataset.jsonl"))
    if not dataset_files:
        raise FileNotFoundError(f"No *_dataset.jsonl files in: {dataset_dir}")

    for df in dataset_files:
        with open(df, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                org = rec.get("org", "")
                repo = rec.get("repo", "")
                base = rec.get("base", {}) or {}
                sha = base.get("sha", "")
                ref = base.get("ref", "") or base.get("label", "") or sha[:7]
                if not org or not repo or not sha:
                    continue
                repo_key = f"{org}/{repo}"
                versions_by_repo.setdefault(repo_key, [])
                if not any(v[0] == sha for v in versions_by_repo[repo_key]):
                    versions_by_repo[repo_key].append((sha, str(ref)))

    # 【测试-仅单仓库】只构建指定 org/repo；为 None 时保持原行为：遍历数据集中出现的全部仓库（多仓库）。
    if only_repo_key:
        key = only_repo_key.strip().replace("\\", "/")
        if key not in versions_by_repo:
            avail = sorted(versions_by_repo.keys())
            raise FileNotFoundError(
                f"only_repo_key={key!r} not in dataset. Available repo keys ({len(avail)}): {avail[:40]}"
                + (" ..." if len(avail) > 40 else "")
            )
        versions_by_repo = {key: versions_by_repo[key]}

    for repo_key, versions in versions_by_repo.items():
        org, repo = repo_key.split("/", 1)
        repo_work_root = cache_root / f"{org}__{repo}"
        repo_clone_dir = repo_work_root / repo  # clone into <work_root>/<repo>
        repo_work_root.mkdir(parents=True, exist_ok=True)

        local_map = _lookup_local_version_map(local_version_roots, repo_key)

        if not local_map:
            if repo_clone_dir.exists():
                chk = subprocess.run(
                    ["git", "-C", str(repo_clone_dir), "rev-parse", "--git-dir"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if chk.returncode != 0:
                    shutil.rmtree(repo_clone_dir, ignore_errors=True)

            canonical = f"https://github.com/{org}/{repo}.git"
            mirror_url = _github_https_clone_url(org, repo, github_clone_prefix)
            clone_urls: List[str] = []
            if github_clone_prefix and mirror_url != canonical:
                clone_urls.append(mirror_url)
            if clone_fallback_direct or not github_clone_prefix:
                if canonical not in clone_urls:
                    clone_urls.append(canonical)
            elif not clone_urls:
                clone_urls.append(canonical)

            if not repo_clone_dir.exists():
                _git_clone_no_checkout(clone_urls, str(repo_clone_dir))
        else:
            missing: List[str] = []
            bad_path: List[str] = []
            for sha_i, ref_i in versions:
                vid = f"{ref_i}-{sha_i[:7]}"
                p = local_map.get(vid)
                if not p:
                    missing.append(vid)
                    continue
                rp = Path(p).expanduser()
                if not rp.is_dir():
                    bad_path.append(f"{vid} -> {p!r}")
            if missing:
                raise FileNotFoundError(
                    f"local_version_roots for {repo_key!r}: missing version_id keys {missing[:20]}"
                    + (" ..." if len(missing) > 20 else "")
                    + f". Each key must be exactly '{{ref}}-{{sha[:7]}}' as in the dataset order."
                )
            if bad_path:
                raise FileNotFoundError(
                    f"local_version_roots for {repo_key!r}: not a directory: {bad_path[:10]}"
                )

        prev_sha: Optional[str] = None
        prev_output_dir: Optional[str] = None
        manifest_versions: Dict[str, Dict[str, Any]] = {}

        for idx, (sha, ref) in enumerate(versions):
            version_id = f"{ref}-{sha[:7]}"
            safe_v = safe_version_dirname(version_id)
            if local_map:
                root_path = Path(local_map[version_id]).expanduser().resolve()
                code_graph_root = str(root_path)
                java_scan_roots = [str(root_path)]
                worktree_rel = f"local:{code_graph_root}"
                use_git_checkout = False
                repo_dir_for_files = str(root_path)
            elif use_version_worktrees:
                wt_path = repo_work_root / "versions" / safe_v
                ensure_git_worktree(Path(repo_clone_dir), wt_path, sha)
                code_graph_root = str(wt_path)
                java_scan_roots = [str(wt_path)]
                worktree_rel = f"versions/{safe_v}"
                use_git_checkout = False
                repo_dir_for_files = str(wt_path)
            else:
                code_graph_root = str(repo_work_root)
                java_scan_roots = [str(repo_clone_dir)]
                worktree_rel = repo
                use_git_checkout = True
                repo_dir_for_files = str(repo_clone_dir)

            manifest_versions[version_id] = {
                "sha": sha,
                "ref": ref,
                "worktree_rel": worktree_rel,
                "repograph_version_dir": version_id,
            }

            version_output_dir = out_root / repo / version_id
            tags_json_path = str(version_output_dir / "tags.json")
            graph_pkl_path = str(version_output_dir / "graph.pkl")
            if os.path.exists(graph_pkl_path) and os.path.exists(tags_json_path):
                prev_sha = sha
                prev_output_dir = str(version_output_dir)
                continue

            if local_map or idx == 0 or prev_sha is None or prev_output_dir is None:
                _checkout_and_build_first(
                    repo_clone_dir=str(repo_clone_dir),
                    repo_work_root=str(repo_work_root),
                    repo_name=repo,
                    sha=sha,
                    version_output_dir=str(version_output_dir),
                    use_git_checkout=use_git_checkout,
                    code_graph_root=code_graph_root,
                    java_scan_roots=java_scan_roots,
                )
            else:
                if use_git_checkout:
                    _git_checkout_commit(str(repo_clone_dir), sha)
                changed_java_files_rel = _git_diff_changed_java_files(
                    str(repo_clone_dir), old_sha=prev_sha, new_sha=sha
                )
                update_graph(
                    None,
                    changed_java_files_rel,
                    old_tags_json=str(Path(prev_output_dir) / "tags.json"),
                    new_tags_json=str(Path(version_output_dir) / "tags.json"),
                    output_graph_json=str(Path(version_output_dir) / "graph.pkl"),
                    old_graph_pkl=str(Path(prev_output_dir) / "graph.pkl"),
                    repo_dir=repo_dir_for_files,
                    repo_work_root=str(repo_work_root),
                    repo_name=repo,
                    code_graph_root=code_graph_root,
                )

            prev_sha = sha
            prev_output_dir = str(version_output_dir)

        if local_map:
            layout_out = "local_precloned"
        elif use_version_worktrees:
            layout_out = "git_worktrees"
        else:
            layout_out = "shared_checkout"
        write_repo_versions_manifest(
            repo_work_root,
            org,
            repo,
            main_clone_rel=repo,
            versions=manifest_versions,
            layout=layout_out,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct repograph from local repo or Multi-SWE-bench history.")
    parser.add_argument("--dataset-jsonl-dir", type=str, default=None, help="Directory containing *_dataset.jsonl for multi-swe-bench Java.")
    parser.add_argument("--dataset-jsonl-file", type=str, default=None, help="Single jsonl file containing multi-swe-bench Java instances.")
    parser.add_argument("--output-root", type=str, default="resultgraphs/new-multi-swe-bench", help="Where to write repographs.")
    _default_repo_cache = os.environ.get("REPO_CACHE_ROOT", "resultgraphs/repo_cache")
    parser.add_argument(
        "--repo-cache-root",
        type=str,
        default=_default_repo_cache,
        help="Where to clone git repos (default: env REPO_CACHE_ROOT or resultgraphs/repo_cache).",
    )
    parser.add_argument(
        "--github-clone-prefix",
        type=str,
        default="",
        help=(
            "Optional mirror prefix prepended to https://github.com/org/repo.git (e.g. https://ghproxy.net). "
            "If empty, uses env GITHUB_CLONE_PREFIX. Example: --github-clone-prefix https://ghproxy.net"
        ),
    )
    parser.add_argument(
        "--no-github-clone-fallback",
        action="store_true",
        help="With --github-clone-prefix, do not fall back to direct https://github.com/... if the mirror fails.",
    )
    parser.add_argument(
        "--shared-repo-checkout",
        action="store_true",
        help="Legacy: one clone, checkout each version in place. Default: separate git worktree per version under versions/.",
    )
    # 【测试-仅单仓库】与下面 dataset 模式联用：只处理一个 org/repo，避免一次跑完全部仓库。不设则=原版多仓库遍历。
    parser.add_argument(
        "--only-repo",
        type=str,
        default=None,
        metavar="ORG/REPO",
        help="Only build repograph for one repo key from the jsonl (e.g. apache/dubbo). Omit to process all repos in the dataset.",
    )
    parser.add_argument(
        "--local-version-roots-json",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "JSON: { \"org/repo\" | \"org__repo\": { \"<ref>-<sha7>\": \"/abs/path/to/checkout\", ... }, ... }. "
            "For each listed repo, skips git clone/worktree; each version_id must match dataset "
            "(ref + '-' + first 7 hex of base.sha). Each path must be an existing directory with sources. "
            "Repos omitted from this file still use network git as usual."
        ),
    )

    # Legacy mode
    parser.add_argument("legacy_repo_dir", nargs="?", default=None, help="Legacy: local directory to build tags.json/graph.pkl in current working dir.")

    args = parser.parse_args()

    # New dataset-driven mode
    if args.dataset_jsonl_dir or args.dataset_jsonl_file:
        if args.dataset_jsonl_file and args.dataset_jsonl_dir:
            raise SystemExit("Only one of --dataset-jsonl-dir / --dataset-jsonl-file can be provided.")

        dataset_dir = args.dataset_jsonl_dir or str(Path(args.dataset_jsonl_file).parent)
        gh_prefix = (args.github_clone_prefix or "").strip().rstrip("/")
        if not gh_prefix:
            gh_prefix = (os.environ.get("GITHUB_CLONE_PREFIX") or "").strip().rstrip("/")
        use_wt = not args.shared_repo_checkout
        if os.getenv("JLOC_VERSION_WORKTREES", "").strip().lower() in ("0", "false", "no"):
            use_wt = False
        local_roots: Optional[Dict[str, Any]] = None
        if args.local_version_roots_json:
            lp = Path(args.local_version_roots_json).expanduser()
            if not lp.is_file():
                raise FileNotFoundError(
                    f"--local-version-roots-json not a file: {lp}\n"
                    "Create that JSON (see repograph/examples/jloc_local_roots.jackson-databind.template.json) "
                    "or pass a path that exists."
                )
            local_roots = json.loads(lp.read_text(encoding="utf-8"))
            if not isinstance(local_roots, dict):
                raise ValueError("--local-version-roots-json must be a JSON object at the top level")
        build_repographs_from_multiswebench_java_dataset(
            dataset_jsonl_dir=dataset_dir,
            output_root=args.output_root,
            repo_cache_root=args.repo_cache_root,
            github_clone_prefix=gh_prefix,
            clone_fallback_direct=not args.no_github_clone_fallback,
            use_version_worktrees=use_wt,
            only_repo_key=(args.only_repo.strip() if args.only_repo else None),
            local_version_roots=local_roots,
        )
    else:
        if not args.legacy_repo_dir:
            raise SystemExit("Usage: python repograph/construct_graph.py <repo_dir> OR --dataset-jsonl-dir <dir>")

        dir_name = args.legacy_repo_dir
        code_graph = CodeGraph(root=dir_name)
        chat_fnames_new = code_graph.find_files([dir_name])
        tags, G = code_graph.get_code_graph(chat_fnames_new)

        with open(f'{os.getcwd()}/graph.pkl', 'wb') as f:
            pickle.dump(G, f)

        with open(f'{os.getcwd()}/tags.json', 'w', encoding='utf-8') as f:
            for tag in tags:
                line = json.dumps({
                    "fname": tag.fname,
                    'rel_fname': tag.rel_fname,
                    'line': tag.line,
                    'name': tag.name,
                    'kind': tag.kind,
                    'category': tag.category,
                    'info': tag.info,
                    'comment': tag.comment,
                    'annotations': getattr(tag, 'annotations', [])
                }, ensure_ascii=False)
                f.write(line + '\n')

        print(f"🏅 Successfully cached code graph and node tags in {os.getcwd()}")


# ---------------------------------------------------------------------------
# 【说明】单仓库 / 多仓库 与下面注释块的关系
# - 数据集驱动（*_dataset.jsonl）：主入口已支持「仅一个仓库」——加参数
#     --only-repo ORG/REPO
#   例：python repograph/construct_graph.py --dataset-jsonl-dir <dir> --only-repo apache/dubbo
#   不设 --only-repo 时 = 原版行为，会处理 jsonl 中出现的全部仓库（多仓库）。
# - 下面整段为「本地父目录下多个子目录各建一个 repograph」的旧脚本逻辑，已整段注释保留，勿删。
# ---------------------------------------------------------------------------
# 递归循环多个文件夹版本
#    【修改-多仓库遍历生成】运行方式：
#       python repograph/construct_graph.py <repos_root_dir>
#     其中 <repos_root_dir> 为包含多个 repo 子目录的总目录
#     输出固定到 /home/zzz/projects/1.2/Jlocagent/resultgraphs/multi-swe-bench/<repo_name>/ 下
# 假设 CodeGraph 类已在其他地方定义或 import，此处不修改

    # if len(sys.argv) < 2:
    #     raise SystemExit("Usage: python repograph/construct_graph.py <repos_root_dir>")

    # repos_root_dir = sys.argv[1]
    # if not os.path.isdir(repos_root_dir):
    #     raise SystemExit(f"Provided path is not a directory: {repos_root_dir}")

    # # 【修改-输出根目录固定】总输出目录
    # output_root = "/home/zzz/projects/1.2/Jlocagent/resultgraphs/multi-swe-bench"
    # os.makedirs(output_root, exist_ok=True)

    # # 遍历 repos_root_dir 下的所有子目录，每个子目录视为一个独立 repo
    # for entry in sorted(os.listdir(repos_root_dir)):
    #     repo_path = os.path.join(repos_root_dir, entry)
    #     if not os.path.isdir(repo_path):
    #         continue

    #     repo_name = os.path.basename(repo_path.rstrip(os.sep))
    #     repo_output_dir = os.path.join(output_root, repo_name)
    #     os.makedirs(repo_output_dir, exist_ok=True)
    #     print("---------------------------------")
    #     print(f"🚀 Start constructing code graph for repo: {repo_name}")
    #     print(f"   Repo path: {repo_path}")
    #     print(f"   Output dir: {repo_output_dir}")

    #     code_graph = CodeGraph(root=repo_path)
    #     chat_fnames_new = code_graph.find_files([repo_path])
    #     if not chat_fnames_new:
    #         print(f"   ⚠ No files found.")   # 修复了这里不完整的字符串
    #         # 继续处理下一个 repo（原逻辑未明确，这里保持继续循环）
    #         continue

    #     tags, G = code_graph.get_code_graph(chat_fnames_new)

    #     print(f"   Number of nodes: {len(G.nodes)}")
    #     print(f"   Number of edges: {len(G.edges)}")

    #     # 【修改-结果写入按 repo 名命名的子目录】graph.pkl / tags.json
    #     graph_pkl_path = os.path.join(repo_output_dir, "graph.pkl")
    #     tags_json_path = os.path.join(repo_output_dir, "tags.json")

    #     with open(graph_pkl_path, "wb") as f:
    #         pickle.dump(G, f)

    #     # 逐行写入 tags.json（与原逻辑一致，只是写入路径变化）
    #     with open(tags_json_path, "w", encoding="utf-8") as f:
    #         for tag in tags:
    #             line = json.dumps(
    #                 {
    #                     "fname": tag.fname,
    #                     "rel_fname": tag.rel_fname,
    #                     "line": tag.line,
    #                     "name": tag.name,
    #                     "kind": tag.kind,
    #                     "category": tag.category,
    #                     "info": tag.info,
    #                     "comment": tag.comment,
    #                     "annotations": tag.annotations
    #                 },
    #                 ensure_ascii=False,
    #             )
    #             f.write(line + "\n")

    #     print(f"🏅 Successfully cached code graph and node tags in '{repo_output_dir}'")

    # print("---------------------------------")
    # print("✅ All repos under the specified directory have been processed.")
