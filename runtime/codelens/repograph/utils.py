import os
import re
import sys
from pathlib import Path
from tree_sitter_languages import get_parser

_JLOC_ROOT = Path(__file__).resolve().parent.parent
if str(_JLOC_ROOT) not in sys.path:
    sys.path.insert(0, str(_JLOC_ROOT))
from agent.repograph.java_path_filters import is_test_or_benchmark_java_path


# ===== 【最终增强版】同时提取 comment 和 annotations =====
def extract_comment_and_annotations(lines, start_line):
    """
    返回 (comment: str, annotations: list[str])
    - comment: Javadoc + // 注释（字符串）
    - annotations: 所有 @注解 的原始字符串列表（含参数，支持多行）
    """
    if not lines or start_line <= 1:
        return "", []

    idx = start_line - 2
    comment_parts = []
    annotations = []

    while idx >= 0:
        raw_line = lines[idx]
        line = raw_line.strip()

        if not line:
            idx -= 1
            continue

        # 注解（支持单行和多行）
        if line.startswith("@"):
            anno_block = [raw_line.rstrip()]
            idx -= 1
            while idx >= 0:
                next_raw = lines[idx]
                next_line = next_raw.strip()
                if (next_line.startswith("@") or 
                    next_line.startswith(")") or 
                    next_line.startswith("(") or 
                    next_line.startswith(",")):
                    anno_block.append(next_raw.rstrip())
                    idx -= 1
                else:
                    break
            annotations = anno_block[::-1] + annotations
            continue

        # 行注释
        if line.startswith("//"):
            comment_parts.insert(0, raw_line.rstrip())
            idx -= 1
            continue

        # 块注释 / Javadoc
        if line.endswith("*/"):
            block = [raw_line.rstrip()[:-2].strip()]
            idx -= 1
            while idx >= 0:
                l2_raw = lines[idx]
                l2 = l2_raw.strip()
                if not l2:
                    idx -= 1
                    continue
                if l2.startswith("/**") or l2.startswith("/*"):
                    prefix = 3 if l2.startswith("/**") else 2
                    block.append(l2_raw.rstrip()[prefix:].strip())
                    idx -= 1
                    break
                else:
                    cleaned = l2_raw.rstrip().lstrip(" *").strip()
                    if cleaned:
                        block.append(cleaned)
                    idx -= 1
            comment_parts = block[::-1] + comment_parts
            continue

        # 遇到普通代码停止
        break

    comment = "\n".join(comment_parts).strip()

    # 去重并保持顺序
    seen = set()
    unique_annotations = []
    for ann in annotations:
        if ann not in seen:
            seen.add(ann)
            unique_annotations.append(ann)

    return comment, unique_annotations


# 【修复注解-根因说明】仅依赖 extract_comment_and_annotations() 向上扫描时，若 tree-sitter 把
# method_declaration / class_declaration 的 start 放在「@Override」所在行，则 start_line-2 会从
# @Override 的上一行开始扫，永远不会扫到 @Override 这一行本身，导致 annotations 恒为空。
# 因此必须从声明节点的 modifiers 子树（marker_annotation / annotation）用 AST 再取一遍，并与
# 文本扫描结果合并去重。
def extract_modifiers_annotations_java(declaration_node):
    """
    从 class / interface / enum / method / constructor 等声明节点的 modifiers 子树收集
    @annotation 原文（含 @Override、@Deprecated 及带参数的 @Foo(...)）。
    """
    out = []

    def walk_modifiers_subtree(n):
        if n.type in ("annotation", "marker_annotation"):
            try:
                out.append(n.text.decode("utf-8"))
            except Exception:
                pass
            return
        # 【修复注解】部分 grammar 用 modifier 包裹 public / static / @Foo 等
        if n.type in ("modifiers", "modifier"):
            for c in n.children:
                walk_modifiers_subtree(c)
            return
        for c in n.children:
            walk_modifiers_subtree(c)

    for ch in declaration_node.children:
        if ch.type in ("modifiers", "modifier"):
            walk_modifiers_subtree(ch)
        elif ch.type in ("annotation", "marker_annotation"):
            walk_modifiers_subtree(ch)

    # 去重且保持顺序
    seen = set()
    unique = []
    for a in out:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    return unique


def _get_simple_type_name(node):
    if node is None:
        return None
    text = node.text.decode("utf-8")
    return text.split(".")[-1].split("<")[0].strip()


def create_structure(directory_path, only_files=None):
    """
    Create a Java repo structure tree used by RepoGraph.

    Args:
        directory_path: repo root (or work root) passed to CodeGraph.
        only_files: optional iterable of absolute .java file paths. When provided,
            we only parse those files (for incremental repograph updates).
    """
    structure = {}
    repo_name = os.path.basename(directory_path)

    if only_files is not None:
        # Incremental mode: parse only changed files.
        for abs_fp in only_files:
            if not abs_fp or not str(abs_fp).endswith(".java"):
                continue
            abs_fp = str(abs_fp)
            if not os.path.exists(abs_fp):
                continue
            rel_chk = os.path.relpath(abs_fp, directory_path).replace("\\", "/")
            if is_test_or_benchmark_java_path(rel_chk):
                continue

            root_dir = os.path.dirname(abs_fp)
            relative_root = os.path.relpath(root_dir, directory_path)
            if relative_root == ".":
                relative_root = repo_name

            curr_struct = structure
            for part in relative_root.split(os.sep):
                if not part:
                    continue
                curr_struct = curr_struct.setdefault(part, {})

            file_name = os.path.basename(abs_fp)
            class_info, _, file_lines = parse_java_file(abs_fp)
            curr_struct[file_name] = {
                "classes": class_info,
                "functions": [],
                "text": file_lines,
            }

        return structure

    # Full mode: parse all .java under the directory.
    for root, _, files in os.walk(directory_path):
        relative_root = os.path.relpath(root, directory_path)
        if relative_root == ".":
            relative_root = repo_name

        curr_struct = structure
        for part in relative_root.split(os.sep):
            curr_struct = curr_struct.setdefault(part, {})

        for file_name in files:
            if file_name.endswith(".java"):
                file_path = os.path.join(root, file_name)
                rel_chk = os.path.relpath(file_path, directory_path).replace("\\", "/")
                if is_test_or_benchmark_java_path(rel_chk):
                    continue
                class_info, _, file_lines = parse_java_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": [],
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    return structure


def parse_java_file(file_path, file_content=None):
    if file_content is None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return [], [], []

    try:
        parser = get_parser("java")
        tree = parser.parse(bytes(file_content, "utf-8"))
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return [], [], []

    lines = file_content.splitlines()
    class_info = []

    def get_text(node):
        return lines[node.start_point[0]: node.end_point[0] + 1]

    def walk(node):
        if node.type in ("class_declaration", "interface_declaration", "enum_declaration"):
            name_node = node.child_by_field_name("name")
            if not name_node:
                for child in node.children:
                    walk(child)
                return

            cls_name = name_node.text.decode("utf-8")

            # 继承关系
            superclass = None
            interfaces = []
            for child in node.children:
                if child.type == "superclass":
                    t = child.child_by_field_name("type")
                    if t:
                        superclass = _get_simple_type_name(t)
                elif child.type == "super_interfaces":
                    for t in child.children:
                        if t.type == "type_list":
                            for item in t.children:
                                if item.type in ("type_identifier", "generic_type"):
                                    interfaces.append(_get_simple_type_name(item))

            # 【修复注解】类/接口/枚举：comment 仍用向上扫描；annotations = AST(modifiers) ∪ 文本扫描
            comment, ann_from_text = extract_comment_and_annotations(lines, node.start_point[0] + 1)
            ann_from_ast = extract_modifiers_annotations_java(node)
            annotations = list(dict.fromkeys(ann_from_ast + ann_from_text))

            methods = []
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    if child.type in ("method_declaration", "constructor_declaration"):
                        m_name_node = child.child_by_field_name("name")
                        if not m_name_node:
                            continue
                        m_name = m_name_node.text.decode("utf-8")

                        m_comment, m_ann_from_text = extract_comment_and_annotations(
                            lines, child.start_point[0] + 1
                        )
                        # 【修复注解】方法/构造：同上，必须用 modifiers AST，否则 @Override 在声明首行时会丢
                        m_ann_from_ast = extract_modifiers_annotations_java(child)
                        m_annotations = list(dict.fromkeys(m_ann_from_ast + m_ann_from_text))

                        methods.append({
                            "name": m_name,
                            "start_line": child.start_point[0] + 1,
                            "end_line": child.end_point[0] + 1,
                            "text": get_text(child),
                            "comment": m_comment,
                            "annotations": m_annotations,
                        })

            class_info.append({
                "name": cls_name,
                "type": node.type.replace("_declaration", ""),
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "text": get_text(node),
                "methods": methods,
                "comment": comment,
                "annotations": annotations,          # ← 类级别注解
                "superclass": superclass,
                "interfaces": interfaces,
            })

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return class_info, [], lines