"""
Shared path parsing, gold extraction, and scoring for Java issue localization.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from agent.repograph.java_path_filters import is_localizable_java_source_path

HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_fix_patch_gold_files(fix_patch: str) -> Set[str]:
    """
    Extract evaluable file targets from a unified diff.

    Localization runs on the pre-change repository, so only the ``---`` (old)
    side is a valid target.  New files (``/dev/null``), non-Java files, and Java
    tests/benchmarks are outside the Repograph search space and are excluded.
    """
    gold: Set[str] = set()
    if not fix_patch:
        return gold
    for line in fix_patch.splitlines():
        if not line.startswith("--- "):
            continue
        parts = line[4:].strip().split()
        if not parts:
            continue
        f = parts[0]
        if f == "/dev/null":
            continue
        if f.startswith("a/"):
            f = f[2:]
        f = normalize_code_path(f)
        if is_localizable_java_source_path(f):
            gold.add(f)
    return gold


def parse_fix_patch_old_changed_lines(fix_patch: str) -> Dict[str, Set[int]]:
    """
    From a unified diff (`fix_patch`), collect **1-based line numbers in the pre-change (a/) file**
    that are touched: every removed line (`-`) and insertion points for pure additions (`+` without
    consuming an old line). Paths are normalized like `normalize_code_path`.
    Skips `--- /dev/null` (new file only on the b/ side).
    """
    touched: Dict[str, Set[int]] = {}
    if not (fix_patch or "").strip():
        return touched

    lines = fix_patch.splitlines()
    i = 0
    old_path: Optional[str] = None

    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            raw = line[4:].strip().split()
            p = raw[0] if raw else ""
            if "/dev/null" in p:
                old_path = None
            elif p.startswith("a/"):
                old_path = normalize_code_path(p[2:])
            else:
                old_path = normalize_code_path(p)
            if old_path and not is_localizable_java_source_path(old_path):
                old_path = None
            i += 1
            continue

        if old_path is None:
            i += 1
            continue

        m = HUNK_HEADER_RE.match(line)
        if m:
            old_start = int(m.group(1))
            cur_old = old_start if old_start > 0 else 1
            i += 1
            saw_minus_in_hunk = False
            while i < len(lines):
                hl = lines[i]
                if hl.startswith("--- ") or hl.startswith("diff --git"):
                    break
                if HUNK_HEADER_RE.match(hl):
                    break
                if hl == "":
                    i += 1
                    continue
                if hl.startswith("\\"):
                    i += 1
                    continue
                prefix = hl[0]
                if prefix not in " +-":
                    i += 1
                    continue
                if prefix == " ":
                    cur_old += 1
                elif prefix == "-":
                    saw_minus_in_hunk = True
                    touched.setdefault(old_path, set()).add(cur_old)
                    cur_old += 1
                elif prefix == "+":
                    if not saw_minus_in_hunk:
                        # Pure insertion hunk: anchor at the current old-file line index.
                        touched.setdefault(old_path, set()).add(cur_old)
                i += 1
            continue
        i += 1

    return touched


def gold_lines_to_sorted_list(lines_by_file: Dict[str, Set[int]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for f in sorted(lines_by_file.keys()):
        for ln in sorted(lines_by_file[f]):
            out.append({"file": f, "line": ln})
    return out


def load_repograph_tag_objects(tags_json_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not tags_json_path.is_file():
        return rows
    with tags_json_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _tag_line_span(tag_line: Any) -> Optional[Tuple[int, int]]:
    if tag_line is None:
        return None
    if isinstance(tag_line, int):
        return (tag_line, tag_line)
    if isinstance(tag_line, (list, tuple)) and len(tag_line) >= 2:
        a, b = int(tag_line[0]), int(tag_line[1])
        return (min(a, b), max(a, b))
    return None


def _tag_span_size(tag_line: Any) -> int:
    sp = _tag_line_span(tag_line)
    if not sp:
        return 10**9
    return sp[1] - sp[0] + 1


def _tag_span_distance(tag_line: Any, line_num: int) -> int:
    sp = _tag_line_span(tag_line)
    if not sp:
        return 10**9
    lo, hi = sp
    if lo <= line_num <= hi:
        return 0
    return min(abs(line_num - lo), abs(line_num - hi))


def _source_search_roots() -> List[Path]:
    root = Path(__file__).resolve().parents[2]
    return [
        root / "datasets/original_github/multi-swe-bench-java",
        root / "resultgraphs/repo_cache",
    ]


def _find_source_file(rel_fname_norm: str) -> Optional[Path]:
    if not rel_fname_norm:
        return None
    target_suffix = rel_fname_norm.replace("\\", "/").lstrip("/")
    tail_name = Path(target_suffix).name
    for base in _source_search_roots():
        if not base.exists():
            continue
        try:
            for cand in base.rglob(tail_name):
                if not cand.is_file():
                    continue
                cand_norm = str(cand).replace("\\", "/")
                if cand_norm.endswith(target_suffix):
                    return cand
        except OSError:
            continue
    return None


def _infer_symbol_from_source_file(rel_fname_norm: str, line_num: int) -> Optional[str]:
    source_path = _find_source_file(rel_fname_norm)
    if not source_path:
        return None

    try:
        lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None

    class_name: Optional[str] = None
    best_symbol: Optional[str] = None
    class_re = re.compile(r"\b(class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)")
    method_re = re.compile(
        r"^\s*(?:@[A-Za-z_][A-Za-z0-9_\.]*\s*)*"
        r"(?:(?:public|protected|private|static|final|abstract|synchronized|native|strictfp|default)\s+)*"
        r"(?:<[A-Za-z0-9_,\s?&<>\[\]\.]+>\s+)?"
        r"[A-Za-z0-9_\[\]<>.?]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*(?:\{|;)?\s*$"
    )

    limit = max(0, min(line_num, len(lines)))
    for raw in lines[:limit]:
        line = raw.strip()
        if not line or line.startswith("//") or line.startswith("*"):
            continue
        m_class = class_re.search(line)
        if m_class:
            class_name = m_class.group(2)
            best_symbol = class_name
        m_method = method_re.match(raw)
        if m_method:
            best_symbol = m_method.group(1)

    if best_symbol:
        return f"{rel_fname_norm}::{best_symbol}"
    return f"{rel_fname_norm}::L{line_num}"


def resolve_def_symbol_for_line(
    rel_fname_norm: str,
    line_num: int,
    tag_rows: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Map a (file, 1-based line) in the base snapshot to a best-effort enclosing definition symbol
    using repograph `tags.json` (kind=def, category=function|method|class|...).
    Returns a stable string like ``relative/path.java::symbolName``.
    """
    cands: List[Tuple[int, str, str]] = []
    for o in tag_rows:
        if o.get("kind") != "def":
            continue
        cat = (o.get("category") or "").lower()
        if cat not in ("function", "method", "constructor", "class", "interface", "enum"):
            continue
        rel = normalize_code_path(o.get("rel_fname") or "")
        if not rel:
            continue
        if not match_file(rel, {rel_fname_norm}):
            continue
        sp = _tag_line_span(o.get("line"))
        if not sp:
            continue
        lo, hi = sp
        if not (lo <= line_num <= hi):
            continue
        name = (o.get("name") or "").strip()
        if not name:
            continue
        cands.append((_tag_span_size(o.get("line")), cat, name))

    if not cands:
        fallback_cands: List[Tuple[int, int, str, str]] = []
        for o in tag_rows:
            if o.get("kind") != "def":
                continue
            cat = (o.get("category") or "").lower()
            if cat not in ("function", "method", "constructor", "class", "interface", "enum"):
                continue
            rel = normalize_code_path(o.get("rel_fname") or "")
            if not rel:
                continue
            if not match_file(rel, {rel_fname_norm}):
                continue
            name = (o.get("name") or "").strip()
            if not name:
                continue
            dist = _tag_span_distance(o.get("line"), line_num)
            fallback_cands.append((dist, _tag_span_size(o.get("line")), 0 if cat in ("function", "method", "constructor") else 1, name))
        if fallback_cands:
            fallback_cands.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
            return f"{rel_fname_norm}::{fallback_cands[0][3]}"
        return _infer_symbol_from_source_file(rel_fname_norm, line_num)
    cands.sort(key=lambda x: (x[0], 0 if x[1] == "function" else 1, x[2]))
    sym = cands[0][2]
    return f"{rel_fname_norm}::{sym}"


def build_gold_functions_from_lines(
    lines_by_file: Dict[str, Set[int]],
    tag_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for f in sorted(lines_by_file.keys()):
        for ln in sorted(lines_by_file[f]):
            sym = resolve_def_symbol_for_line(f, ln, tag_rows)
            if sym and sym not in seen:
                seen.add(sym)
                out.append({"file": f, "line": ln, "symbol": sym})
    return out


def extract_predicted_line_locations(pred_text: str) -> List[Tuple[str, int]]:
    """
    Ordered (file, line) from phase 2.
    Supports both single line ("行: 123") and ranges ("行: 120-130").
    """
    if not pred_text:
        return []
    text = pred_text
    if PHASE2_TOPK_MARKER in text:
        text = text.split(PHASE2_TOPK_MARKER, 1)[1]
    locs: List[Tuple[str, int]] = []
    seen: Set[Tuple[str, int]] = set()
    pattern = re.compile(
        r"文件:\s*(.*?)\s+行:\s*([0-9]+(?:\s*-\s*[0-9]+)?)",
        re.IGNORECASE,
    )
    range_cap = int(os.environ.get("JLOC_MAX_PRED_LINES_FROM_RANGE", "20") or "20")

    def _add_loc(norm: str, ln: int) -> None:
        key = (norm, ln)
        if key in seen:
            return
        seen.add(key)
        locs.append((norm, ln))

    for line in text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        raw = m.group(1).strip()
        line_raw = m.group(2).replace(" ", "")
        norm = normalize_code_path(raw)
        if not norm:
            continue
        if "-" in line_raw:
            start_s, end_s = line_raw.split("-", 1)
            if not (start_s.isdigit() and end_s.isdigit()):
                continue
            start_ln = int(start_s)
            end_ln = int(end_s)
            if end_ln < start_ln:
                start_ln, end_ln = end_ln, start_ln
            span = end_ln - start_ln + 1
            if range_cap > 0 and span > range_cap:
                end_ln = start_ln + range_cap - 1
            for ln in range(start_ln, end_ln + 1):
                _add_loc(norm, ln)
            continue
        if not line_raw.isdigit():
            continue
        _add_loc(norm, int(line_raw))

    cap = int(os.environ.get("JLOC_MAX_PRED_LINES_FROM_PHASE2", "12") or "12")
    if cap > 0 and len(locs) > cap:
        return locs[:cap]
    return locs


def build_pred_functions_from_locations(
    locs: List[Tuple[str, int]],
    tag_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for f, ln in locs:
        sym = resolve_def_symbol_for_line(f, ln, tag_rows)
        out.append({"file": f, "line": ln, "symbol": sym})
    return out


def normalize_code_path(path_str: str) -> str:
    if not path_str:
        return ""
    p = path_str.strip().replace("\\", "/")
    # Prefer an explicit Java path inside markdown/backtick/link wrappers.
    # This turns e.g. [Foo.java](src/main/java/x/Foo.java) into the linked
    # source path instead of persisting markdown punctuation as a filename.
    java_paths = re.findall(
        r"(?:[A-Za-z]:)?[A-Za-z0-9_.$@+~/-]*[A-Za-z0-9_$-]+\.java",
        p,
    )
    if java_paths:
        p = max(java_paths, key=len)
    else:
        p = p.strip("`*_~'\"()[]{}<>")
    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]
    anchors = [
        "src/main/java/",
        "src/test/java/",
        "src/",
    ]
    for anchor in anchors:
        idx = p.find(anchor)
        if idx >= 0:
            return p[idx:]
    return p.lstrip("/")


PHASE2_TOPK_MARKER = "===PHASE2_TOPK_LOCATION==="


def extract_phase1_candidate_files(pred_text: str) -> List[str]:
    """
    从步骤 A 文案中解析「候选文件 / 怀疑文件」路径。
    若存在 PHASE2 分隔符，只解析其之前的段落，避免与行级块混淆。
    """
    if not pred_text:
        return []
    phase1 = (
        pred_text.split(PHASE2_TOPK_MARKER, 1)[0]
        if PHASE2_TOPK_MARKER in pred_text
        else pred_text
    )
    files: List[str] = []
    seen = set()
    pattern = re.compile(r"(?:怀疑文件|候选文件):\s*(.*?)\s+理由:", re.IGNORECASE)
    for line in phase1.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        norm = normalize_code_path(m.group(1).strip())
        if norm and norm not in seen:
            seen.add(norm)
            files.append(norm)
    return files


def extract_phase1_function_candidates(pred_text: str) -> List[Dict[str, Any]]:
    """
    Parse function-level candidates from step A (before PHASE2 marker).
    Expected format per line:
      函数: <name> 文件: <path.java> 行: <start[-end]> 理由: <...>
    """
    if not pred_text:
        return []
    phase1 = (
        pred_text.split(PHASE2_TOPK_MARKER, 1)[0]
        if PHASE2_TOPK_MARKER in pred_text
        else pred_text
    )
    out: List[Dict[str, Any]] = []
    seen = set()
    pattern = re.compile(
        r"函数:\s*(.*?)\s+文件:\s*(.*?)\s+行:\s*([0-9]+(?:\s*-\s*[0-9]+)?)",
        re.IGNORECASE,
    )
    for line in phase1.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        func = m.group(1).strip()
        file_path = normalize_code_path(m.group(2).strip())
        line_raw = m.group(3).replace(" ", "")
        if not file_path or not func:
            continue
        key = (file_path, func, line_raw)
        if key in seen:
            continue
        seen.add(key)
        line_start = None
        line_end = None
        if "-" in line_raw:
            parts = line_raw.split("-", 1)
            if parts[0].isdigit() and parts[1].isdigit():
                line_start = int(parts[0])
                line_end = int(parts[1])
        elif line_raw.isdigit():
            line_start = int(line_raw)
            line_end = int(line_raw)
        out.append(
            {
                "function_name": func,
                "file_path": file_path,
                "line_start": line_start,
                "line_end": line_end,
            }
        )
    return out


def extract_suspected_files_phase1(pred_text: str) -> List[str]:
    """Parse 步骤 A / 阶段1 候选或怀疑文件行（与 extract_phase1_candidate_files 一致）。"""
    return extract_phase1_candidate_files(pred_text or "")


def extract_predicted_file_paths(pred_text: str) -> List[str]:
    """
    评估与落盘用的文件列表：优先 PHASE2 区块内「文件: … 行:」顺序；
    若无（模型未输出 PHASE2），回退到步骤 A 的候选文件行，避免 pred_files 恒为空。
    """
    out = extract_predicted_files(pred_text)
    if out:
        return out
    return extract_phase1_candidate_files(pred_text)


def extract_predicted_files(pred_text: str) -> List[str]:
    """
    Parse line-level predictions. If the model used two-stage output, only the section after
    PHASE2_TOPK_MARKER is used (see localization_llm.PHASE2_MARKER).
    """
    if not pred_text:
        return []
    text = pred_text
    if PHASE2_TOPK_MARKER in text:
        text = text.split(PHASE2_TOPK_MARKER, 1)[1]
    files: List[str] = []
    seen = set()
    pattern = re.compile(r"文件:\s*(.*?)\s+行:", re.IGNORECASE)
    for line in text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        raw = m.group(1).strip()
        norm = normalize_code_path(raw)
        if norm and norm not in seen:
            seen.add(norm)
            files.append(norm)
    cap = int(os.environ.get("JLOC_MAX_PRED_FILES_FROM_PHASE2", "12") or "12")
    if cap > 0 and len(files) > cap:
        return files[:cap]
    return files


def match_file(pred_file: str, gold_files: Set[str]) -> bool:
    if not pred_file or not gold_files:
        return False
    if pred_file in gold_files:
        return True
    for g in gold_files:
        if pred_file.endswith(g) or g.endswith(pred_file):
            return True
    return False


def calc_topn_hit(pred_files: List[str], gold_files: Set[str], n: int) -> float:
    topn = pred_files[:n]
    return 1.0 if any(match_file(p, gold_files) for p in topn) else 0.0


def calc_first_hit_rank_file(pred_files: List[str], gold_files: Set[str]) -> Optional[int]:
    """首个命中 gold 文件的预测在列表中的 **1-based** 排名；无命中为 ``None``。"""
    if not gold_files:
        return None
    for i, p in enumerate(pred_files):
        if match_file(p, gold_files):
            return i + 1
    return None


def calc_file_mrr(pred_files: List[str], gold_files: Set[str]) -> float:
    """文件级 MRR：首个命中排名 r 的贡献为 ``1/r``，无命中为 ``0``。"""
    r = calc_first_hit_rank_file(pred_files, gold_files)
    return (1.0 / float(r)) if r else 0.0


def calc_first_hit_rank_line(
    pred_locs: List[Tuple[str, int]],
    gold_lines_by_file: Dict[str, Set[int]],
) -> Optional[int]:
    """首个与任一 gold (file,line) 严格匹配（含与 ``calc_line_topn_hit`` 一致的路径规则）的 **1-based** 排名。"""
    if not gold_lines_by_file:
        return None
    gold_keys = _gold_line_key_set(gold_lines_by_file)
    for i, (f, ln) in enumerate(pred_locs):
        nf = normalize_code_path(f)
        pl = int(ln)
        if (nf, pl) in gold_keys:
            return i + 1
        for gf, gl in gold_keys:
            if pl != int(gl):
                continue
            if match_file(nf, {gf}):
                return i + 1
    return None


def calc_line_mrr(
    pred_locs: List[Tuple[str, int]],
    gold_lines_by_file: Dict[str, Set[int]],
) -> float:
    r = calc_first_hit_rank_line(pred_locs, gold_lines_by_file)
    return (1.0 / float(r)) if r else 0.0


def calc_first_hit_rank_function(
    pred_symbols: List[Optional[str]],
    gold_symbols: Set[str],
) -> Optional[int]:
    """首个与 gold 函数符号集完全相同的预测符号的 **1-based** 排名。"""
    if not gold_symbols:
        return None
    for i, s in enumerate(pred_symbols):
        if s and s in gold_symbols:
            return i + 1
    return None


def calc_function_mrr(pred_symbols: List[Optional[str]], gold_symbols: Set[str]) -> float:
    r = calc_first_hit_rank_function(pred_symbols, gold_symbols)
    return (1.0 / float(r)) if r else 0.0


def _gold_line_key_set(lines_by_file: Dict[str, Set[int]]) -> Set[Tuple[str, int]]:
    keys: Set[Tuple[str, int]] = set()
    for f, lns in lines_by_file.items():
        nf = normalize_code_path(f)
        for ln in lns:
            keys.add((nf, ln))
    return keys


def calc_line_topn_hit(
    pred_locs: List[Tuple[str, int]],
    gold_lines_by_file: Dict[str, Set[int]],
    n: int,
) -> float:
    if not gold_lines_by_file:
        return 0.0
    gold_keys = _gold_line_key_set(gold_lines_by_file)
    for f, ln in pred_locs[:n]:
        nf = normalize_code_path(f)
        if (nf, ln) in gold_keys:
            return 1.0
        for gf, gl in gold_keys:
            if gl != ln:
                continue
            if match_file(nf, {gf}):
                return 1.0
    return 0.0


def calc_line_topn_hit_window(
    pred_locs: List[Tuple[str, int]],
    gold_lines_by_file: Dict[str, Set[int]],
    n: int,
    window: int,
) -> float:
    """
    Windowed line hit: success if any top-n prediction is within +/- window lines of any gold line.
    """
    if not gold_lines_by_file:
        return 0.0
    if window < 0:
        window = 0
    gold_keys = _gold_line_key_set(gold_lines_by_file)
    for f, ln in pred_locs[:n]:
        nf = normalize_code_path(f)
        for gf, gl in gold_keys:
            if not match_file(nf, {gf}):
                continue
            if abs(int(ln) - int(gl)) <= window:
                return 1.0
    return 0.0


def calc_min_line_distance(
    pred_locs: List[Tuple[str, int]],
    gold_lines_by_file: Dict[str, Set[int]],
    n: int,
) -> Optional[int]:
    """
    Min absolute line distance between top-n predictions and gold lines (same file match).
    Returns None if no comparable pairs exist.
    """
    if not gold_lines_by_file or not pred_locs:
        return None
    gold_keys = _gold_line_key_set(gold_lines_by_file)
    best: Optional[int] = None
    for f, ln in pred_locs[:n]:
        nf = normalize_code_path(f)
        for gf, gl in gold_keys:
            if not match_file(nf, {gf}):
                continue
            dist = abs(int(ln) - int(gl))
            if best is None or dist < best:
                best = dist
    return best


def build_gold_function_spans(
    lines_by_file: Dict[str, Set[int]],
    tag_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Return function/class def spans that cover any gold line.
    Each span item: {"file": str, "lo": int, "hi": int, "symbol": str}.
    """
    spans: Dict[str, Dict[str, Any]] = {}
    for obj in tag_rows or []:
        if obj.get("kind") != "def":
            continue
        cat = (obj.get("category") or "").lower()
        if cat not in ("function", "method", "constructor", "class", "interface", "enum"):
            continue
        rel = normalize_code_path(obj.get("rel_fname") or "")
        if not rel or rel not in lines_by_file:
            continue
        sp = _tag_line_span(obj.get("line"))
        if not sp:
            continue
        lo, hi = sp
        if not any(lo <= ln <= hi for ln in lines_by_file.get(rel, set())):
            continue
        name = (obj.get("name") or "").strip()
        sym = f"{rel}::{name}" if name else rel
        if sym in spans:
            continue
        spans[sym] = {"file": rel, "lo": int(lo), "hi": int(hi), "symbol": sym}
    return list(spans.values())


def calc_function_span_hit_topn(
    pred_locs: List[Tuple[str, int]],
    gold_spans: List[Dict[str, Any]],
    n: int,
) -> float:
    """
    Hit if any top-n predicted line falls within a gold function/class span.
    """
    if not gold_spans:
        return 0.0
    for f, ln in pred_locs[:n]:
        nf = normalize_code_path(f)
        for sp in gold_spans:
            if not match_file(nf, {sp.get("file") or ""}):
                continue
            lo = int(sp.get("lo", -1))
            hi = int(sp.get("hi", -1))
            if lo <= int(ln) <= hi:
                return 1.0
    return 0.0


def calc_func_topn_hit(pred_symbols: List[Optional[str]], gold_symbols: Set[str], n: int) -> float:
    if not gold_symbols:
        return 0.0
    for sym in pred_symbols[:n]:
        if sym and sym in gold_symbols:
            return 1.0
    return 0.0


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall <= 1e-12:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def line_pred_matches_gold_file_line(pf: str, pl: int, gf: str, gl: int) -> bool:
    """True if predicted (file, line) refers to the same location as gold (file, line)."""
    if pl != gl:
        return False
    np = normalize_code_path(pf)
    ng = normalize_code_path(gf)
    if np == ng:
        return True
    return match_file(np, {ng})


def metrics_file_level_sl_rec_pre_f1(gold_files: Set[str], pred_files: List[str]) -> Dict[str, float]:
    """
    Success Location (SL): 1 iff every gold file is matched by at least one predicted path.
    Recall: fraction of gold files that are matched.
    Precision: fraction of predicted files that match some gold file.
    F1: harmonic mean of precision and recall.
    """
    pred_u = list(dict.fromkeys(pred_files))
    if not gold_files:
        pre = 1.0 if not pred_u else 0.0
        return {
            "SL": 1.0,
            "REC": 1.0,
            "PRE": pre,
            "F1": _safe_f1(pre, 1.0),
        }

    cov = sum(1 for g in gold_files if any(match_file(p, {g}) for p in pred_u))
    hits = sum(1 for p in pred_u if match_file(p, gold_files))
    rec = cov / len(gold_files)
    pre = hits / len(pred_u) if pred_u else 0.0
    sl = 1.0 if cov == len(gold_files) else 0.0
    return {"SL": sl, "REC": rec, "PRE": pre, "F1": _safe_f1(pre, rec)}


def metrics_line_level_sl_rec_pre_f1(
    gold_lines_by_file: Dict[str, Set[int]],
    pred_locs: List[Tuple[str, int]],
) -> Dict[str, float]:
    """Same definitions as file-level, on (file, line) locations (with fuzzy file match)."""
    gold_keys = _gold_line_key_set(gold_lines_by_file)
    pred_u = list(dict.fromkeys(pred_locs))
    if not gold_keys:
        pre = 1.0 if not pred_u else 0.0
        return {
            "SL": 1.0,
            "REC": 1.0,
            "PRE": pre,
            "F1": _safe_f1(pre, 1.0),
        }

    def gold_item_covered(gf: str, gl: int) -> bool:
        return any(
            line_pred_matches_gold_file_line(pf, pl, gf, gl) for pf, pl in pred_u
        )

    cov = sum(1 for gf, gl in gold_keys if gold_item_covered(gf, gl))
    hits = sum(
        1
        for pf, pl in pred_u
        if any(line_pred_matches_gold_file_line(pf, pl, gf, gl) for gf, gl in gold_keys)
    )
    rec = cov / len(gold_keys)
    pre = hits / len(pred_u) if pred_u else 0.0
    sl = 1.0 if cov == len(gold_keys) else 0.0
    return {"SL": sl, "REC": rec, "PRE": pre, "F1": _safe_f1(pre, rec)}


def metrics_function_level_sl_rec_pre_f1(
    gold_symbols: Set[str],
    pred_symbols: List[Optional[str]],
) -> Dict[str, float]:
    """Symbol strings must match exactly (same as gold/pred `::` ids)."""
    pred_set = {s for s in pred_symbols if s}
    if not gold_symbols:
        pre = 1.0 if not pred_set else 0.0
        return {
            "SL": 1.0,
            "REC": 1.0,
            "PRE": pre,
            "F1": _safe_f1(pre, 1.0),
        }

    inter = gold_symbols & pred_set
    rec = len(inter) / len(gold_symbols)
    pre = len(inter) / len(pred_set) if pred_set else 0.0
    sl = 1.0 if gold_symbols <= pred_set else 0.0
    return {"SL": sl, "REC": rec, "PRE": pre, "F1": _safe_f1(pre, rec)}


def oracle_level_recall_scores(
    localization_text: str,
    gold_files: Set[str],
    gold_lines_by_file: Dict[str, Set[int]],
    gold_symbols: Set[str],
    tag_rows: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Per-level recall in [0, 1] for reflection-loop stopping rules:
    file / function / line use the same REC definitions as metrics_*_level_sl_rec_pre_f1.
    """
    pred_files = extract_predicted_files(localization_text)
    pred_locs = extract_predicted_line_locations(localization_text)
    pred_func_objs = build_pred_functions_from_locations(pred_locs, tag_rows)
    pred_sym_list = [p.get("symbol") for p in pred_func_objs]
    mf = metrics_file_level_sl_rec_pre_f1(gold_files, pred_files)
    ml = metrics_line_level_sl_rec_pre_f1(gold_lines_by_file, pred_locs)
    mfn = metrics_function_level_sl_rec_pre_f1(gold_symbols, pred_sym_list)
    return {
        "file": float(mf["REC"]),
        "line": float(ml["REC"]),
        "function": float(mfn["REC"]),
    }


def oracle_reflect_should_stop(
    scores: Dict[str, float],
    *,
    has_gold_files: bool = True,
    has_gold_lines: bool = True,
    has_gold_funcs: bool = True,
) -> bool:
    """
    Localize↔reflection loop: stop early if file recall is 1, or function >= 0.8, or line >= 0.7.
    Dimensions with no gold in the dataset are ignored (metrics otherwise return REC=1.0 for empty gold).
    """
    if has_gold_files and float(scores.get("file", 0.0)) >= 1.0 - 1e-9:
        return True
    if has_gold_funcs and float(scores.get("function", 0.0)) >= 0.8:
        return True
    if has_gold_lines and float(scores.get("line", 0.0)) >= 0.7:
        return True
    return False


def extract_valid_locations_from_context(context: str) -> Dict[str, Set[int]]:
    """
    从 GraphRAG 上下文字符串中解析 File:/Line: 对，构建 {归一化路径 -> 有效行号集合}。
    支持单行号（281）和范围（281-294），跳过 -1 及非数字。
    """
    result: Dict[str, Set[int]] = {}
    cur_file: Optional[str] = None
    for raw_line in context.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("File:"):
            cur_file = normalize_code_path(stripped[5:].strip())
        elif stripped.startswith("Line:") and cur_file:
            val = stripped[5:].strip()
            for part in val.replace(",", " ").split():
                part = part.strip()
                if not part:
                    continue
                if "-" in part and not part.startswith("-"):
                    try:
                        a, b = part.split("-", 1)
                        lo, hi = int(a), int(b)
                        if lo > 0 and hi >= lo:
                            result.setdefault(cur_file, set()).update(range(lo, hi + 1))
                    except ValueError:
                        pass
                else:
                    try:
                        n = int(part)
                        if n > 0:
                            result.setdefault(cur_file, set()).add(n)
                    except ValueError:
                        pass
    return result


def format_valid_locations_hint(valid_locs: Dict[str, Set[int]], max_files: int = 30) -> str:
    """
    生成有效位置白名单提示字符串，注入到定位提示词中，约束模型只引用 GraphRAG 中出现的行号。
    """
    if not valid_locs:
        return ""
    lines = [
        "【有效位置白名单】下列文件与行号来自 GraphRAG 上下文中的实际 Line: 字段；",
        "阶段 2 行号必须来自此列表（若某文件不在列表则上下文中无该文件，禁止在阶段 2 报该文件的任何行号）：",
    ]
    for f in sorted(valid_locs.keys())[:max_files]:
        lns = sorted(valid_locs[f])
        if len(lns) <= 25:
            ln_str = ", ".join(str(n) for n in lns)
        else:
            ln_str = ", ".join(str(n) for n in lns[:25]) + f" …(共 {len(lns)} 行)"
        lines.append(f"  {f}: [{ln_str}]")
    if len(valid_locs) > max_files:
        lines.append(f"  …(另有 {len(valid_locs) - max_files} 个文件未显示)")
    return "\n".join(lines)


def filter_line_predictions(pred_text: str, valid_locs: Dict[str, Set[int]]) -> str:
    """
    后处理过滤：删除阶段 2 中预测的 (file, line) 对，若该文件在白名单中存在但行号不在白名单里。
    对上下文中没有收录的文件（不在 valid_locs 中）保持原样，不过滤（无法验证）。
    阶段 1（怀疑文件列表）不修改。
    """
    if not valid_locs or not pred_text or PHASE2_TOPK_MARKER not in pred_text:
        return pred_text
    phase1, phase2 = pred_text.split(PHASE2_TOPK_MARKER, 1)
    pattern = re.compile(r"文件:\s*(.*?)\s+行:\s*(\d+)", re.IGNORECASE)
    kept: List[str] = []
    removed = 0
    for line in phase2.splitlines():
        m = pattern.search(line)
        if m:
            norm_file = normalize_code_path(m.group(1).strip())
            try:
                ln = int(m.group(2))
            except ValueError:
                kept.append(line)
                continue
            if norm_file not in valid_locs:
                kept.append(line)
            elif ln in valid_locs[norm_file]:
                kept.append(line)
            else:
                removed += 1
        else:
            kept.append(line)
    filtered = phase1 + PHASE2_TOPK_MARKER + "\n".join(kept)
    if removed:
        filtered += f"\n# [filter] 已删除 {removed} 条不在上下文白名单中的行号预测"
    return filtered


def combined_score_403030(file_rec: float, func_rec: float, line_rec: float) -> float:
    """Reflection Agent 推荐：0.4*file + 0.3*func + 0.3*line，均在 [0,1]。"""
    return 0.4 * float(file_rec) + 0.3 * float(func_rec) + 0.3 * float(line_rec)


def combined_localization_score(
    pred_text: str,
    gold_files: Set[str],
    gold_lines_by_file: Dict[str, Set[int]],
    gold_symbols: Set[str],
    tag_rows: List[Dict[str, Any]],
    weights: Tuple[float, float, float] = (0.3, 0.5, 0.2),
) -> float:
    """
    加权联合召回分：file × 0.3 + line × 0.5 + func × 0.2（可调权重）。
    用于从多轮 reflection 候选中选出最优结果（取代仅按文件级 rank 选优）。
    缺少 gold 的维度权重归 0（不计入分母）。
    """
    if not gold_files and not gold_lines_by_file and not gold_symbols:
        return 0.0
    scores = oracle_level_recall_scores(
        pred_text, gold_files, gold_lines_by_file, gold_symbols, tag_rows
    )
    wf, wl, wfn = weights
    total_w = 0.0
    score = 0.0
    if gold_files:
        score += wf * scores["file"]
        total_w += wf
    if gold_lines_by_file:
        score += wl * scores["line"]
        total_w += wl
    if gold_symbols:
        score += wfn * scores["function"]
        total_w += wfn
    return score / total_w if total_w > 0 else 0.0


def localization_rank_score(pred_text: str, gold_files: Set[str]) -> float:
    """
    Scalar score for comparing multiple localization attempts (higher is better).
    First matching rank in top-20 dominates; small tie-break for filename overlap.
    """
    if not gold_files:
        return 0.0
    preds = extract_predicted_files(pred_text or "")
    if not preds:
        return 0.0
    gold_norm = {normalize_code_path(g) for g in gold_files}
    for i, p in enumerate(preds[:20]):
        if match_file(p, gold_norm):
            return 1000.0 - float(i)
    bonus = 0.0
    for p in preds[:10]:
        pl = p.lower()
        for g in gold_norm:
            name = g.split("/")[-1].lower()
            if len(name) > 4 and name in pl:
                bonus = max(bonus, 2.0)
            elif name and name == pl.split("/")[-1]:
                bonus = max(bonus, 1.0)
    return bonus
