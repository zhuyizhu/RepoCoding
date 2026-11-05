from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from agent.tools.localization_llm import invoke_llm_with_ollama_fallback


def _strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def parse_reflection_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出中解析严格 JSON 对象。"""
    if not text:
        return None
    raw = _strip_code_fence(text)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                pass
    return None


REFLECTION_JSON_SCHEMA = r"""
{
  "best_combined_score": float,
  "verification_best_score": float,
  "oracle_stop": boolean,
  "localize_candidates": [],
  "localize_refinement_feedback": "string",
  "next_action_hint": "string",
  "reflection_summary": "string",
  "n_reflection_calls": int
}
"""


def run_reflection_agent_json(
    *,
    issue_text: str,
    localization_result: str,
    localize_candidates: List[Dict[str, Any]],
    localize_refinement_feedback: str,
    last_oracle_scores: Optional[Dict[str, float]],
    verification_history: List[Dict[str, Any]],
    deterministic_scores: Dict[str, float],
    n_reflection_calls: int,
    graph_context_excerpt: str,
) -> Dict[str, Any]:
    """
    JLocAgent Reflection Agent：输出严格 JSON（见 prompt）。
    若解析失败，返回空 dict，由调用方回退到规则打分。
    """
    los = last_oracle_scores if last_oracle_scores is not None else {}
    cand_json = json.dumps(localize_candidates, ensure_ascii=False)
    if len(cand_json) > 12000:
        cand_json = cand_json[:12000] + "…"
    hist_json = json.dumps(verification_history, ensure_ascii=False)
    if len(hist_json) > 12000:
        hist_json = hist_json[:12000] + "…"

    prompt = f"""你是 JLocAgent 的 Reflection Agent，负责对 localize_agent_phase2 的结果进行自我反思和打分，并决定是否继续优化。

输入信息：
- localization_result: 当前最终定位结果
- localize_candidates: 候选文件/函数/行列表（带置信度）
- localize_refinement_feedback: 上一次 refinement 的反馈
- last_oracle_scores: 上一次 oracle 打分
- verification_history: 历史验证记录

你的输出必须包含以下字段（严格 JSON 格式）：

{REFLECTION_JSON_SCHEMA}

字段说明：
- best_combined_score: 0-1，综合 file + function + line 得分（推荐公式：0.4*file_score + 0.3*func_score + 0.3*line_score）
- verification_best_score: oracle 验证最高分
- oracle_stop: true 表示已足够好，可以结束
- localize_candidates: 优化后的候选列表（最多保留 top-5）
- localize_refinement_feedback: 本次反思给 localize_agent 的具体改进建议
- next_action_hint: 给 supervisor 的明确建议，例如："需要再跑一次 phase2 聚焦 XXX 函数的第 45-52 行" 或 "已达标，可直接 persist"
- reflection_summary: 3-5 句话总结本次反思发现的问题和改进方向
- n_reflection_calls: 当前轮次（应与下方输入一致）

【反思原则】（必须遵守）：
- 优先关注 line-level 和 function-level 精度（当前 smoke test 中 line F1=0 是最大短板）
- 如果候选行超过 8 行，必须在 localize_refinement_feedback 中明确要求“进一步缩小到具体 1-3 行”
- 结合 Dubbo 特性（动态路由、服务治理、SPI 等），判断定位是否合理
- 只有当 file 已命中 且 function 命中率 ≥80% 且 line 候选 ≤3 行 时，才允许 oracle_stop=True

【客观 recall（基于 gold 的 deterministic_scores，0-1）】
{json.dumps(deterministic_scores, ensure_ascii=False)}

【Issue 摘要】
{issue_text[:6000]}

【localization_result】
{localization_result[:8000]}

【localize_candidates】
{cand_json}

【localize_refinement_feedback】
{(localize_refinement_feedback or "(无)")[:4000]}

【last_oracle_scores】
{json.dumps(los, ensure_ascii=False)}

【verification_history】
{hist_json}

【graph_context_excerpt】
{(graph_context_excerpt or "")[:4500]}

【当前反思轮次 n_reflection_calls】
{n_reflection_calls}

请直接输出 JSON，不要添加任何额外文字。
"""
    try:
        resp = invoke_llm_with_ollama_fallback(prompt, call_site="run_reflection_agent_json")
        content = (getattr(resp, "content", None) or str(resp) or "").strip()
        parsed = parse_reflection_json(content)
        return parsed or {}
    except Exception:
        return {}
