"""
进程内、单实例顺序安全：累加各次 Chat 调用的 token 用量（供论文成本/效率分析）。

在 ``run_in_executor`` 的工作线程中也会写入，故使用 ``threading.Lock``。
单条 instance 开始前须 ``reset_session_llm_usage()``，流水线结束后 ``snapshot_session_llm_usage()``。
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

_lock = threading.Lock()
_prompt_tokens: int = 0
_completion_tokens: int = 0
_call_count: int = 0
_per_call: List[Dict[str, Any]] = []
_models_reported: List[str] = []


def reset_session_llm_usage() -> None:
    """在新 instance 的 Localization 流水线开始前调用。"""
    global _prompt_tokens, _completion_tokens, _call_count, _per_call, _models_reported
    with _lock:
        _prompt_tokens = 0
        _completion_tokens = 0
        _call_count = 0
        _per_call = []
        _models_reported = []


def _usage_dict_from_message(msg: Any) -> Dict[str, Any]:
    """
    从 ``AIMessage``（或兼容对象）解析 token 与服务商回传的 model id。

    兼容：``usage_metadata``（LangChain）、OpenAI ``response_metadata.token_usage``、
    Ollama ``prompt_eval_count`` / ``eval_count``。
    """
    pt = ct = tt = 0
    model_reported: Optional[str] = None

    um = getattr(msg, "usage_metadata", None)
    if um is not None:
        if isinstance(um, dict):
            pt = int(um.get("input_tokens") or um.get("prompt_tokens") or 0)
            ct = int(um.get("output_tokens") or um.get("completion_tokens") or 0)
            tt = int(um.get("total_tokens") or 0)
            model_reported = um.get("model") if isinstance(um.get("model"), str) else model_reported
        else:
            pt = int(getattr(um, "input_tokens", None) or getattr(um, "prompt_tokens", None) or 0)
            ct = int(getattr(um, "output_tokens", None) or getattr(um, "completion_tokens", None) or 0)
            tt = int(getattr(um, "total_tokens", None) or 0)
            m = getattr(um, "model", None)
            if isinstance(m, str):
                model_reported = m

    rm = getattr(msg, "response_metadata", None) or {}
    if isinstance(rm, dict):
        if not model_reported:
            model_reported = rm.get("model_name") or rm.get("model")
            if not isinstance(model_reported, str):
                model_reported = None
        if not pt and not ct:
            tu = rm.get("token_usage")
            if isinstance(tu, dict):
                pt = int(tu.get("prompt_tokens") or 0)
                ct = int(tu.get("completion_tokens") or 0)
                tt = int(tu.get("total_tokens") or (pt + ct))
        if not pt and not ct:
            pe = rm.get("prompt_eval_count")
            ec = rm.get("eval_count")
            if pe is not None:
                pt = int(pe)
            if ec is not None:
                ct = int(ec)
            if pt or ct:
                tt = pt + ct if not tt else tt

    if not tt and (pt or ct):
        tt = pt + ct

    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
        "model_reported": model_reported,
    }


def record_llm_ai_message(resp: Any, *, call_site: str = "") -> None:
    """在一次 ``llm.invoke`` 返回后调用，累加用量。"""
    global _prompt_tokens, _completion_tokens, _call_count, _per_call, _models_reported
    u = _usage_dict_from_message(resp)
    with _lock:
        _call_count += 1
        _prompt_tokens += int(u["prompt_tokens"])
        _completion_tokens += int(u["completion_tokens"])
        rec = {
            "call_index": _call_count,
            "call_site": call_site,
            "prompt_tokens": u["prompt_tokens"],
            "completion_tokens": u["completion_tokens"],
            "total_tokens": u["total_tokens"],
            "model_reported": u["model_reported"],
        }
        _per_call.append(rec)
        mid = u.get("model_reported")
        if isinstance(mid, str) and mid and mid not in _models_reported:
            _models_reported.append(mid)


def snapshot_session_llm_usage() -> Dict[str, Any]:
    """返回当前会话累计（浅拷贝 list/dict，可写入 json）。"""
    with _lock:
        return {
            "call_count": _call_count,
            "prompt_tokens": _prompt_tokens,
            "completion_tokens": _completion_tokens,
            "total_tokens": _prompt_tokens + _completion_tokens,
            "models_reported": list(_models_reported),
            "per_call": list(_per_call),
        }
