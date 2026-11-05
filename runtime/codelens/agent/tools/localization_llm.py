from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Set, Tuple, TypeVar, Union
import time

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agent.utils.llm_http_timeout import resolve_llm_request_timeout_sec
from agent.utils.llm_usage import record_llm_ai_message

PHASE2_MARKER = "===PHASE2_TOPK_LOCATION==="
_T = TypeVar("_T")


def _granularity_hint(granularity: str) -> str:
    g = (granularity or "file").strip().lower()
    if g == "line":
        return "【粒度指令】当前是行级聚焦。优先给出 1-3 个最关键行号，避免泛化到整个文件。"
    if g == "function":
        return "【粒度指令】当前是函数级聚焦。优先锁定具体方法或类的核心入口与关键调用。"
    return "【粒度指令】当前是文件级定位。优先覆盖与 Issue 语义最匹配的 3-6 个核心实现文件。"


def _output_structure(granularity: str) -> str:
    g = (granularity or "file").strip().lower()
    if g == "function":
        return (
            "输出结构（两步都必须完成；分隔符用于下游解析，勿改）\n\n"
            "【步骤 A — 候选函数/类（函数级）】\n"
            "- 在 GraphRAG 上下文中选出最多 {k} 个最可能相关的函数或类定义。\n"
            "- 函数或类名必须来自上下文中的 NodeId 或 Info 片段，禁止臆造。\n"
            "- 行号必须来自上下文中的 ``Line:`` 字段；可用起止行范围。\n"
            "- 格式（每行，从 1 开始编号）：\n"
            "  1. 函数: <方法或类名> 文件: <路径.java> 行: <起始行或范围> 原因: <一句中文>\n\n"
            "【步骤 B — Top-K 行级候选（行级）】\n"
            "- 在步骤 A 的函数范围内（若步骤 A 为空，则可在全 GraphRAG 上下文中选），"
            "选出最多 {k} 个最可能修改点，精确到行号。\n"
            "- 行号必须来自上下文中的 ``Line:`` 字段；若下方提供了「有效位置白名单」，则只能引用白名单中的行号。\n"
            "- **禁止**仅因白名单里多个文件都出现同名生命周期方法（如 ``destroy``）而机械复制同一类理由凑满 {k} 条；"
            "每条必须与 Issue 中的症状/根因有**可区分**的关联。\n"
            "- 格式（每行，从 1 开始编号）：\n"
            "  1. 文件: <路径.java> 行: <整数或起止范围> 原因: <一句中文>\n\n"
            "输出时**先完整输出步骤 A**，然后**单独一行**输出分隔标记（必须完全一致、单独成行）：\n"
            "{phase2_marker}\n"
            "然后输出步骤 B；步骤 B 中不要重复步骤 A 的列表。"
        )
    if g == "line":
        return (
            "输出结构（两步都必须完成；分隔符用于下游解析，勿改）\n\n"
            "【步骤 A — 候选修改文件（文件级）】\n"
            "- 仅列出你认为最可能需改动的 **.java 文件**（最多 12 个），每行一条。\n"
            "- 路径必须**逐字**来自下方 GraphRAG 上下文中出现过的 ``File:`` 行（禁止臆造）。\n"
            "- **强制排序**：``src/main/java/`` 下业务实现文件**绝对优先**排在前面；\n"
            "  ``src/test/``、``androidTest``、``benchmark``、``jmh`` 等测试/基准路径**必须排在最后**，\n"
            "  除非 Issue 文本明确只涉及测试代码。\n"
            "- 若最佳候选不在 ``src/main/java/`` 下，必须在理由中说明为什么排除主源码。\n"
            "- 格式（每行）：\n"
            "  候选文件: <相对路径.java> 理由: <一句中文，说明与 Issue 症状或栈的关联>\n\n"
            "【步骤 B — Top-K 行级候选（行级）】\n"
            "- 在步骤 A 的文件范围内（若步骤 A 为空，则可在全 GraphRAG 上下文中选），"
            "选出最多 {k} 个**最可能修改点**，精确到行号。\n"
            "- 行号必须来自上下文中的 ``Line:`` 字段；若下方提供了「有效位置白名单」，则只能引用白名单中的行号。\n"
            "- **禁止**仅因白名单里多个文件都出现同名生命周期方法（如 ``destroy``）而机械复制同一类理由凑满 {k} 条；"
            "每条必须与 Issue 中的症状/根因有**可区分**的关联。\n"
            "- 格式（每行，从 1 开始编号）：\n"
            "  1. 文件: <路径.java> 行: <整数或起止范围> 原因: <一句中文>\n\n"
            "输出时**先完整输出步骤 A**，然后**单独一行**输出分隔标记（必须完全一致、单独成行）：\n"
            "{phase2_marker}\n"
            "然后输出步骤 B；步骤 B 中不要重复步骤 A 的「候选文件」列表。"
        )
    return (
        "输出结构（两步都必须完成；分隔符用于下游解析，勿改）\n\n"
        "【步骤 A — 候选修改文件（文件级）】\n"
        "- 仅列出你认为最可能需改动的 **.java 文件**（最多 12 个），每行一条。\n"
        "- 路径必须**逐字**来自下方 GraphRAG 上下文中出现过的 ``File:`` 行（禁止臆造）。\n"
        "- **强制排序**：``src/main/java/`` 下业务实现文件**绝对优先**排在前面；\n"
        "  ``src/test/``、``androidTest``、``benchmark``、``jmh`` 等测试/基准路径**必须排在最后**，\n"
        "  除非 Issue 文本明确只涉及测试代码。\n"
        "- 若最佳候选不在 ``src/main/java/`` 下，必须在理由中说明为什么排除主源码。\n"
        "- 格式（每行）：\n"
        "  候选文件: <相对路径.java> 理由: <一句中文，说明与 Issue 症状或栈的关联>\n\n"
        "【步骤 B — Top-K 函数候选（函数级）】\n"
        "- 在步骤 A 的文件范围内，选出最多 {k} 个最关键的函数或类定义。\n"
        "- 函数或类名必须来自上下文中的 NodeId 或 Info 片段，禁止臆造。\n"
        "- 行号必须来自上下文中的 ``Line:`` 字段；可用起止行范围。\n"
        "- 格式（每行，从 1 开始编号）：\n"
        "  1. 函数: <方法或类名> 文件: <路径.java> 行: <起始行或范围> 原因: <一句中文>\n\n"
        "输出时**先完整输出步骤 A**，然后**单独一行**输出分隔标记（必须完全一致、单独成行）：\n"
        "{phase2_marker}\n"
        "然后输出步骤 B；步骤 B 中不要重复步骤 A 的「候选文件」列表。"
    )


def _parse_optional_float(name: str) -> Optional[float]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_optional_int(name: str) -> Optional[int]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def is_retryable_llm_error(exc: BaseException) -> bool:
    """Return whether an exception chain represents a transient LLM/API failure."""
    seen: Set[int] = set()
    current: Optional[BaseException] = exc
    transient_markers = (
        "api connection error",
        "apiconnectionerror",
        "connection error",
        "connection reset",
        "connection refused",
        "server disconnected",
        "remoteprotocolerror",
        "temporarily unavailable",
        "timed out",
        "timeout",
        "read error",
        "service unavailable",
        "rate limit",
        "too many requests",
    )

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status_code = getattr(current, "status_code", None)
        if isinstance(status_code, int) and (
            status_code in {408, 409, 429} or status_code >= 500
        ):
            return True

        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )

            if isinstance(
                current,
                (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError),
            ):
                return True
        except (ImportError, TypeError):
            pass

        text = f"{type(current).__name__}: {current}".lower()
        if any(marker in text for marker in transient_markers):
            return True
        current = current.__cause__ or current.__context__
    return False


def invoke_with_transient_retries(
    operation: Callable[[], _T],
    *,
    operation_name: str,
    max_retries: int,
    initial_delay_seconds: float,
    backoff_factor: float = 2.0,
    max_delay_seconds: float = 60.0,
) -> _T:
    """Run an operation with bounded exponential backoff for transient failures.

    ``max_retries`` is the number of retries after the first attempt. The final
    exception is re-raised and annotated with attempt metadata so the batch
    runner can persist a useful per-instance failure record.
    """
    retry_index = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            attempts = retry_index + 1
            retryable = is_retryable_llm_error(exc)
            if not retryable or retry_index >= max(0, max_retries):
                try:
                    setattr(exc, "jloc_attempts", attempts)
                    setattr(exc, "jloc_retryable", retryable)
                except Exception:
                    pass
                raise

            delay = min(
                max(0.0, max_delay_seconds),
                max(0.0, initial_delay_seconds)
                * max(1.0, backoff_factor) ** retry_index,
            )
            retry_index += 1
            print(
                f"[retry] {operation_name} transient failure; "
                f"retry {retry_index}/{max_retries} in {delay:.1f}s: "
                f"{type(exc).__name__}: {str(exc)[:300]}",
                flush=True,
            )
            if delay > 0:
                time.sleep(delay)


def get_llm_runtime_provenance() -> Dict[str, Any]:
    """
    写入 eval ``run_provenance``：采样参数与「请求」的模型名（实际 id 以各次响应 ``models_reported`` 为准）。
    """
    prov: Dict[str, Any] = {
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
        "llm_temperature_env": os.environ.get("LLM_TEMPERATURE"),
        "llm_seed_env": os.environ.get("LLM_SEED"),
        "ollama_model": os.getenv("OLLAMA_MODEL"),
        "ollama_num_ctx_env": os.getenv("OLLAMA_NUM_CTX"),
        "ollama_url_set": bool(os.getenv("OLLAMA_URL")),
        "llm_model_env": os.getenv("LLM_MODEL"),
    }
    try:
        import langchain_core

        prov["langchain_core_version"] = getattr(langchain_core, "__version__", "")
    except Exception:
        prov["langchain_core_version"] = ""
    try:
        import langchain_ollama

        prov["langchain_ollama_version"] = getattr(langchain_ollama, "__version__", "")
    except Exception:
        prov["langchain_ollama_version"] = ""
    try:
        import langchain_openai

        prov["langchain_openai_version"] = getattr(langchain_openai, "__version__", "")
    except Exception:
        prov["langchain_openai_version"] = ""
    prov["llm_request_timeout_sec_env"] = os.environ.get("JLOC_LLM_REQUEST_TIMEOUT_SEC")
    prov["llm_request_timeout_sec_effective"] = resolve_llm_request_timeout_sec(
        prov["llm_provider"]
    )
    return prov


def _get_llm(
    *,
    ollama_num_ctx_override: Optional[int] = None,
    temperature_override: Optional[float] = None,
    seed_override: Optional[int] = None,
):
    # Prefer OPENAI-compatible providers (DeepSeek etc.) via ChatOpenAI + base_url.
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    temperature = temperature_override if temperature_override is not None else _parse_optional_float("LLM_TEMPERATURE")
    seed = seed_override if seed_override is not None else _parse_optional_int("LLM_SEED")

    timeout_sec = resolve_llm_request_timeout_sec(llm_provider)

    if llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
        base_url = os.getenv("OLLAMA_URL", None)
        num_ctx = ollama_num_ctx_override or _parse_optional_int("OLLAMA_NUM_CTX")
        kwargs: Dict[str, Any] = {"model": model}
        if base_url:
            kwargs["base_url"] = base_url
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed
        if num_ctx is not None and num_ctx > 0:
            kwargs["num_ctx"] = num_ctx
        if timeout_sec is not None:
            sk = dict(kwargs.get("sync_client_kwargs") or {})
            ak = dict(kwargs.get("async_client_kwargs") or {})
            sk.setdefault("timeout", timeout_sec)
            ak.setdefault("timeout", timeout_sec)
            kwargs["sync_client_kwargs"] = sk
            kwargs["async_client_kwargs"] = ak
        return ChatOllama(**kwargs)

    model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_API_URL")
    kwargs2: Dict[str, Any] = {"model": model, "api_key": api_key, "base_url": base_url}
    if temperature is not None:
        kwargs2["temperature"] = temperature
    if seed is not None:
        kwargs2["seed"] = seed
    if timeout_sec is not None:
        kwargs2["timeout"] = timeout_sec
    return ChatOpenAI(**kwargs2)


def invoke_llm_with_ollama_fallback(
    prompt: str,
    *,
    call_site: str,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    ollama_num_ctx: Optional[int] = None,
):
    """
    Invoke the configured chat model.

    Ollama can terminate its runner with HTTP 500 when the requested context is
    too large for the available memory. In that case, retry once with a smaller
    context instead of killing the whole batch run.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    max_retry = _parse_optional_int("JLOC_LLM_RETRY_MAX") or 0
    retry_sleep = _parse_optional_float("JLOC_LLM_RETRY_SLEEP_SEC") or 0.0
    retry_backoff = _parse_optional_float("JLOC_LLM_RETRY_BACKOFF_FACTOR") or 2.0
    retry_max_sleep = _parse_optional_float("JLOC_LLM_RETRY_MAX_SLEEP_SEC") or 60.0

    def _invoke_once():
        # Recreate the client for every outer attempt. This avoids reusing a
        # stale HTTP connection after a vLLM worker/server restart.
        llm = _get_llm(
            ollama_num_ctx_override=ollama_num_ctx,
            temperature_override=temperature,
            seed_override=seed,
        )
        return llm.invoke(prompt)

    if provider != "ollama":
        resp = invoke_with_transient_retries(
            _invoke_once,
            operation_name=f"LLM call {call_site}",
            max_retries=max_retry,
            initial_delay_seconds=retry_sleep,
            backoff_factor=retry_backoff,
            max_delay_seconds=retry_max_sleep,
        )
        record_llm_ai_message(resp, call_site=call_site)
        return resp

    last_exc: Optional[Exception] = None
    try:
        resp = _invoke_once()
        record_llm_ai_message(resp, call_site=call_site)
        return resp
    except Exception as exc:
        last_exc = exc

    current_ctx = _parse_optional_int("OLLAMA_NUM_CTX") or 0
    fallback_ctx = _parse_optional_int("OLLAMA_NUM_CTX_FALLBACK") or 8192
    if provider != "ollama" or current_ctx <= fallback_ctx:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM invoke failed without exception context.")
    msg = str(last_exc).lower() if last_exc is not None else ""
    ollama_runner_failed = (
        "llama runner process has terminated" in msg
        or "status code: 500" in msg
        or "internal server error" in msg
    )
    if not ollama_runner_failed:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM invoke failed without exception context.")
    print(
        f"[warn] Ollama failed at num_ctx={current_ctx}; retrying {call_site} with num_ctx={fallback_ctx}: {last_exc}",
        flush=True,
    )
    retry_llm = _get_llm(
        ollama_num_ctx_override=fallback_ctx,
        temperature_override=temperature,
        seed_override=seed,
    )
    resp = retry_llm.invoke(prompt)
    record_llm_ai_message(resp, call_site=f"{call_site}_ollama_ctx_fallback_{fallback_ctx}")
    return resp


def run_localize_issue(
    issue_text: str,
    graph_context: str,
    few_shots: str,
    k: int = 20,
    refinement_feedback: Optional[str] = None,
    valid_locations: Optional[Dict[str, Set[int]]] = None,
    granularity: str = "file",
    llm_temperature: Optional[float] = None,
    llm_seed: Optional[int] = None,
    ollama_num_ctx: Optional[int] = None,
    return_prompt: bool = False,
) -> Union[str, Tuple[str, str]]:
    """
    LocalizeAgent 使用的多粒度双段输出（分隔符 ``PHASE2_MARKER`` 不变）：

    - file: 步骤 A 文件候选，步骤 B 函数候选。
    - function: 步骤 A 函数候选，步骤 B 行级候选。
    - line: 步骤 A 文件候选，步骤 B 行级候选。

    ``return_prompt=True`` 时返回 ``(model_output, full_prompt_string)``，供 SessionMemory 全量落盘。
    """
    refinement_feedback = (refinement_feedback or "").strip()
    refinement_block = ""
    if refinement_feedback:
        refinement_block = (
            "\n====================\n"
            "上轮验证反馈（必须认真对待：调整怀疑列表与 Top-K；仍只能引用 GraphRAG 已出现的路径与行号）\n"
            f"{refinement_feedback}\n"
        )

    # 有效位置白名单块：若提供则注入，约束模型只引用上下文中实际存在的行号
    valid_hint_block = ""
    if valid_locations:
        hint_lines = [
            "\n====================",
            "【有效位置白名单】以下文件与行号来自 GraphRAG 上下文中的实际 Line: 字段。",
            "阶段 2 行号必须来自此列表；若某文件不在此列表则上下文中无该文件，禁止在阶段 2 为其报任何行号：",
        ]
        for f in sorted(valid_locations.keys())[: int(os.environ.get("JLOC_VALID_LOC_FILES_CAP", "18") or "18")]:
            lns = sorted(valid_locations[f])
            if len(lns) <= 30:
                ln_str = ", ".join(str(n) for n in lns)
            else:
                ln_str = ", ".join(str(n) for n in lns[:30]) + f" …(共 {len(lns)} 行)"
            hint_lines.append(f"  {f}: [{ln_str}]")
        valid_hint_block = "\n".join(hint_lines) + "\n"

    prompt_template = ChatPromptTemplate.from_template(
        """你是资深 Java Bug 定位专家（LocalizeAgent）。你必须连续完成 **步骤 A** 与 **步骤 B**，且**严格遵守路径与行号来源**。

背景：RetrievalAgent 提供的 GraphRAG 上下文融合了向量检索（dense embedding）、BM25、RRF 融合与代码图上的 k-hop 结构扩展（类—方法—引用等）。请利用与 Issue 关键词共现的邻接节点，优先关注 **核心库实现**，压低纯测试/基准路径除非 Issue 明确只关它们。

{granularity_hint}

====================
{output_structure}

====================
Issue 描述
{issue_text}
{few_shots_block}
{refinement_block}
{valid_hint_block}
====================
GraphRAG 上下文（RetrievalAgent）
{graph_context}
"""
    )

    few_shots_block = ""
    fs = (few_shots or "").strip()
    if fs:
        few_shots_block = (
            "\n====================\nFew-shot 历史记忆\n"
            + fs
            + "\n"
        )

    prompt = prompt_template.format(
        issue_text=issue_text,
        few_shots_block=few_shots_block,
        refinement_block=refinement_block,
        valid_hint_block=valid_hint_block,
        graph_context=graph_context,
        k=k,
        phase2_marker=PHASE2_MARKER,
        granularity_hint=_granularity_hint(granularity),
        output_structure=_output_structure(granularity).format(
            k=k,
            phase2_marker=PHASE2_MARKER,
        ),
    )
    resp = invoke_llm_with_ollama_fallback(
        prompt,
        call_site="run_localize_issue",
        temperature=llm_temperature,
        seed=llm_seed,
        ollama_num_ctx=ollama_num_ctx,
    )
    out = (resp.content or "").strip()
    if out.startswith('"') and out.endswith('"'):
        out = out[1:-1].strip()
    if return_prompt:
        return out, prompt
    return out


@tool
def localize_issue_with_graph_rag_memory(
    issue_text: str,
    graph_context: str,
    few_shots: str,
    k: int = 20,
    refinement_feedback: str = "",
) -> str:
    """
    Localize Java issue by combining GraphRAG context, few-shots, and optional refinement feedback.
    """
    return run_localize_issue(
        issue_text=issue_text,
        graph_context=graph_context,
        few_shots=few_shots,
        k=k,
        refinement_feedback=refinement_feedback or None,
    )
