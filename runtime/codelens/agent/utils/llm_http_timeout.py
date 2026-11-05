"""
单次 LLM HTTP 读超时：避免远程 API 长时间无响应时进程一直阻塞。

环境变量 ``JLOC_LLM_REQUEST_TIMEOUT_SEC``：
- 未设置：对非 ``ollama`` 的提供商默认 ``1200`` 秒；``ollama`` 默认不限制（本地生成可能很久）。
- 正数：使用该秒数作为 httpx/OpenAI 客户端请求超时。
- ``0`` / ``none`` / ``off`` / ``disable`` / 空字符串：显式关闭超时（与旧行为一致）。
"""

from __future__ import annotations

import os
from typing import Optional

_DEFAULT_REMOTE_LLM_TIMEOUT_SEC = 1200.0


def resolve_llm_request_timeout_sec(llm_provider: Optional[str] = None) -> Optional[float]:
    """
    返回传给 ChatOpenAI ``timeout`` 或 Ollama ``Client``/``AsyncClient`` 的 ``timeout`` 秒数；``None`` 表示不限制。
    """
    prov = (llm_provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
    is_ollama = prov == "ollama"
    raw = os.environ.get("JLOC_LLM_REQUEST_TIMEOUT_SEC")
    if raw is not None:
        s = str(raw).strip().lower()
        if s in ("0", "none", "off", "disable", ""):
            return None
        try:
            v = float(s)
            if v <= 0:
                return None
            return v
        except ValueError:
            pass
    if is_ollama:
        return None
    return _DEFAULT_REMOTE_LLM_TIMEOUT_SEC
