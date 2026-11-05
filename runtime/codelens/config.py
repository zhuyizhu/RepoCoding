import os
from dotenv import load_dotenv
load_dotenv()

GRAPH_PATH = "/home/zzz/projects/RepoGraph/data_mine/repographs/alibaba__fastjson2.pkl"
VECTORSTORE_PATH = "data/vectorstore.faiss"

# Shared with repograph/construct_graph.py: local clone cache for org__repo/repo
REPO_CACHE_ROOT = os.getenv("REPO_CACHE_ROOT", "resultgraphs/repo_cache")

# Code / text embedding for FAISS (HuggingFace id). Examples: BAAI/bge-m3, microsoft/unixcoder-base,
# microsoft/graphcodebert-base (may need CODE_EMBEDDING_TRUST_REMOTE_CODE=1).
CODE_EMBEDDING_MODEL = os.getenv("CODE_EMBEDDING_MODEL", "BAAI/bge-m3")

# ---------------------------------------------------------------------------
# LLM Backend Configuration
# ---------------------------------------------------------------------------
# LLM_PROVIDER: "ollama" | "openai" | "deepseek"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# ── Ollama (local models) ──────────────────────────────────────────────
# qwen2.5:14b           — 通用模型，14B 参数
# qwen2.5-coder:7b      — 代码专用模型，7B 参数（推荐用于 Java 定位任务）
# qwen2.5-coder:14b     — 代码专用模型，14B 参数
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

# Qwen-Coder 7B / 14B 专用配置
QWEN_CODER_7B_MODEL = os.getenv("QWEN_CODER_7B_MODEL", "qwen2.5-coder:7b")
QWEN_CODER_14B_MODEL = os.getenv("QWEN_CODER_14B_MODEL", "qwen2.5-coder:14b")

# Ollama 上下文窗口 (token 数)
# 7B 模型建议 8192-16384，14B 模型建议 4096-8192 (取决于 VRAM)
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

# ── OpenAI-compatible API ───────────────────────────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── DeepSeek (via OpenAI-compatible API) ─────────────────────────────────
DEEPSEEK_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
DEEPSEEK_API_KEY = os.getenv("LLM_API_KEY")
DEEPSEEK_API_URL = os.getenv("LLM_API_URL", "https://api.deepseek.com")