# ═══════════════════════════════════════════════════════════
# API 配置
# ═══════════════════════════════════════════════════════════

# ── 阿里云百炼：Embedding（text-embedding-v4）────────────
API_KEY   = "sk-f771855105fe43b28584a0f4d68fb5e9"
API_BASE  = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_MODEL = "text-embedding-v4"

# 若使用华北2（北京）地域，替换为：
# API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# ── DeepSeek：数据生成 + LLM-as-Judge ───────────────────
DEEPSEEK_API_KEY  = "sk-51c4b4e82a0a4af9817b24c88b931b61"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL    = "deepseek-chat"          # DeepSeek-V3

# ── LLM-as-Judge（consensus.py 中 llm_judge 方法使用）──
LLM_API_KEY = DEEPSEEK_API_KEY
LLM_API_URL = DEEPSEEK_API_BASE + "/chat/completions"
LLM_MODEL   = DEEPSEEK_MODEL

# ── 其他 ────────────────────────────────────────────────
DEVICE      = "cpu"
ENABLE_BERT = False   # 已有 Embedding API，无需加载本地 BERT
