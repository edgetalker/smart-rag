import os
from dotenv import load_dotenv

load_dotenv()
class Config:
    """ 后端统一配置管理 """

    # API配置
    HOST: str = os.getenv("HOST", "localhost")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # LLM配置
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("EMBEDDING_MODEL_NAME")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION"))

    # Qdrant配置
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION")

    # Elasticsearch配置
    ES_HOST: str = os.getenv("ES_HOST", "localhost")
    ES_PORT: int = int(os.getenv("ES_PORT", 9200))
    ES_INDEX: str = os.getenv("ES_INDEX")

    # 检索配置
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", 5))

    # Context Manager配置
    ENABLE_CONTEXT_OPTIMIZATION: bool = True  # 是否启用上下文优化
    MAX_CONTEXT_RESULTS: int = 15  # 最大结果数量
    ENABLE_QUALITY_FILTER: bool = True  # 启用质量筛选
    ENABLE_DEDUPLICATION: bool = True  # 启用去重

    # Gemini特定配置
    GEMINI_MAX_CONTEXT_TOKENS: int = 1000000  # Gemini-1.5-Flash上下文限制

    # 缓存配置
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", 3600))

    @classmethod
    def validate(cls):
        """验证关键配置"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")

        return True


# 全局配置实例
config = Config()

# 验证配置
try:
    config.validate()
    print("✅ Configuration validated successfully")
except ValueError as e:
    print(f"❌ Configuration error: {e}")
    exit(1)