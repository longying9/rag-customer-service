"""
全局配置管理，所有配置通过环境变量读取。
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """集中管理所有配置项。"""

    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "dashscope")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen-plus")
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    # Embedding
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Document processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Vector store
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./vector_db")

    # Retriever
    RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "3"))

    @classmethod
    def validate(cls) -> list[str]:
        """验证必要的配置是否完整，返回缺失项列表。"""
        errors = []
        if cls.LLM_PROVIDER == "dashscope" and not cls.DASHSCOPE_API_KEY:
            errors.append("DASHSCOPE_API_KEY 未设置")
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY 未设置")
        return errors
