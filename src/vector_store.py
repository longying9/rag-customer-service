"""
向量数据库管理，基于 ChromaDB。
"""

import os
import shutil
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.config import Config

_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """获取嵌入模型实例（单例）。首次调用会下载模型。"""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def create_vector_store(
    documents: List[Document],
    collection_name: str = "customer_service_kb",
) -> Chroma:
    """从文档创建新的向量数据库。"""
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        persist_directory=Config.CHROMA_PERSIST_DIR,
        collection_name=collection_name,
    )
    return vector_store


def load_vector_store(
    collection_name: str = "customer_service_kb",
) -> Optional[Chroma]:
    """加载已有的向量数据库。"""
    persist_dir = Config.CHROMA_PERSIST_DIR
    if not os.path.exists(persist_dir):
        return None

    try:
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=get_embeddings(),
            collection_name=collection_name,
        )
    except Exception as e:
        print(f"[错误] 加载向量数据库失败: {e}")
        return None


def add_documents(
    documents: List[Document],
    collection_name: str = "customer_service_kb",
) -> int:
    """向现有向量数据库添加文档。返回已存储的文档块数量。"""
    store = load_vector_store(collection_name)
    if store is None:
        store = create_vector_store(documents, collection_name)
        return len(documents)
    store.add_documents(documents)
    return len(documents)


def clear_vector_store() -> bool:
    """删除本地向量数据库。"""
    persist_dir = Config.CHROMA_PERSIST_DIR
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        return True
    return False


def get_store_stats() -> dict:
    """获取向量数据库的统计信息。"""
    store = load_vector_store()
    if store is None:
        return {"exists": False, "document_count": 0}
    try:
        count = store._collection.count()
        return {"exists": True, "document_count": count}
    except Exception:
        return {"exists": True, "document_count": "未知"}
