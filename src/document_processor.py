"""
文档加载与分块处理。
支持 PDF、TXT、Markdown 格式。
"""

import os
from pathlib import Path
from typing import List

from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from src.config import Config

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def load_single_document(file_path: str) -> List[Document]:
    """根据文件扩展名加载单个文档。"""
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式: {ext}，支持的格式: {SUPPORTED_EXTENSIONS}")

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    return loader.load()


def load_documents(file_paths: List[str]) -> List[Document]:
    """批量加载多个文档。"""
    all_docs = []
    errors = []
    for fp in file_paths:
        try:
            all_docs.extend(load_single_document(fp))
        except Exception as e:
            errors.append(f"加载 {os.path.basename(fp)} 失败: {e}")
    if errors:
        error_msg = "\n".join(errors)
        print(f"[警告] 部分文档加载失败:\n{error_msg}")
    return all_docs


def split_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """将文档切分为适合检索的文本块。"""
    chunk_size = chunk_size or Config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    return chunks
