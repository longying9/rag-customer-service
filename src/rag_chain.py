"""
RAG 检索增强生成链，核心问答逻辑。
"""

from typing import List, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document
from src.config import Config

SYSTEM_PROMPT = """你是一个专业、友好的智能客服助手。请严格按照以下规则回答问题：

## 规则
1. **仅根据知识库内容回答**，不要编造或猜测信息
2. 如果知识库中有相关信息，请**完整、准确地**回答用户
3. 如果知识库中**没有相关信息**，请诚实地说："抱歉，我目前的知识库中没有关于这个问题的信息。建议您联系人工客服获取更多帮助。"
4. 回答要**简洁清晰**，使用友好、专业的语气
5. 如果适用，可以使用**列表或分段**来组织信息

## 知识库内容
{context}

## 对话历史
{history}
"""

USER_PROMPT = """用户问题：{question}

请根据上述知识库内容回答。"""


def get_llm():
    """根据配置创建 LLM 实例。"""
    provider = Config.LLM_PROVIDER
    model = Config.LLM_MODEL

    if provider == "dashscope":
        from langchain_community.chat_models import ChatTongyi

        return ChatTongyi(model=model, temperature=0.3)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=0.3,
            base_url=Config.OPENAI_API_BASE,
            api_key=Config.OPENAI_API_KEY,
        )
    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")


def _format_docs(docs: List[Document]) -> str:
    """将检索到的文档格式化为上下文字符串。"""
    if not docs:
        return "（知识库中暂无相关内容）"

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[文档{i}] 来源: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_history(history: List[Tuple[str, str]]) -> str:
    """将对话历史格式化为字符串。"""
    if not history:
        return "（暂无对话历史）"

    lines = []
    for q, a in history[-5:]:  # 最多保留最近5轮对话
        lines.append(f"用户: {q}\n客服: {a}")
    return "\n".join(lines)


def create_rag_chain(vector_store):
    """创建 RAG 问答链。"""
    retriever = vector_store.as_retriever(
        search_kwargs={"k": Config.RETRIEVER_TOP_K}
    )
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

    def retrieve_and_format(input_dict: dict) -> dict:
        question = input_dict["question"]
        history = input_dict.get("history", [])

        docs = retriever.invoke(question)
        return {
            "context": _format_docs(docs),
            "history": _format_history(history),
            "question": question,
        }

    chain = retrieve_and_format | prompt | llm | StrOutputParser()
    return chain


def ask(
    chain,
    question: str,
    history: List[Tuple[str, str]] | None = None,
    vector_store=None,
) -> Tuple[str, List[Document]]:
    """执行一次 RAG 问答。

    Returns:
        (回答文本, 检索到的来源文档列表)
    """
    history = history or []

    answer = chain.invoke({"question": question, "history": history})

    docs = []
    if vector_store:
        docs = vector_store.similarity_search(question, k=Config.RETRIEVER_TOP_K)

    return answer, docs
