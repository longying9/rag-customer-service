"""
RAG 智能客服系统 - Streamlit Web 界面

使用方法:
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="RAG 智能客服",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- 初始化 session state ----------
def init_session():
    defaults = {
        "messages": [],  # 聊天记录: [{"role": "user"/"assistant", "content": ..., "sources": [...]}]
        "history": [],  # RAG 用: [(question, answer), ...]
        "chain": None,
        "vector_store": None,
        "kb_ready": False,
        "kb_stats": {"exists": False, "document_count": 0},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()


# ---------- 延迟导入 (避免首次启动过慢) ----------
@st.cache_resource
def get_embeddings():
    from src.vector_store import get_embeddings as _get

    return _get()


@st.cache_resource
def get_rag_components():
    """缓存 RAG 组件，避免重复初始化。"""
    from src.vector_store import load_vector_store
    from src.rag_chain import create_rag_chain

    vs = load_vector_store()
    if vs is None:
        return None, None
    chain = create_rag_chain(vs)
    return vs, chain


def refresh_kb_state():
    """刷新知识库状态。"""
    from src.vector_store import get_store_stats

    stats = get_store_stats()
    st.session_state.kb_stats = stats
    st.session_state.kb_ready = stats["exists"] and stats["document_count"] > 0
    if st.session_state.kb_ready:
        vs, chain = get_rag_components()
        st.session_state.vector_store = vs
        st.session_state.chain = chain
    else:
        st.session_state.vector_store = None
        st.session_state.chain = None


# ---------- 侧边栏 ----------
with st.sidebar:
    st.title("⚙️ 配置面板")

    # ---- 文档上传 ----
    st.subheader("📁 知识库管理")

    uploaded_files = st.file_uploader(
        "上传文档（PDF / TXT / Markdown）",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="上传的文档将被切分并存入向量数据库",
    )

    col1, col2 = st.columns(2)
    if col1.button("📥 导入到知识库", use_container_width=True, disabled=not uploaded_files):
        with st.spinner("正在处理文档..."):
            from src.document_processor import load_documents, split_documents
            from src.vector_store import add_documents, clear_vector_store

            saved_files = []
            for uf in uploaded_files:
                tmp_path = Path(tempfile.gettempdir()) / uf.name
                with open(tmp_path, "wb") as f:
                    f.write(uf.getbuffer())
                saved_files.append(str(tmp_path))

            docs = load_documents(saved_files)
            if not docs:
                st.error("文档加载失败，请检查文件格式。")
            else:
                chunks = split_documents(docs)
                count = add_documents(chunks)
                st.success(f"✅ 成功导入 {len(saved_files)} 个文件，共 {count} 个文本块")

                # 清理临时文件
                for p in saved_files:
                    try:
                        os.remove(p)
                    except OSError:
                        pass

                # 清除缓存以重新加载
                get_rag_components.clear()
                refresh_kb_state()

    if col2.button("🗑️ 清空知识库", use_container_width=True):
        from src.vector_store import clear_vector_store

        if clear_vector_store():
            get_rag_components.clear()
            refresh_kb_state()
            st.session_state.messages = []
            st.session_state.history = []
            st.success("知识库已清空")
            st.rerun()
        else:
            st.info("知识库原本就是空的")

    # ---- 知识库状态 ----
    refresh_kb_state()
    stats = st.session_state.kb_stats
    if stats["exists"] and stats["document_count"] > 0:
        st.success(f"📊 已存储 {stats['document_count']} 个文本块")
    else:
        st.info("📊 知识库为空，请上传文档")

    # ---- 示例数据 ----
    st.divider()
    if st.button("📦 加载示例数据", use_container_width=True):
        from src.document_processor import load_documents, split_documents
        from src.vector_store import clear_vector_store, add_documents

        sample_path = Path(__file__).parent / "data" / "sample_knowledge.md"
        if sample_path.exists():
            clear_vector_store()
            docs = load_documents([str(sample_path)])
            chunks = split_documents(docs)
            add_documents(chunks)
            get_rag_components.clear()
            refresh_kb_state()
            st.session_state.messages = []
            st.session_state.history = []
            st.success("✅ 示例数据已加载")
            st.rerun()
        else:
            st.error("示例数据文件不存在")

    # ---- 模型配置 ----
    st.divider()
    st.subheader("🤖 模型配置")
    from src.config import Config

    provider = st.selectbox("LLM 提供商", ["dashscope", "openai"], index=0 if Config.LLM_PROVIDER == "dashscope" else 1)
    model = st.text_input("模型名称", value=Config.LLM_MODEL)

    if provider != Config.LLM_PROVIDER or model != Config.LLM_MODEL:
        Config.LLM_PROVIDER = provider
        Config.LLM_MODEL = model
        get_rag_components.clear()


# ---------- 主界面 ----------
st.title("🤖 RAG 智能客服系统")
st.caption("基于检索增强生成（RAG）的智能客服助手 — 上传知识库文档后即可开始问答")

# 检查 API Key
errors = Config.validate()
if errors:
    for err in errors:
        st.error(f"⚠️ 配置缺失: {err}")
    st.info("请复制 .env.example 为 .env 并填入你的 API 密钥。", icon="💡")
    st.stop()

# 检查知识库
if not st.session_state.kb_ready:
    st.warning("📢 知识库为空，请先上传文档或加载示例数据（左侧边栏）")
    st.stop()

# ---------- 聊天区域 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 参考来源"):
                for i, doc in enumerate(msg["sources"], 1):
                    src = doc.metadata.get("source", "未知来源")
                    st.caption(f"**来源 {i}:** {src}")
                    st.text(doc.page_content[:500])

# ---------- 输入框 ----------
if question := st.chat_input("请输入您的问题..."):

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                from src.rag_chain import ask

                answer, sources = ask(
                    chain=st.session_state.chain,
                    question=question,
                    history=st.session_state.history,
                    vector_store=st.session_state.vector_store,
                )
            except Exception as e:
                answer = f"❌ 出错了: {e}"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("📎 参考来源"):
                for i, doc in enumerate(sources, 1):
                    src = doc.metadata.get("source", "未知来源")
                    st.caption(f"**来源 {i}:** {src}")
                    st.text(doc.page_content[:500])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.session_state.history.append((question, answer))

# ---------- 底部操作 ----------
st.divider()
col1, col2, col3 = st.columns(3)
if col1.button("🔄 清空对话", use_container_width=True):
    st.session_state.messages = []
    st.session_state.history = []
    st.rerun()
if col2.button("🗑️ 清空知识库", use_container_width=True):
    from src.vector_store import clear_vector_store

    clear_vector_store()
    get_rag_components.clear()
    refresh_kb_state()
    st.session_state.messages = []
    st.session_state.history = []
    st.rerun()
