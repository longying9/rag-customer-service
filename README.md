# 🤖 RAG 智能客服系统

基于 **RAG（检索增强生成）** 技术的智能客服系统，支持自定义知识库问答。使用 LangChain + ChromaDB + Streamlit 构建，可一键部署运行。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ 功能特性

- 📁 **多格式文档支持** — 支持上传 PDF、TXT、Markdown 文档构建知识库
- 🔍 **语义检索** — 基于向量相似度的智能文档检索，精准定位相关信息
- 💬 **智能问答** — 结合知识库内容生成准确、专业的回答
- 🧠 **对话记忆** — 支持多轮对话，上下文连贯
- 📎 **来源追溯** — 每个回答均附参考文档来源，可验证可追溯
- 🌐 **Web 界面** — 基于 Streamlit 的现代化聊天界面，开箱即用
- 🔌 **多模型支持** — 支持通义千问 (DashScope) 和 OpenAI 兼容接口
- 📦 **一键启动** — 内置示例知识库，0 配置即可体验

## 🏗️ 技术架构

```
┌─────────────────────────────────────────┐
│              Streamlit UI               │
│    (文档上传 / 聊天界面 / 配置面板)        │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│            RAG Chain (LCEL)             │
│   Prompt Template → LLM → Output Parser │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        ChromaDB Vector Store            │
│     (文档向量存储 / 语义相似度搜索)       │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│       Embedding Model (Local)           │
│   paraphrase-multilingual-MiniLM-L12-v2 │
└─────────────────────────────────────────┘
```

**核心技术栈：**

| 组件 | 技术 | 说明 |
|------|------|------|
| LLM | 通义千问 (Qwen) / OpenAI | 支持 qwen-plus/max/turbo 及 GPT 系列 |
| 嵌入模型 | Sentence Transformers | 本地运行，无需 API，支持中英文 |
| 向量数据库 | ChromaDB | 轻量级，本地持久化存储 |
| 框架 | LangChain + LCEL | 链式调用，模块化设计 |
| 前端 | Streamlit | 纯 Python Web UI，快速部署 |

## 🚀 快速开始

### 环境要求

- Python 3.10+
- pip

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/longying9/rag-customer-service.git
cd rag-customer-service

# 2. 创建虚拟环境（推荐）
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API 密钥
cp .env.example .env
# 编辑 .env 文件，填入你的 DASHSCOPE_API_KEY

# 5. 启动应用
streamlit run app.py
```

启动后浏览器访问 `http://localhost:8501`，点击左侧「加载示例数据」即可体验。

### 获取 API 密钥

**通义千问 (推荐国内用户):**
1. 访问 [阿里云 DashScope](https://dashscope.aliyun.com/)
2. 注册/登录 → 控制台 → API Key 管理
3. 创建 API Key 并填入 `.env` 文件

**OpenAI (海外用户):**
1. 修改 `.env` 中 `LLM_PROVIDER=openai`
2. 填入 `OPENAI_API_KEY`

## 📖 使用指南

### 构建知识库

1. 准备你的文档（PDF / TXT / Markdown）
2. 在左侧边栏上传文档
3. 点击「导入到知识库」
4. 等待处理完成，状态栏显示文档块数量

### 开始问答

上传完成后直接在聊天框输入问题，AI 将基于知识库内容回答。

**示例问题（加载示例数据后）:**
- "ZX-OS 3.0 支持哪些车型？"
- "如何申请售后服务？"
- "产品质保政策是什么？"
- "数据隐私如何保障？"

### 模型切换

左侧边栏可以随时切换 LLM 提供商和模型名称。

## 📁 项目结构

```
rag-customer-service/
├── app.py                      # Streamlit 主程序
├── src/
│   ├── __init__.py
│   ├── config.py               # 全局配置管理
│   ├── document_processor.py   # 文档加载与分块
│   ├── vector_store.py         # ChromaDB 向量数据库
│   └── rag_chain.py            # RAG 核心链与问答逻辑
├── data/
│   └── sample_knowledge.md     # 示例知识库（智能出行公司）
├── .env.example                # 环境变量模板
├── .gitignore
├── requirements.txt
└── README.md
```

## 🔧 配置说明

`.env` 文件中的主要配置项：

```ini
# LLM 提供商: dashscope 或 openai
LLM_PROVIDER=dashscope

# DashScope API 密钥
DASHSCOPE_API_KEY=your_key_here

# 模型选择: qwen-plus / qwen-max / qwen-turbo
LLM_MODEL=qwen-plus

# 文档分块大小（字符数）
CHUNK_SIZE=500

# 检索返回文档数
RETRIEVER_TOP_K=3
```

## 📝 后续扩展方向

- [ ] 支持更多文档格式（Word, HTML, CSV）
- [ ] 添加用户认证和对话历史持久化
- [ ] 支持知识库多集合管理
- [ ] 添加 RAG 评估指标（命中率、准确率）
- [ ] Docker 一键部署
- [ ] 导出对话记录

## 📄 License

MIT License

---

