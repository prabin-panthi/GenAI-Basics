# 🤖 GenAI Learning Journey

A personal learning repository documenting my step-by-step exploration of **Generative AI**, **LangChain**, and **LLM-powered applications** — from basic chat models to full RAG pipelines and agentic systems.

---

## 📁 Folder Structure

```
GenAI Learning/
├── GenAI #1/               # Foundations: Chat models, embeddings, prompt engineering
├── GenAI #2/               # RAG: Vector databases, document loaders, retrievers
├── GenAI #3/               # Advanced: Agents, tools, runnables, LCEL chains
├── .env.example            # API key template (never commit .env!)
├── .gitignore
├── requirements.txt        # All dependencies
└── README.md
```

---

## 🗂️ GenAI #1 — Foundations

> Getting comfortable with LangChain, chat models, message history, embeddings, and Streamlit UIs.

```
GenAI #1/
├── chatmodels/
│   ├── chat.py                  # Invoking Mistral, Groq, and Gemini models side by side
│   ├── chatbot_core.py          # Simple CLI chatbot with message history loop
│   ├── chatbot_LangMessages.py  # Using LangChain HumanMessage / AIMessage types
│   ├── ChatBot_UI.py            # Streamlit chatbot UI
│   ├── huggingFace.py           # HuggingFace model integration
│   └── localmodel.py            # Running a local LLM
├── embeddingmodel/
│   └── embed.py                 # HuggingFace sentence-transformer embeddings
└── Company/
    ├── summaryApp.py            # Streamlit app: extract features from movie paragraphs
    ├── core.py                  # Core prompt/chain logic
    ├── json_summary.py          # Structured JSON output parsing
    └── paragraphs.txt           # Sample input text
```

**Key concepts learned:**
- Initializing and invoking multiple LLM providers (Mistral, Groq, Gemini) with `init_chat_model`
- Building stateful CLI chatbots using message lists
- `ChatPromptTemplate` for structured prompting
- HuggingFace embedding models (`sentence-transformers/all-mpnet-base-v2`)
- Streamlit for quick AI app UIs

---

## 🗂️ GenAI #2 — RAG (Retrieval-Augmented Generation)

> Building document-aware AI: loading PDFs, chunking text, storing in vector DBs, and retrieving relevant context.

```
GenAI #2/
├── document loaders.py          # Loading various document types with LangChain
├── text splitters.py            # RecursiveCharacterTextSplitter experiments
├── vector DB.py                 # Chroma vector store basics
├── create database.py           # Full pipeline: PDF → chunks → ChromaDB
├── main.py                      # Query the vector DB with a question
├── RAG app/
│   └── app.py                   # Full Streamlit RAG app (upload PDF → ask questions)
├── retriever/
│   ├── arxiv_retriever.py       # Fetching papers from ArXiv
│   ├── mmr_retriever.py         # MMR (Max Marginal Relevance) retrieval
│   └── multiquery_retriever.py  # Multi-query retrieval for better recall
├── Vector_DB/                   # Persisted ChromaDB (deep learning PDF)
├── test-db/                     # Test vector database
├── deep learning.pdf            # Source document used for RAG
├── MySQL Handbook.pdf           # Additional reference material
└── text_file.txt                # Plain text source for experiments
```

**Key concepts learned:**
- Document loaders (PDF, text) and text splitting strategies
- ChromaDB as a local persistent vector store
- Embedding + similarity search pipeline
- Building a full RAG app with Streamlit (upload any PDF → Q&A)
- Retrieval strategies: similarity, MMR, multi-query

---

## 🗂️ GenAI #3 — Agents, Tools & LCEL Chains

> LangChain Expression Language (LCEL), custom tools, and autonomous agents that can plan and act.

```
GenAI #3/
├── runnable/
│   ├── sequemce_runnable.py     # LCEL pipe chains: prompt | model | parser
│   ├── parallel_runnable.py     # Running multiple chains in parallel
│   └── passthrough_runnable.py  # RunnablePassthrough for passing data through
├── Tools_/
│   ├── tool_calling.py          # Binding tools to LLMs
│   ├── custom_tool.py           # Defining custom @tool functions
│   └── newsSummarizer.py        # Tavily search → summarize news with LCEL
└── Agents/
    ├── Basic weather api.py         # Simple OpenWeather API tool
    ├── self_agent.py                # Agent with self-reasoning loop
    ├── human_in_the_loop_agents.py  # Agent that pauses for human approval
    └── Agent ChatBot.py             # Streamlit agent chatbot (weather + Tavily search)
```

**Key concepts learned:**
- LCEL (LangChain Expression Language): composing chains with `|` operator
- `RunnableParallel` and `RunnablePassthrough`
- Creating and binding custom `@tool` functions to LLMs
- Tool-calling with real APIs (OpenWeatherMap, Tavily search)
- Building autonomous agents with a reasoning loop
- Human-in-the-loop agents for supervised decision-making
- Streamlit agent chatbot combining multiple tools

---

## 🔑 API Keys Used

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Key | Provider | Used For |
|-----|----------|----------|
| `GROQ_API_KEY` | [Groq](https://console.groq.com) | Fast LLM inference |
| `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com) | Gemini models |
| `MISTRAL_API_KEY` | [Mistral](https://console.mistral.ai) | Primary chat model |
| `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` | [HuggingFace](https://huggingface.co/settings/tokens) | Embeddings & local models |
| `TAVILY_API_KEY` | [Tavily](https://tavily.com) | Web search tool for agents |
| `OPENWEATHER_API_KEY` | [OpenWeatherMap](https://openweathermap.org/api) | Weather tool for agents |

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd genai-learning

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows with python version 3.11.9
# source .venv/bin/activate  # Linux/Mac with python version 3.11.9

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Fill in your API keys in .env

# 5. Run any script
python "GenAI #1/chatmodels/chat.py"

# Or run a Streamlit app
streamlit run "GenAI #2/RAG app/app.py"
```

---

## 🧠 Learning Progression

```
GenAI #1  →  GenAI #2  →  GenAI #3
   │              │              │
Chat Models    Documents      Agents
Embeddings     Vector DBs     Tools
Prompts        RAG Pipeline   LCEL Chains
Streamlit UI   Retrievers     Autonomous AI
```

---

## 🛠️ Main Tech Stack

- **LangChain** — core framework for chaining LLM components
- **LangGraph** — agent orchestration
- **Mistral AI** — primary LLM provider
- **Groq** — fast inference
- **Google Gemini** — multimodal model experiments
- **HuggingFace** — open-source embeddings
- **ChromaDB** — local vector database
- **Streamlit** — rapid AI app UIs
- **Tavily** — web search API for agents

---

*Personal learning repository — built step by step while exploring Generative AI.* 🚀
