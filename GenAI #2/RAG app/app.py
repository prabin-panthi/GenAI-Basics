import tempfile
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions about it.")


# ---------------------------
# LLM
# ---------------------------

llm = init_chat_model(
    "mistral-medium-3-5",
    model_provider="mistralai"
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful AI assistant.

Answer ONLY using the provided context.

Rules:
1. Use only information from context.
2. Do not use external knowledge.
3. If answer is not present, say:
"I cannot find the answer in the provided context."
4. Be concise and accurate.
5. Use page numbers if available.
        """
    ),
    (
        "human",
        """
Context:
{context}

Question:
{query}
        """
    )
])


# ---------------------------
# EMBEDDING MODEL
# ---------------------------

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


embedding_model = load_embedding_model()


# ---------------------------
# SESSION STATE INIT
# ---------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------
# PDF UPLOAD
# ---------------------------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    with st.spinner("Processing PDF..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        st.session_state.vectorstore = vectorstore

        # reset chat when new PDF is uploaded
        st.session_state.messages = []

    st.success(
        f"PDF processed successfully.\n\n"
        f"Pages: {len(documents)}\n"
        f"Chunks: {len(chunks)}"
    )


# ---------------------------
# CHAT HISTORY DISPLAY
# ---------------------------

if "vectorstore" in st.session_state:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


# ---------------------------
# CHAT INPUT (FIXED UX)
# ---------------------------

if "vectorstore" in st.session_state:

    query = st.chat_input("Ask a question about the PDF")

    if query:

        # show user message immediately
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.write(query)

        # retrieve context
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )

        with st.spinner("Retrieving relevant context..."):
            docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke({
            "context": context,
            "query": query
        })

        # LLM response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = llm.invoke(final_prompt)
                st.write(response.content)

        # save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.content
        })

        # ---------------------------
        # SOURCES
        # ---------------------------

        with st.expander("📚 Sources"):
            for i, doc in enumerate(docs, start=1):
                page = doc.metadata.get("page", "Unknown")

                st.write(f"**Source {i} | Page {page + 1}**")
                st.write(doc.page_content)
                st.divider()