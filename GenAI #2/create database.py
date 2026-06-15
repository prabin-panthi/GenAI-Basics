import os
# os.environ["HF_HUB_OFFLINE"] = "1"     uncommentout for offline use

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load your PDF
loader = PyPDFLoader("GenAI #2/deep learning.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# 3. Local embedding model (downloads once, ~420MB, then cached)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)

# 5. Build and save new DB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="GenAI #2/Vector_DB"
)
print(f"Done! Stored {len(chunks)} chunks into Vector DB")