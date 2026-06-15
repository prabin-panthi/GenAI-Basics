from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings



docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "Ai book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "Data book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL book"})
]

embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="GenAI #2/test-db"
)

result = vectorstore.similarity_search("what is used for data analysis?", k=2)

for r in result:
    print(r)