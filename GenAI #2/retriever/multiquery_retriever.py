from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma.from_documents(docs, embeddings)

AImodel = ChatMistralAI(model_name="mistral-small-latest")

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":3}
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=AImodel
)

docs = multi_query_retriever.invoke("What is gradient descent?")

for doc in docs:
    print(doc.page_content)