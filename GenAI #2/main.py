from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma(
    embedding_function= embedding_model,
    persist_directory="GenAI #2/Vector_DB"
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)

llm = init_chat_model("mistral-medium-3-5", model_provider="mistralai")
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant that answers questions strictly from the provided context.

        Rules:
        1. Use ONLY the exact information present in the context below.
        2. Do NOT add any information from your own knowledge.
        3. If the answer is not fully present in the context, say exactly:
        "I cannot find the answer in the provided context."
        4. Do not infer, assume, or expand beyond what is explicitly stated.
        5. Quote or closely paraphrase the context when answering.
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

print(f"---------------Welcome to chatbot on deep leaning---------------(press 0 to end)\n\n")
while True:
    query = input("You :  ")
    if query == "0":
        break

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    final_promt = prompt.invoke({"context":context, "query":query})
    response = llm.invoke(final_promt)
    print(f"AI :  {response.content}")
