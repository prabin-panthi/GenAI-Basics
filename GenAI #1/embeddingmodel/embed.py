from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpointEmbeddings

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2"
)

query_result = embeddings.embed_query("This is a test document.")
print(query_result)