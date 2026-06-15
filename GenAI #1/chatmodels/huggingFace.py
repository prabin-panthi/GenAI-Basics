from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1"
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("make a short 50 word on solar system")

print(f"{response.content} \n")



from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro"
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("make a short 50 word on solar system")

print(response.content)