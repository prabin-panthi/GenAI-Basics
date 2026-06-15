from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

# mistrl model 

model_mistral = init_chat_model("mistral-medium-3-5", model_provider="mistralai")
response = model_mistral.invoke("who is pm of nepal")
print(f"From Mistral AI: \n {response.content} \n")

# groq model 

model_groq = init_chat_model("openai/gpt-oss-120b", model_provider="groq")
response = model_groq.invoke("who is pm of nepal")
print(f"From Groq AI: \n {response.content} \n")

# gemini model 

model_gemini = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
response = model_gemini.invoke("who is pm of nepal")
print(f"From Gemini AI: \n {response.content} \n")