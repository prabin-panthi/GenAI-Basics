from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
model_mistral = init_chat_model("mistral-medium-3-5", model_provider="mistralai")

messages = []

print(f"\n--------------Welcome--------------[Enter 0 at any time to end]")
while True:
    promt = input("You : ")
    if promt == "0":
        break

    messages.append(promt)
    response = model_mistral.invoke(messages)
    messages.append(response.content)
    print(f"Mistral AI : {response.content} \n")

print(messages)