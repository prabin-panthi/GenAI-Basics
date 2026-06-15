from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
model_mistral = init_chat_model("mistral-medium-3-5", model_provider="mistralai")

print(f"\n--------------Welcome--------------[Enter 0 at any time to end]")
choice = int(input(f"Chose mode for AI \n Press 1 => Funny : 2 => Sad : 3 => Angry  "))

if choice == 1:
    mode = "You are a very funny AI agent. You respond with humor and jokes."
elif choice == 2:
    mode = "You are an sad AI agent. You respond with sadness in each message and make user cry."
elif choice == 3:
    mode = "You are an angry AI agent. You respond aggressively and impatiently."

messages = [
    SystemMessage(content=mode)
]

while True:
    promt = input("You : ")
    if promt == "0":
        break

    messages.append(HumanMessage(content=promt))
    response = model_mistral.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print(f"Mistral AI : {response.content} \n")

print(messages)