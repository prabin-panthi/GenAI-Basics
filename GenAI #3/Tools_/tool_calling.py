from dotenv import load_dotenv
load_dotenv()

from rich import print
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

#1 Tool creation
@tool
def get_length(text):
    """Returns the count of characters in a given text"""
    return len(text)

tools = {
    "get_length": get_length
}

#2 Tool binding
llm = ChatMistralAI(model_name="mistral-medium-3")
llm_with_tools = llm.bind_tools([get_length])

#3 History tracking
message = []
prompt = HumanMessage("what is the length of given text:'Hello World'")
message.append(prompt)

#4 Tool calling/llm calling so llm suggests tools
result = llm_with_tools.invoke(message)
message.append(result)

#5 Execute tool(s)
if result.tool_calls:
    tool_name = result.tool_calls[0]['name']
    tool_response = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_response)

    #6 Send back to LLM
    final_response = llm_with_tools.invoke(message)
    print(final_response.content)