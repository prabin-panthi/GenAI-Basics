import os
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ---------------- Tools ---------------- #

@tool
def get_weather(city: str) -> dict:
    """Fetches the current weather details for a given city in Nepal."""
    city_f = city.strip()
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_f}&units=metric&appid={API_KEY}"

    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        return f"Weather lookup failed: {data.get('message', 'Unknown error')}"

    return {
        "city": data["name"],
        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "cloud_description": data["weather"][0]["description"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"],
    }


tavily = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def get_news(city: str):
    """Fetches the latest news articles about a specific city."""
    city = city.strip()

    response = tavily.search(
        query=f"latest news about {city}",
        max_results=5
    )

    results = []
    for item in response["results"]:
        results.append(f"Title: {item['title']}\n\n")
        results.append(f"Content: {item['content']}\n\n")
        results.append(f"Url: {item['url']}\n\n\n")

    return "".join(results)


tools = {
    "get_weather": get_weather,
    "get_news": get_news
}


# ---------------- LLM ---------------- #

llm = ChatMistralAI(model_name="mistral-medium-3")
llm_with_tools = llm.bind_tools([get_weather, get_news])


# ---------------- Streamlit UI ---------------- #

st.title("AI Agent Chatbot (Weather + News)")

if "messages" not in st.session_state:
    st.session_state.messages = []


# display chat history (only show messages that have content to avoid printing blank tool calls)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").write(msg.content)


# user input
user_input = st.chat_input("Ask something...")

if user_input:
    # store and display user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # Agent execution loop
    while True:
        # LLM call
        llm_response = llm_with_tools.invoke(st.session_state.messages)
        st.session_state.messages.append(llm_response)

        # If the LLM wants to call tools, execute them and stay in the loop
        if llm_response.tool_calls:
            for tool_call in llm_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                tool_result = tools[tool_name].invoke(tool_args)

                st.session_state.messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    )
                )
            # Continue the loop so the LLM can see the tool results
            continue
        
        # If no tool calls were made, this is the final response
        if llm_response.content:
            st.chat_message("assistant").write(llm_response.content)
        break