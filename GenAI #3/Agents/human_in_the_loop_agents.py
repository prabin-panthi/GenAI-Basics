from dotenv import load_dotenv
load_dotenv()

import os
import requests
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from tavily import TavilyClient
from rich import print

API_KEY = os.getenv("OPENWEATHER_API_KEY")

@tool
def get_weather(city: str) -> dict:
    """
    A tool which provides the weather information about the city provided.
    Weather information includes: temperature, feels_like temperature, cloud description, humidity and wind speed.
    """

    city_f = city.strip()
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_f}&units=metric&appid={API_KEY}"

    response = requests.get(url)
    data = response.json()
    if response.status_code != 200:
        return f"Weather lookup failed: {data.get('message', 'Unknown error')}"
    
    weather = {
        "city": data['name'],
        "temperature": data['main']['temp'],
        "feels_like": data['main']['feels_like'],
        "cloud_description": data['weather'][0]['description'],
        "humidity": data['main']['humidity'],
        "wind_speed": data['wind']['speed']
    }
    
    return weather

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str):
    """a tool used for providing news"""
    city = city.strip()
    response = tavily.search(
        query=f"Provide me the latest news about the city: {city} which mainly should include the current affair of city: {city}",
        max_results=5
    )
    
    results = []

    for item in response['results']:
        results.append(f"Title: {item['title']}\n\n")
        results.append(f"Content: {item['content']}\n\n")
        results.append(f"Url: {item['url']}\n\n\n\n")

    return "".join(results)

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm = ChatMistralAI(model_name="mistral-medium-3")
llm_with_tools = llm.bind_tools([get_weather, get_news])

messages = []

print("---------Welcome to AI agents{Weather and News} chatbot---------(Write 'Exit' to exit the chat)")
while True:
    prompt = input("You : ")

    prompt = prompt.strip().lower()
    if prompt == "exit":
        break

    messages.append(HumanMessage(content=prompt))
    llm_response = llm_with_tools.invoke(messages)
    messages.append(llm_response)

    if llm_response.tool_calls:
        for tool_call in llm_response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call["args"]

            # Human In The Loop
            user_consent = input(f"LLM is asking to call {tool_name}. Let it call {tool_name}? (y/n)")
            user_consent = user_consent.strip().lower()
            if user_consent == "n":
                continue

            tool_response = tools[tool_name].invoke(tool_args)
            messages.append((ToolMessage(content=str(tool_response), tool_call_id=tool_call["id"])))

        llm_tool_response = llm_with_tools.invoke(messages)
        print(llm_tool_response.content)

    else:
        print(llm_response.content)