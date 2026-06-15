from dotenv import load_dotenv
load_dotenv()

import os
import requests
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent

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

llm = ChatMistralAI(model_name="mistral-medium-3")
agent = create_agent(
    model=llm,
    tools=[get_news, get_weather],
    system_prompt="You are a helpful assistant. Be concise and accurate."
)

print("---------Welcome to AI agent ChatBot--------press 0 to exit")
while True:
    prompt = input("You : ")
    if prompt == "0":
        break

    response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    print(f"AI : {response['messages'][-1].content}")