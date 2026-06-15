from dotenv import load_dotenv
load_dotenv()

import os
import requests

API_KEY = os.getenv("OPENWEATHER_API_KEY")
deg_c = "\u00B0C"

def get_weather(city: str):
    city_f = city.strip()
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_f},NP&units=metric&appid={API_KEY}"

    response = requests.get(url)
    data = response.json()
    if response.status_code != 200:
        return {
            "error": True,
            "message": data.get("message", "Unknown error")
        }
    
    weather = {
        "city": data['name'],
        "temperature": data['main']['temp'],
        "feels_like": data['main']['feels_like'],
        "cloud_desc": data['weather'][0]['description'],
        "humidity": data['main']['humidity'],
        "wind_speed": data['wind']['speed']
    }
    return weather

result = get_weather('pokhara')

if "error" in result:
    print("Error:", result["message"])
else:
    print(f"""
        For city named {result['city']}
        Temperature is {result['temperature']}{deg_c}
        It feels like {result['feels_like']}{deg_c}
        Sky seems {result['cloud_desc']}
        Humidity is {result['humidity']}%
        Wind speed is {result['wind_speed']}m/s
    """)