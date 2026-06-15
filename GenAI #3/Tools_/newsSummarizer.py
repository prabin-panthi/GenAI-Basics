from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_results=5, include_images=False)

prompt = ChatPromptTemplate.from_template(
    "Since you are a news summarizer, generate few bullet points of news provided as :\n {news}"
)

llm = ChatMistralAI(model_name="mistral-medium-3")

chain = search_tool | prompt | llm | StrOutputParser()

result = chain.invoke("Latest news about fifa world cup 2026 about match info and result")
print(result)