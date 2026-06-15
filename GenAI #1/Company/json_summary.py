from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

model = init_chat_model("mistral-medium-3-5", model_provider="mistralai")

class Movie(BaseModel):
    title : str
    release_year : int
    genre : List[str]
    director : Optional[str]
    cast : List[str]
    rating : Optional[float]
    summary : str

parser = PydanticOutputParser(pydantic_object = Movie)

promt = ChatPromptTemplate.from_messages([
    ('system', "You are a movie information extractor, {format_instruction}"),
    ('human', "Extract the movie information from paragraph as: {movie_paragraph}")
])

paragraph = input("Enter movie paragraph :  ")

final_promt = promt.invoke({"movie_paragraph" : paragraph, "format_instruction" : parser.get_format_instructions()})

response = model.invoke(final_promt)

output = parser.parse(response.content)
print(output)