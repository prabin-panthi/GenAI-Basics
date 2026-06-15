from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "mistral-medium-3-5",
    model_provider="mistralai"
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict, deterministic data extraction engine. Your sole task is to extract features from the user's movie paragraph.\n\n"

        "### CRITICAL EXTRACTION RULES:\n"
        "1. Strictly extract information directly mentioned or strongly implied by the text.\n"
        "2. If a specific attribute (like Target Audience) cannot be derived from the text, write 'Not specified in text'.\n"
        "3. Output ONLY the raw text fields exactly matching the template below.\n"
        "4. DO NOT wrap the response in markdown code fences, JSON, or conversational filler like 'Sure, here is the extraction:'.\n\n"

        "### EXPECTED OUTPUT FORMAT:\n"
        "Title: [Extract the title]\n"
        "Actor: [Extract all actors as a comma-separated list]\n"
        "Genre: [Extract the genre]\n"
        "Release Date: [Extract the release year/date]\n"
        "Director: [Extract the director's name]\n"
        "IMDb Rating: [Extract the numeric rating]\n"
        "Core Theme: [Extract the main conflict or thematic elements mentioned]\n"
        "Summary: [Provide a concise 1-2 sentence overview of the plot based on the paragraph]"
    ),
    (
        "human",
        "### INPUT MOVIE PARAGRAPH:\n{movie_paragraph}"
    )
])

st.title("Movie Information Extractor")

paragraph = st.text_area(
    "Enter movie paragraph:",
    height=250
)

if st.button("Extract"):
    final_prompt = prompt.invoke(
        {"movie_paragraph": paragraph}
    )

    response = model.invoke(final_prompt)

    st.text_area(
        "Result",
        value=response.content,
        height=400
    )