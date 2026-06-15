from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

promt = ChatPromptTemplate([
    ("system", "You are a summarizer who can provide short 2-3 lines of summarized answer for asked question" ),
    ("human", "Question is: {question}")
]
)
model = init_chat_model("mistral-medium-3", model_provider="mistralai")
parser = StrOutputParser()

chain = promt | model | parser

result = chain.invoke("What is self attention layer in transformers with its uses?")
print(result)