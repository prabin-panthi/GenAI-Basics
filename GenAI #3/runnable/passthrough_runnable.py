from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

promt1 = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator who generates a code on topics provided" ),
    ("human", "{question}")
])
promt2 = ChatPromptTemplate.from_template("Explain the followning code:\n{code}")

model = init_chat_model("mistral-medium-3", model_provider="mistralai")
parser = StrOutputParser()

chain1 = promt1 | model | parser
chain2 = RunnableParallel(
    {
        "code": RunnablePassthrough(),
        "summary": promt2 | model | parser
    }
)
final_chain = chain1 | chain2

result = final_chain.invoke("Write a code on python about prime number listing up to 16 numbers listed")
print(result['code'])
print()
print()
print(result['summary'])