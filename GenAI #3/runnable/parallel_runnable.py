from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableParallel, RunnableLambda

short_promt = ChatPromptTemplate.from_template("Explain in 2 lines about {topic_short}")
long_promt = ChatPromptTemplate.from_template("Provide detailed explanation about 6-7 lines on {topic_long}")
model = init_chat_model("mistral-medium-3", model_provider="mistralai")
parser = StrOutputParser()

chain_parallel = RunnableParallel(
    {"short": short_promt | model | parser, "long": long_promt | model | parser}
)
lambda_chain = RunnableLambda(
    lambda x: f"SHORT :\n {x['short']} \n\n LONG:\n {x['long']}"
)
chain_final = chain_parallel | lambda_chain

result = chain_final.invoke({"topic_short": "Machine Learning", "topic_long": "Deep Learning"})
print(result)


#Uncomment(and Comment out above) and use the below code {use case if you want same variable name '{topic}'}

# from dotenv import load_dotenv
# load_dotenv()

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.chat_models import init_chat_model
# from langchain_core.runnables import RunnableParallel, RunnableLambda

# short_promt = ChatPromptTemplate.from_template("Explain in 2 lines about {topic}")
# long_promt = ChatPromptTemplate.from_template("Provide detailed explanation about 6-7 lines on {topic}")
# model = init_chat_model("mistral-medium-3", model_provider="mistralai")
# parser = StrOutputParser()

# chain_parallel = RunnableParallel(
#     {
#      "short": RunnableLambda(lambda y: y['short']) | short_promt | model | parser,
#      "long": RunnableLambda(lambda y: y['long']) | long_promt | model | parser
#     }
# )
# lambda_chain = RunnableLambda(
#     lambda x: f"SHORT :\n {x['short']} \n\n LONG:\n {x['long']}"
# )
# chain_final = chain_parallel | lambda_chain

# result = chain_final.invoke(
#     {
#      "short": {"topic": "Machine Learning"},
#      "long": {"topic": "Deep Learning"}
#     }
# )
# print(result)