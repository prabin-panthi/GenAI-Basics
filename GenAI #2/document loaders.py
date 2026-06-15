from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

#text file loader

from langchain_community.document_loaders import TextLoader

txt_loader = TextLoader("GenAI #2/text_file.txt")
txt_documents = txt_loader.load()

# print(txt_documents[0].page_content)

#pdf loader

from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader("GenAI #2/MySQL Handbook.pdf")
pdf_documents = pdf_loader.load()

# print(pdf_documents[5].page_content)

#web page loader

from langchain_community.document_loaders import WebBaseLoader

url = "https://www.prabinpanthi.com.np/"
web_loader = WebBaseLoader(url)
web_documents = web_loader.load()

# print(web_documents[0].page_content)


model = init_chat_model("mistral-medium-3-5", model_provider="mistralai")

promt = ChatPromptTemplate.from_messages([(
        "system", 
        "You are an intelligent AI capable of summarizing any length content into just 6 lines summary"
    ),
    (
        "human", 
        "Summarize {text}"
    )
])

# final_promt = promt.format_messages(text = txt_documents[0].page_content)
# final_promt = promt.format_messages(text = pdf_documents)
final_promt = promt.format_messages(text = web_documents[0].page_content)

response = model.invoke(final_promt)

print(response.content)