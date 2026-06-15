from langchain_community.document_loaders import PyPDFLoader
pdf_loader = PyPDFLoader("GenAI #2/MySQL Handbook.pdf")
pdf_documents = pdf_loader.load()

from langchain_text_splitters import CharacterTextSplitter
charSplitter = CharacterTextSplitter(
    separator = "",
    chunk_size = 300,
    chunk_overlap = 30
)
# chunks = charSplitter.split_documents(pdf_documents)

from langchain_text_splitters import TokenTextSplitter
tokenSplitter = TokenTextSplitter(
    chunk_size = 300,
    chunk_overlap = 30
)
# chunks = tokenSplitter.split_documents(pdf_documents)

from langchain_text_splitters import RecursiveCharacterTextSplitter
recurSplitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)
chunks = recurSplitter.split_documents(pdf_documents)

print(len(chunks))
print(chunks[4].page_content)