from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]

loaded_document=PyPDFLoader(r"E:\Git-Repos\RAG\RAG_PDF_Reader\Rogers-2025-03-19.pdf").load()
text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
chunks_of_text=text_splitter.split_documents(loaded_document)
vector_db=Chroma.from_documents(chunks_of_text,OpenAIEmbeddings())

retriever=vector_db.as_retriever(search_kwargs={"k": 3})
#question="What is the total due Mr Chittaranjan has to pay?"
#response=vector_db.similarity_search(question)
#print(response[0].page_content)

prompt_template="""Answer the question based on the following context {context}
question:{question}
"""
prompt=ChatPromptTemplate.from_template(prompt_template)
model=ChatOpenAI()

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain=(
    {"context":retriever|format_docs,"question":RunnablePassthrough()}
    |prompt
    |model
    |StrOutputParser()
)
res=chain.invoke("What is the total due Mr Chittaranjan has to pay?")
print(res)