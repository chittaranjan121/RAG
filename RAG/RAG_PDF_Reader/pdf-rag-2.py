import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]
llm=ChatOpenAI()

loaded_doc=PyPDFLoader(r"E:\Git-Repos\RAG\RAG_PDF_Reader\Rogers-2025-03-19.pdf").load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunk_text=text_splitter.split_documents(loaded_doc)
vector_db=Chroma.from_documents(chunk_text,OpenAIEmbeddings())

res=vector_db.as_retriever()

sys_prompt="""you are an assistant for question answer chat"
    "use the following pieces of retrieved context to answer"
    "the question.If you don't know the answer say that you don't know.Use
    three sentense maximum and keep answer concise"
    "\n\n"
    "{context}"
"""

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",sys_prompt),
        ("human","{input}")
    ]
)
question_ans_chain=create_stuff_documents_chain(llm,prompt)
reg_chain=create_retrieval_chain(res,question_ans_chain)
result=reg_chain.invoke({"input":"What is the name of the customer"})
print(result['answer'])