import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

load_dotenv()
llm=ChatOpenAI()

def process_pdf(file_path):
    loaded_doc=PyPDFLoader(file_path).load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunk_of_text=text_splitter.split_documents(loaded_doc)
    vector_db=Chroma.from_documents(chunk_of_text,OpenAIEmbeddings())
    return vector_db

def save_load_file(uploaded_file,dest_folder=r'E:\Git-Repos\RAG\RAG_PDF_Reader_Streamlit'):
    pdf_path=os.path.join(dest_folder,uploaded_file.name)
    with open(pdf_path,"wb") as f:
        f.write(uploaded_file.getbuffer())
    return pdf_path

def answer_question(vector_db, question):
    # Use the vector database to retrieve relevant documents
    retriever = vector_db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(ChatOpenAI(), retriever=retriever)
    
    # Get the answer to the question
    answer = qa_chain.run(question)
    return answer

    
    