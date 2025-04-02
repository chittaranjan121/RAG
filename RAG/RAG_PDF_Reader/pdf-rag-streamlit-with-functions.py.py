import os
import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]
llm=ChatOpenAI()

def pdf_file_process(pdfpath):
    loaded_doc=PyPDFLoader(pdfpath).load()
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunk_of_text=text_splitter.split_documents(loaded_doc)
    vector_db=Chroma.from_documents(chunk_of_text,OpenAIEmbeddings(),persist_directory="./chroma_db")
    return vector_db
def anser_question(question,retriever,prompt,llm):
    ques_ans_chain=create_stuff_documents_chain(llm,prompt)
    reg_chain=create_retrieval_chain(retriever,ques_ans_chain)
    result=reg_chain.invoke({"input":question})
    return result['answer']

st.title("RAG Based Question Answer prompt")
uploaded_file=st.file_uploader("upload a file",type="pdf")

if uploaded_file is not None:
    pdf_path=r"E:\Git-Repos\RAG\Basic Connection\{uploaded_file.name}"
    with open(pdf_path,"wb") as f:
        f.write(uploaded_file.getbuffer())
    
    vector_db=pdf_file_process(pdf_path)
    retriever=vector_db.as_retriever()

    sys_prompt="""
        you are an assistant for question-answer chat.
        use the following pieces of retrieved context to answer the question.
        If you don't know the answer, say that you don't know.
        Use three sentences maximum and keep the answer concise.
        \n\n
        {context}
        """
    system_prompt=ChatPromptTemplate.from_messages([
        ("system",sys_prompt),
        ("human","{input}")
    ])

    st.success("PDF uploaded successfully")
    question=st.text_input("Eneter a question of your choice")
    if question:
        answer=anser_question(question,retriever,system_prompt,llm)
        st.write(f"Answer:{answer}")

