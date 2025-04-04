import os
import streamlit as st
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

def process_pdf(pdf_path):
    loaded_doc=PyPDFLoader(pdf_path).load()
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    text_split_doc=text_splitter.split_documents(loaded_doc)
    vector_db=Chroma.from_documents(text_split_doc,OpenAIEmbeddings())
    return vector_db

def answer_question(question, retriever, prompt, llm):
    question_ans_chain = create_stuff_documents_chain(llm, prompt)
    reg_chain = create_retrieval_chain(retriever, question_ans_chain)
    result = reg_chain.invoke({"input": question})
    return result["answer"]

st.title("RAG based question answer system")
uploaded_file=st.file_uploader("upload a PDF file",type="pdf")

if uploaded_file is not None:
    pdf_path=r"E:\Git-Repos\RAG\Basic Connection\{uploaded_file.name}"
    with open(pdf_path,"wb") as f:
        f.write(uploaded_file.getbuffer())
    
    vector_db=process_pdf(pdf_path)
    retriever=vector_db.as_retriever()

    sys_prompt="""
        you are an assistant for question-answer chat.
        use the following pieces of retrieved context to answer the question.
        If you don't know the answer, say that you don't know.
        Use three sentences maximum and keep the answer concise.
        \n\n
        {context}
        """
    system_prompt=ChatPromptTemplate.from_messages(
        [
                ("system",sys_prompt),
                ("human","{input}")
        ]
    )
    st.success("PDF processed and ready for questions!")

    question = st.text_input("Ask a question about the document")
    if question:
        answer=answer_question(question,retriever,system_prompt,llm)
        st.write(f"Answer: {answer}")
    
