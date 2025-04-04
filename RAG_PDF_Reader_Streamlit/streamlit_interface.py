import streamlit as st
from data_processing import process_pdf,save_load_file,answer_question

st.title("RAG based application")
uploaded_file=st.file_uploader("upload a file",type="pdf")

if uploaded_file is not None:
    pdf_path=save_load_file(uploaded_file)
    vector_db=process_pdf(pdf_path)

    st.write(f"successfully uploaded {uploaded_file.name}")

    question=st.text_input("Please ask a question")
    if question:
        answer = answer_question(vector_db, question)
        st.write(answer)