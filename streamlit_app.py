from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
import streamlit as st
import os 
import pandas as pd
import csv
import re
import requests

API_URL_mistral = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
API_URL_ro = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

def query_mistral(payload,headers):
	response = requests.post(API_URL_mistral, headers=headers, json=payload)
	return response.json()
	
def file_processing(file_path):

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen += page.page_content
        
    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 30
    )


    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen,question_gen

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def get_questions(pdf_file_path):
    x,y,z = file_processing(pdf_file_path)

    questions = []
    for i in range(len(x)):   
        prompt_template = f"""
            You are an expert at creating questions based on coding materials and documentation.
            Your goal is to prepare a coder or programmer for their exam and coding tests.
            You do this by asking questions about the text below:

            ------------
            {x[i].page_content}
            ------------

            Create questions that will prepare the coders or programmers for their tests.
            Make sure not to lose any important information.

            QUESTIONS:
            
            """

        output = query_mistral({
            "inputs": f"{prompt_template} ",
        },headers)
        questions.append(output[0]["generated_text"].split('QUESTIONS:')[-1].replace('\n','').replace('-',''))
    questions = [re.sub(" +"," ",element) for element in questions if len(element) > 0]

    separated_questions = []

    for combined_question in questions:
        individual_questions = combined_question.split(' 2. ')
        separated_questions.extend(individual_questions)
    separated_questions = [element.replace("1. ","") for element in separated_questions if element.endswith('?')]
    return separated_questions,z



def query_ro(payload,headers):
	response = requests.post(API_URL_ro, headers=headers, json=payload)
	return response.json()
	
def get_answers(questions,context):
    answers = []
    for i in range(len(questions)):   
        output = query_ro({
        "inputs": {
            "question": f"{questions[i]}",
            "context": f"{context}"
        },
            },headers)
        answers.append(output["answer"].replace('\n','').replace('-',''))
    return answers


def get_csv (Qestions,answers):
   
    output_file = "QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])  # Writing the header row

        for Q,A in zip(Qestions,answers):
            # Save answer to CSV file
            csv_writer.writerow([Q, A])
    return output_file


st.title("Question Answering System")
st.write("you can get the huggingface token from here \n https://huggingface.co/settings/tokens")
with st.form(key='my-form'):
    token = st.text_input("token key")
    
    if token:
        
        headers = {"Authorization": f"Bearer {token}"}
        
        uploaded_file = st.file_uploader("Upload a pdf file max 5 pages", type=["pdf"])
        # store the pdf
        if uploaded_file is not None:
            pdf_path = os.path.join("context.pdf")
            with open(pdf_path,"wb") as f:
                f.write(uploaded_file.getbuffer())

            
            
    gen = st.form_submit_button(label='Generate QA')
if gen and token and uploaded_file:
        questions,context = get_questions("context.pdf")

        answers = get_answers(questions,context)
        output_file_path = get_csv(questions,answers)
        csv = convert_df(pd.read_csv(output_file_path))
        st.download_button(
        label="Download Q-A pairs as CSV",
        data=csv,
        file_name='QA.csv',
        )
        
