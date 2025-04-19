import pandas as pd
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline

def load_pdf(file_path):
    text = extract_text(file_path)
    return text

def load_csv(file_path):
    
    df = pd.read_csv(file_path)
    
    numeric_data = df.select_dtypes(include=['float64', 'int64'])
    
    summary_stats = numeric_data.describe().transpose() 
    
    summary_text = ""
    for column in summary_stats.index:
        stats = summary_stats.loc[column]
        summary_text += (f"The column '{column}' has an average of {stats['mean']:.2f}, "
                         f"a standard deviation of {stats['std']:.2f}, a minimum of {stats['min']}, "
                         f"and a maximum of {stats['max']}.\n")
    return summary_text

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    vector_store = FAISS.from_texts(chunks, embedding_model)
    return vector_store

def get_relevant_chunks(vector_store, query):
    results = vector_store.similarity_search(query, k=5)  
    return [result.page_content for result in results]

def generate_summary(relevant_chunks):
    context = " ".join(relevant_chunks)
    
    summarization_pipeline = pipeline("summarization", model="t5-base")

    llm = HuggingFacePipeline(pipeline=summarization_pipeline)
    
    prompt_template = """Summarize the following text:

    {text}

    Summary:"""
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    summarization_chain = LLMChain(llm=llm, prompt=prompt)
    summary = summarization_chain.run({"text": context})
    return summary

pdf_text = load_pdf("iehp101.pdf")
csv_text = load_csv("Financial Statements.csv")

all_text = pdf_text + "\n" + csv_text
chunks = split_text(all_text)

vector_store = create_vector_store(chunks)

query = "what is financial?"

relevant_chunks = get_relevant_chunks(vector_store, query)
summary = generate_summary(relevant_chunks)

print("Summary:", summary)
