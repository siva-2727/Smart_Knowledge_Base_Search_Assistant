import app as st
import pandas as pd
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline

# Functions
def load_pdf(file):
    return extract_text(file)

def load_csv(file):
    df = pd.read_csv(file)
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
    return text_splitter.split_text(text)

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.from_texts(chunks, embeddings)

def get_relevant_chunks(vector_store, query):
    results = vector_store.similarity_search(query, k=5)
    return [r.page_content for r in results]

def generate_summary(chunks):
    context = " ".join(chunks)
    summarizer = pipeline("summarization", model="t5-base")
    llm = HuggingFacePipeline(pipeline=summarizer)
    prompt_template = """Summarize the following text:

    {text}

    Summary:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"text": context})

# Streamlit UI
st.set_page_config(page_title="PDF & CSV Analyzer", layout="wide")
st.title("📊 PDF & CSV Summarizer with Query")

pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])
csv_file = st.file_uploader("Upload CSV file", type=["csv"])
query = st.text_input("Enter your query", value="what is financial?")

if st.button("Analyze"):
    if not pdf_file or not csv_file:
        st.warning("Please upload both PDF and CSV files.")
    else:
        with st.spinner("Processing..."):
            pdf_text = load_pdf(pdf_file)
            csv_summary = load_csv(csv_file)

            combined_text = pdf_text + "\n" + csv_summary
            chunks = split_text(combined_text)
            vector_store = create_vector_store(chunks)
            relevant_chunks = get_relevant_chunks(vector_store, query)
            summary = generate_summary(relevant_chunks)

            st.success("Done!")
            st.subheader("Summary")
            st.write(summary)
