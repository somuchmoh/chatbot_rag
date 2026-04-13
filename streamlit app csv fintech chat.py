import streamlit as st
import os

# Import LangChain and related modules
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# get Groq api key by reading the first line of the text file groq.txt

with open('groq.txt') as file:
    api_key = str(file.readline())

os.environ['GROQ_API_KEY'] = api_key

# Load CSV document
loader = CSVLoader(r'fintechdf_categorized.csv', encoding="latin-1")

doc = loader.load()

# Create embeddings for all the documents
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store embeddings in vectordatabase
vectorstore = FAISS.from_documents(doc, embeddings)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
)

# Setting up the retrieval function using modern LCEL approach
template = """Answer the question based only on the following context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = vectorstore.as_retriever()

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

st.title("Question Answering with Groq Chat")

# User input for the query
question = st.text_input("Enter your question:")

if st.button("Submit") and question:
    with st.spinner("Processing..."):
        result = chain.invoke(question)
    st.write("**Result:**")
    st.write(result)
