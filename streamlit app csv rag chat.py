import streamlit as st

# Import LangChain and related modules
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Let users provide their own Groq API key via a masked input.
# If none is provided, fall back to groq.txt when available.
st.sidebar.header("Groq API Configuration")
user_api_key = st.sidebar.text_input(
    "Enter your Groq API key",
    type="password",
    help="Your key is masked and stored only in this browser session.",
)

if user_api_key:
    st.session_state["groq_api_key"] = user_api_key.strip()

api_key = st.session_state.get("groq_api_key")

if not api_key:
    try:
        with open("groq.txt", encoding="utf-8") as file:
            fallback_key = file.readline().strip()
            if fallback_key:
                api_key = fallback_key
    except FileNotFoundError:
        api_key = None

if not api_key:
    st.warning("Please enter your Groq API key in the sidebar to continue.")
    st.stop()

# Load CSV document
loader = CSVLoader(r'fake_startup_founders_europe.csv', encoding="latin-1")

doc = loader.load()

# Create embeddings for all the documents
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store embeddings in vectordatabase
vectorstore = FAISS.from_documents(doc, embeddings)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
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
