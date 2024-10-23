import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Define constants
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
USER_ICON = "🧑‍💻"
BOT_ICON = "🤖"
ICON_SIZE = 20
USER_COLOR = "#e0f7fa"
BOT_COLOR = "#e3f2fd"

# Set up Streamlit page configuration
st.set_page_config(page_title="USCIS InfoBot", page_icon="🤖", layout="wide")

def load_document(file_path):
    """Load the document using UnstructuredPDFLoader."""
    loader = UnstructuredPDFLoader(file_path)
    return loader.load()

def setup_vectorstore(documents):
    """Create a FAISS vector store from the given document."""
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=800, chunk_overlap=150
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

class QueryModel(BaseModel):
    """Model for user queries."""
    question: str

def generate_rag_response(query, vectorstore, llm):
    """Generate a response using RAG approach with vectorstore retrieval."""
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(query)  # Updated method call
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Get only the response text from the LLM output
    response = llm.invoke(f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
    
    # Extract the content from the response if it's structured
    response_text = response.get("content") if isinstance(response, dict) else response

    return response_text  # Return only the relevant text


# Sidebar for PDF upload
st.sidebar.header("Upload USCIS Manual")
uploaded_file = st.sidebar.file_uploader(
    "Drop your PDF file here:", type=["pdf"]
)

if uploaded_file:
    file_path = f"{WORKING_DIR}/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    vectorstore = setup_vectorstore(load_document(file_path))

# Initialize ChatGroq instance here
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

st.title("🤖 USCIS Chatbot")
st.write("Ask me any question about the USCIS manual you uploaded.")

# User input field at the bottom of the page
user_input = st.text_input("Type your question:", placeholder="e.g., What is OPT for F-1 students?")

if user_input:
    # Generate a bot response using the RAG method
    try:
        query_model = QueryModel(question=user_input)  # Validate the input
        response = generate_rag_response(
            query_model.question, vectorstore, llm  # Pass llm directly here
        )
        st.markdown(f"<div style='background-color:{BOT_COLOR};"
                     f"padding:10px;border-radius:8px;margin:5px;'>"
                     f"<span style='font-size:{ICON_SIZE}px;margin-right:10px;'>{BOT_ICON}</span>"
                     f"{response}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")
