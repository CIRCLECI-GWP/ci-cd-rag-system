import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load environment variables from the .env file
load_dotenv()

# Retrieve API Key once at the top
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")

# Function to Load PDF and Extract Text
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# Function to Split Text into Chunks
def split_text(documents, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

# Function to Create FAISS Vector Store
def create_faiss_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# Function to Create and Run RAG Pipeline
def rag_pipeline(pdf_path, query):
    # Load and Split PDF
    documents = load_pdf(pdf_path)
    chunks = split_text(documents)

    # Create FAISS Vector Store
    vector_store = create_faiss_vector_store(chunks)

    # Define Retriever
    retriever = vector_store.as_retriever()

    # Define LLM (Google Gemini)
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)

    # Define Custom Prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # Create RAG Chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Get Response
    response = rag_chain.invoke({"query": query})
    return response["result"]

# Example Usage
if __name__ == "__main__":
    pdf_file = "https://services.google.com/fh/files/misc/evaluation_framework.pdf"
    question = "What is the main topic of the document?"
    response = rag_pipeline(pdf_file, question)
    print("RAG Response:", response) 