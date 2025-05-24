import os # For file path operations
from dotenv import load_dotenv # Load environment variables from .env file
import streamlit as st # Streamlit library for web app
from langchain_community.document_loaders import UnstructuredPDFLoader # Document loader for PDF files
from langchain_text_splitters.character import CharacterTextSplitter # Text splitter for breaking documents into smaller chunks
from langchain_community.vectorstores import FAISS # Vector store for storing document embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings # Embeddings model for converting text to vectors
from langchain_groq import ChatGroq # Groq model for generating text
from langchain.memory import ConversationBufferMemory # Memory for storing conversation history
from langchain.chains import ConversationalRetrievalChain  # Chain for conversational retrieval
#from langchain.schema import Document
#import pdfplumber


# Load environment variables from .env file
load_dotenv()

# Get the current working directory
working_dir = os.path.dirname(os.path.abspath(__file__)) 


# read the pdf and extract text
def load_documents(file_path):
    loader = UnstructuredPDFLoader(file_path) # Load the PDF file
    documents = loader.load() # Load the documents
    return documents # Return the loaded documents


def setup_vector_store(documents):
    embeddings = HuggingFaceEmbeddings() # Load the embeddings model
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    ) # Split the text into smaller chunks
    doc_chunks = text_splitter.split_documents(documents) # Split the documents into chunks
    vector_store = FAISS.from_documents(doc_chunks, embeddings) # Create a vector store from the document chunks
    return vector_store # Return the vector store

def create_chain(vector_store):
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0,
    )
    retriver = vector_store.as_retriever() # Create a retriever from the vector store
    # Create a memory for storing conversation history
    memory = ConversationBufferMemory(
        llm=llm,  # Use the same LLM for memory
        output_key="answer",  # Key for the output
        memory_key="chat_history", # Key for the memory
        return_messages=True # Return messages in the memory
        ) 
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,  # Use the same LLM for the chain
        retriever=retriver,  # Use the retriever from the vector store
        memory=memory,  # Use the memory for conversation history
        verbose=True,  # Enable verbose mode for debugging
    )
    return chain # Return the chain

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon=":robot:",
    layout="centered",
)
st.title("PDF Chatbot") # Set the title of the app

# initialize the chat history in streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Initialize the chat history

# File uploader for PDF files
uploaded_file = st.file_uploader(label="Upload a PDF file", type=["pdf"]) # File uploader for PDF files

if uploaded_file is not None:# If a file is uploaded
    file_path = f"{working_dir}/{uploaded_file.name}" # Get the file path
    with open(file_path, "wb") as f: # Open the file in write-binary mode
        f.write(uploaded_file.getbuffer()) # Write the uploaded file to the path

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = setup_vector_store(load_documents(file_path)) # Setup the vector store
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vector_store) # Create the conversation chain

#display the chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
user_input = st.chat_input("Ask a question about the PDF") # Input box for user questions    

if user_input:# If the user input is not empty
    st.session_state.chat_history.append({"role": "user", "content": user_input}) # Append the user input to the chat history
    
    with st.chat_message("user"): # Display the user input
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input}) # Get the response from the conversation chain
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response}) # Append the assistant response to the chat history