"""
Main Streamlit application for IS Knowledge Hub
"""

import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Add project root to sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database import SupabaseManager
from src.search import SearchEngine, ResultFormatter
from src.rag import RAGService, PromptManager
from src.embeddings import EmbeddingGenerator
from src.database import PineconeManager
from config import settings

# --- INITIALIZATION ---
def init_services():
    """Initialize all services and store them in session state"""
    if "services_initialized" not in st.session_state:
        st.session_state.supabase_manager = SupabaseManager()
        st.session_state.pinecone_manager = PineconeManager()
        st.session_state.embedding_generator = EmbeddingGenerator()
        st.session_state.search_engine = SearchEngine(
            embedding_generator=st.session_state.embedding_generator,
            pinecone_manager=st.session_state.pinecone_manager
        )
        st.session_state.prompt_manager = PromptManager()
        st.session_state.rag_service = RAGService(
            search_engine=st.session_state.search_engine,
            prompt_manager=st.session_state.prompt_manager
        )
        st.session_state.services_initialized = True
        print("Services initialized")

# --- UI COMPONENTS ---
def home_page():
    st.header("Welcome to the IS Knowledge Hub")
    st.markdown("""
    This platform transforms Indian Standards (IS) Codes for construction into a structured, searchable, and intelligent knowledge base.
    
    **Get started by:**
    1.  Uploading an IS code PDF in the **Upload IS Code** page.
    2.  Asking questions about the uploaded codes in the **Search** page.
    """)

def upload_page():
    st.header("Upload IS Code PDF")
    
    uploaded_file = st.file_uploader("Choose an IS code PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Save the file to a temporary location
        save_path = Path("data/pdfs") / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.info("The processing pipeline will be implemented in a future task.")

def search_page():
    st.header("Search IS Codes")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about IS codes..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Searching for answers..."):
            # Get response from RAG service
            rag_service = st.session_state.rag_service
            response = rag_service.answer_question(prompt)
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
                
                # Show context in an expander
                with st.expander("Show Context"):
                    st.text(response["context"])
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

def admin_page():
    st.header("Admin Dashboard")
    st.warning("Admin functionality is under construction.")
    
    # Display loaded IS codes
    st.subheader("Available IS Codes")
    supabase_manager = st.session_state.supabase_manager
    codes = supabase_manager.list_is_codes()
    
    if codes:
        for code in codes:
            st.write(f"- **{code.code_number}**: {code.title} (Status: {code.status})")
    else:
        st.info("No IS codes found in the database.")

# --- MAIN APPLICATION ---
def main():
    """Main function to run the Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="IS Knowledge Hub",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize services
    init_services()

    # Sidebar Navigation
    with st.sidebar:
        st.title("🏗️ IS Knowledge Hub")
        page = st.radio(
            "Navigation",
            ("Home", "Upload IS Code", "Search", "Admin")
        )

    # Page Content
    if page == "Home":
        home_page()
    elif page == "Upload IS Code":
        upload_page()
    elif page == "Search":
        search_page()
    elif page == "Admin":
        admin_page()

if __name__ == "__main__":
    main()