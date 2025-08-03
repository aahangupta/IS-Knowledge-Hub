"""
Main Streamlit application for IS Knowledge Hub
"""

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="IS Knowledge Hub",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Main application
def main():
    # Header
    st.title("🏗️ IS Knowledge Hub")
    st.markdown("*AI-Powered Construction Standards Engine*")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Home", "Upload IS Code", "Search", "Admin"],
            index=0
        )
    
    # Main content area
    if page == "Home":
        show_home_page()
    elif page == "Upload IS Code":
        show_upload_page()
    elif page == "Search":
        show_search_page()
    elif page == "Admin":
        show_admin_page()

def show_home_page():
    """Display the home page"""
    st.header("Welcome to IS Knowledge Hub")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 What is this?
        
        This platform transforms Indian Standards (IS) Codes for construction into a 
        structured, searchable knowledge base using AI.
        
        ### ✨ Key Features
        
        - **📘 IS Code Ingestion**: Parse PDFs into structured Markdown
        - **🧱 Intelligent Chunking**: Chunk by clause headers
        - **🔍 Semantic Search**: Vector-based search
        - **🤖 AI-Powered Q&A**: Get answers with clause citations
        """)
    
    with col2:
        st.markdown("""
        ### 📚 Supported Standards
        
        - IS 10262 - Concrete Mix Proportioning
        - IS 456 - Plain and Reinforced Concrete
        - IS 383 - Coarse and Fine Aggregates
        - IS 2386 - Methods of Test for Aggregates
        - And many more...
        
        ### 🚀 Getting Started
        
        1. **Upload** your IS Code PDFs
        2. **Search** for specific clauses or requirements
        3. **Ask questions** and get AI-powered answers
        """)

def show_upload_page():
    """Display the upload page"""
    st.header("📤 Upload IS Code")
    
    st.markdown("""
    Upload IS Code PDFs to parse and add them to the knowledge base.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an IS Code PDF file",
        type=['pdf'],
        help="Select a PDF file containing an IS Code document"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"📄 File: {uploaded_file.name}")
            st.info(f"📊 Size: {uploaded_file.size / 1024:.2f} KB")
        
        with col2:
            if st.button("🚀 Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    st.success("✅ PDF processing functionality will be implemented soon!")

def show_search_page():
    """Display the search page"""
    st.header("🔍 Search IS Codes")
    
    # Search input
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., What is the minimum cement content for M25 grade concrete?",
        help="Ask questions about IS codes or search for specific clauses"
    )
    
    # Search button
    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                st.info("🔍 Search functionality will be implemented soon!")
        else:
            st.warning("Please enter a search query")
    
    # Example queries
    st.markdown("### 💡 Example Queries")
    example_queries = [
        "What are the durability requirements in IS 456?",
        "How to calculate water-cement ratio for M30 concrete?",
        "What are the exposure conditions for concrete as per IS 456?",
        "Minimum grade of concrete for RCC work"
    ]
    
    for example in example_queries:
        if st.button(f"📌 {example}", key=example):
            st.text_input("Enter your search query", value=example, key="search_input_example")

def show_admin_page():
    """Display the admin page"""
    st.header("⚙️ Admin Panel")
    
    # Check for environment variables
    st.subheader("🔧 Configuration Status")
    
    env_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_KEY": os.getenv("SUPABASE_KEY")
    }
    
    for var_name, var_value in env_vars.items():
        if var_value:
            st.success(f"✅ {var_name} is configured")
        else:
            st.error(f"❌ {var_name} is not configured")
    
    # Database status
    st.subheader("🗄️ Database Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total IS Codes", "0")
        st.metric("Total Clauses", "0")
    
    with col2:
        st.metric("Total Embeddings", "0")
        st.metric("Search Queries", "0")

if __name__ == "__main__":
    main()