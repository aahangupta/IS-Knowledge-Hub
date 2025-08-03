# 🏗️ IS Knowledge Hub - AI-Powered Construction Standards Engine

An intelligent platform that transforms Indian Standards (IS) Codes for construction into a structured, searchable knowledge base using AI.

## 🎯 Overview

This system parses BIS documents (IS 10262, IS 456, IS 383, etc.) into clean structured Markdown, embeds them into a vector database, and enables chat-based RAG (retrieval-augmented generation) using GPT-4o and Gemini 2.5 Pro.

## ✨ Key Features

- **📘 IS Code Ingestion**: Parse PDFs into structured Markdown with YAML frontmatter
- **🧱 Intelligent Chunking**: Chunk by clause headers with metadata preservation
- **🔍 Semantic Search**: Vector-based search over IS code content
- **🤖 AI-Powered Q&A**: Get answers with exact clause citations
- **📊 Export Options**: Export full codes, sections, or specific clauses

## 🛠️ Tech Stack

- **Language**: Python 3.12+
- **Frontend**: Streamlit
- **Vector DB**: Pinecone
- **Metadata DB**: Supabase
- **LLMs**: GPT-4o, Gemini 2.5 Pro
- **Embeddings**: text-embedding-3-small

## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- Git
- API keys for OpenAI, Google AI, Pinecone, and Supabase

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd IS-Knowledge-Hub
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp env.template .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   streamlit run src/app.py
   ```

## 🚀 Usage

1. **Upload IS Code PDFs**: Use the Streamlit interface to upload BIS standard documents
2. **Automatic Processing**: The system will parse, chunk, and embed the content
3. **Ask Questions**: Query the knowledge base using natural language
4. **Get Cited Answers**: Receive AI-generated responses with exact clause references

## 📁 Project Structure

```
IS-Knowledge-Hub/
├── src/
│   ├── parsers/          # PDF parsing modules
│   ├── embeddings/       # Embedding generation
│   ├── database/         # Supabase & Pinecone interfaces
│   ├── api/              # FastAPI endpoints
│   └── utils/            # Utility functions
├── tests/                # Test suite
├── data/
│   ├── pdfs/            # Input PDF files
│   ├── markdown/        # Parsed Markdown files
│   └── processed/       # Processed data
├── config/              # Configuration files
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔧 Configuration

Key configuration options in `.env`:
- `EMBEDDING_MODEL`: Choose embedding model
- `CHUNK_SIZE`: Token size for chunking
- `PRIMARY_LLM`: Primary language model
- `FALLBACK_LLM`: Fallback model for redundancy

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is proprietary software owned by PillarX.

## 🏢 About PillarX

Construction materials manufacturer, aggregate supplier, and AI developer focused on building India's most measured, tech-forward concrete ecosystem.

---

For questions or support, please contact the development team.