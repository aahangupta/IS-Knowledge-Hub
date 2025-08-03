"""
Test script for semantic search engine
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
from unittest.mock import MagicMock
from src.search import SearchEngine, SearchResult, ResultFormatter
from src.embeddings import EmbeddingGenerator, EmbeddingResult
from src.database import PineconeManager
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_search_engine():
    """Test the search engine functionality"""
    print("\n" + "=" * 60)
    print("Testing Search Engine")
    print("=" * 60)
    
    # 1. Mock dependencies
    mock_generator = MagicMock(spec=EmbeddingGenerator)
    mock_pinecone = MagicMock(spec=PineconeManager)
    mock_index = MagicMock()
    
    # 2. Configure mocks
    # Mock embedding generation
    query_embedding_vector = np.random.rand(1536).tolist()
    mock_generator.generate_embedding.return_value = EmbeddingResult(
        text="test query",
        embedding=query_embedding_vector,
        model="text-embedding-3-small",
        dimensions=1536
    )
    
    # Mock Pinecone query
    mock_pinecone.query.return_value = [
        {
            'id': 'chunk_1', 
            'score': 0.95, 
            'metadata': {
                'content': 'Concrete shall be M20 grade for RCC work.',
                'is_code': 'IS 456',
                'clause': '5.1'
            }
        },
        {
            'id': 'chunk_2', 
            'score': 0.88, 
            'metadata': {
                'content': 'The minimum cement content is 300 kg/m³.',
                'is_code': 'IS 456',
                'clause': '8.2.4.1'
            }
        }
    ]
    
    mock_pinecone.get_index.return_value = mock_index
    
    # 3. Initialize Search Engine
    search_engine = SearchEngine(
        embedding_generator=mock_generator,
        pinecone_manager=mock_pinecone,
        index_name="test-index"
    )
    
    # 4. Perform search
    print("\nPerforming mock search...")
    query = "What is the concrete grade for RCC?"
    results = search_engine.search(query, top_k=2)
    
    # 5. Assertions
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].chunk_id == 'chunk_1'
    assert results[0].score > 0.9
    
    print("Search successful, found 2 results.")
    
    # 6. Test result formatting
    print("\nTesting result formatting...")
    
    # String format
    formatted_string = ResultFormatter.format_to_string(results)
    print("\nString Format:")
    print(formatted_string)
    assert "Result 1" in formatted_string
    assert "Content:" in formatted_string
    
    # Markdown format
    formatted_md = ResultFormatter.format_to_markdown(results)
    print("\nMarkdown Format:")
    print(formatted_md)
    assert "### Search Results" in formatted_md
    assert "**Score:" in formatted_md
    
    # RAG format
    formatted_rag = ResultFormatter.format_for_rag(results, max_tokens=100)
    print("\nRAG Context Format:")
    print(formatted_rag)
    assert "[Source: IS 456, Clause: 5.1]" in formatted_rag
    assert "---" in formatted_rag
    
    print("\n✅ Search engine tests completed!")

if __name__ == "__main__":
    test_search_engine()