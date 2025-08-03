"""
Test script for the RAG system
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
from unittest.mock import MagicMock
from src.rag import RAGService, PromptManager
from src.search import SearchEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_rag_service():
    """Test the RAG service functionality"""
    print("\n" + "=" * 60)
    print("Testing RAG Service")
    print("=" * 60)
    
    # 1. Mock dependencies
    mock_search_engine = MagicMock(spec=SearchEngine)
    mock_llm_client = MagicMock()
    
    # 2. Configure mocks
    # Mock search engine context retrieval
    mock_context = """
    [Source: IS 456, Clause: 5.1]
    The characteristic compressive strength of concrete is denoted by fck.
    
    ---
    
    [Source: IS 456, Clause: 6.1]
    Minimum grade of concrete for reinforced concrete shall be M20.
    """
    mock_search_engine.get_context_for_query.return_value = mock_context
    
    # Mock LLM response
    mock_llm_response = MagicMock()
    mock_llm_response.choices = [MagicMock()]
    mock_llm_response.choices[0].message.content = (
        "According to IS 456, Clause 6.1, the minimum grade of concrete for "
        "reinforced concrete shall be M20."
    )
    mock_llm_client.chat.completions.create.return_value = mock_llm_response
    
    # 3. Initialize services
    prompt_manager = PromptManager()
    rag_service = RAGService(
        search_engine=mock_search_engine,
        prompt_manager=prompt_manager,
        llm_model="gpt-4o"
    )
    # Inject the mock LLM client
    rag_service.client = mock_llm_client
    
    # 4. Ask a question
    print("\nAsking a question to the RAG service...")
    query = "What is the minimum grade of concrete for RCC?"
    result = rag_service.answer_question(query)
    
    # 5. Assertions
    assert "answer" in result
    assert "context" in result
    assert "M20" in result['answer']
    assert "IS 456" in result['answer']
    assert mock_search_engine.get_context_for_query.called
    assert mock_llm_client.chat.completions.create.called
    
    print("\nQuestion answered successfully:")
    print(f"  Query: {query}")
    print(f"  Answer: {result['answer']}")
    
    # Test with no context found
    print("\nTesting with no context found...")
    mock_search_engine.get_context_for_query.return_value = ""
    result_no_context = rag_service.answer_question("A question with no context")
    assert "could not find any relevant information" in result_no_context['answer']
    print("Handled no context scenario correctly.")
    
    print("\n✅ RAG service tests completed!")

if __name__ == "__main__":
    test_rag_service()