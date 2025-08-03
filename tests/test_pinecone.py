"""
Test script for Pinecone integration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
from src.database import PineconeManager
from src.embeddings import EmbeddingGenerator
from src.chunkers import DocumentChunk, ISCodeChunker
import numpy as np
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test configuration
TEST_INDEX_NAME = "test-is-codes-index"
TEST_NAMESPACE = "test-namespace"
VECTOR_DIMENSION = 1536  # Must match embedding model

def cleanup_test_index(manager: PineconeManager):
    """Clean up test index"""
    print(f"\nCleaning up test index: {TEST_INDEX_NAME}")
    try:
        manager.delete_index(TEST_INDEX_NAME)
        time.sleep(10) # Wait for deletion to complete
    except Exception as e:
        print(f"Error during cleanup: {e}")

def test_pinecone_integration():
    """Test full Pinecone integration workflow"""
    print("\n" + "=" * 60)
    print("Testing Pinecone Integration")
    print("=" * 60)
    
    print("\nNote: This test requires PINECONE_API_KEY in .env file")
    
    try:
        # Initialize Pinecone Manager
        manager = PineconeManager()
        
        # 1. Cleanup previous test index if it exists
        cleanup_test_index(manager)
        
        # 2. Create index
        print("\nCreating index...")
        index = manager.create_index_if_not_exists(
            index_name=TEST_INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric='cosine'
        )
        assert index is not None
        print(f"Index '{TEST_INDEX_NAME}' created/retrieved")
        
        # Describe index
        desc = manager.describe_index(TEST_INDEX_NAME)
        print(f"\nIndex description: {desc}")
        
        # 3. Create mock data
        print("\nCreating mock data...")
        # Use a real embedding generator for dimension consistency
        # but mock the embeddings to avoid API calls
        generator = EmbeddingGenerator(dimensions=VECTOR_DIMENSION)
        
        # Mock chunks and embeddings
        chunks = []
        embeddings = []
        
        texts = [
            "Minimum grade of concrete for RCC is M20",
            "Maximum water-cement ratio for M20 is 0.55",
            "Standard deviation for M20 concrete is 4.0 N/mm²"
        ]
        
        for i, text in enumerate(texts):
            chunk = DocumentChunk(
                content=text,
                metadata={'clause': f'A.{i+1}', 'is_code': 'IS 456'},
                chunk_id=f'chunk_{i}',
                token_count=len(text.split()),
                start_char=0,
                end_char=len(text)
            )
            chunks.append(chunk)
            
            # Create a mock embedding result
            embedding_vector = (np.random.rand(VECTOR_DIMENSION) - 0.5).tolist()
            from src.embeddings import EmbeddingResult
            embedding = EmbeddingResult(
                text=text,
                embedding=embedding_vector,
                model="text-embedding-3-small",
                dimensions=VECTOR_DIMENSION
            )
            embeddings.append(embedding)
            
        chunk_embedding_pairs = list(zip(chunks, embeddings))
        
        # 4. Upsert embeddings
        print("\nUpserting embeddings...")
        upserted_count = manager.upsert_embeddings(
            index=index,
            chunk_embedding_pairs=chunk_embedding_pairs,
            namespace=TEST_NAMESPACE
        )
        assert upserted_count == len(texts)
        print(f"Upserted {upserted_count} vectors")
        
        # Wait for index to update
        time.sleep(10)
        
        # 5. Get index stats
        print("\nGetting index stats...")
        stats = manager.get_index_stats(index)
        print(stats)
        assert stats['namespaces'][TEST_NAMESPACE]['vector_count'] == len(texts)
        
        # 6. Query index
        print("\nQuerying index...")
        query_text = "What is the standard deviation for M20?"
        
        # Mock query embedding
        query_embedding_vector = (np.random.rand(VECTOR_DIMENSION) - 0.5).tolist()
        query_embedding = EmbeddingResult(
            text=query_text,
            embedding=query_embedding_vector,
            model="text-embedding-3-small",
            dimensions=VECTOR_DIMENSION
        )
        
        # Use the same embedding for query to get a perfect match
        query_embedding.embedding = embeddings[2].embedding
        
        results = manager.query(
            index=index,
            query_embedding=query_embedding,
            top_k=2,
            namespace=TEST_NAMESPACE,
            include_metadata=True
        )
        
        print("\nQuery results:")
        for res in results:
            print(f"  ID: {res['id']}, Score: {res['score']:.4f}, Metadata: {res['metadata']}")
            
        assert len(results) > 0
        assert results[0]['id'] == 'chunk_2'
        
        # 7. Delete vectors
        print("\nDeleting vectors...")
        ids_to_delete = ['chunk_0', 'chunk_1']
        manager.delete_vectors(index, ids_to_delete, namespace=TEST_NAMESPACE)
        print(f"Deleted vectors: {ids_to_delete}")
        
        # Wait for index to update
        time.sleep(5)
        
        stats_after_delete = manager.get_index_stats(index)
        print(f"\nStats after deletion: {stats_after_delete}")
        assert stats_after_delete['namespaces'][TEST_NAMESPACE]['vector_count'] == 1
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Make sure PINECONE_API_KEY and PINECONE_ENVIRONMENT are set in .env")
        
    finally:
        # Final cleanup
        if 'manager' in locals():
            cleanup_test_index(manager)
        
        print("\n✅ Pinecone integration test completed!")

if __name__ == "__main__":
    test_pinecone_integration()