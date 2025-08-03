"""
Test script for embedding generation functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.embeddings import EmbeddingGenerator, BatchEmbeddingProcessor
from src.chunkers import ISCodeChunker
from src.utils.chunk_utils import load_chunks_from_json
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_basic_embedding():
    """Test basic embedding generation"""
    print("Testing Basic Embedding Generation")
    print("=" * 60)
    
    # Initialize generator (without API key for demo)
    print("\nNote: This test requires OPENAI_API_KEY in .env file")
    
    # Sample texts
    test_texts = [
        "IS 456:2000 deals with plain and reinforced concrete",
        "The characteristic compressive strength of concrete is denoted by fck",
        "Minimum cement content for moderate exposure is 300 kg/m³",
        "The maximum water-cement ratio for severe exposure is 0.45"
    ]
    
    try:
        # Initialize generator
        generator = EmbeddingGenerator(
            model="text-embedding-3-small",
            dimensions=1536
        )
        
        # Generate embeddings
        for text in test_texts:
            tokens = generator.count_tokens(text)
            print(f"\nText: {text}")
            print(f"Tokens: {tokens}")
            
            # Generate embedding (would need API key)
            # result = generator.generate_embedding(text)
            # print(f"Embedding dimensions: {result.dimensions}")
            
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure OPENAI_API_KEY is set in .env file")


def test_embedding_similarity():
    """Test embedding similarity computation"""
    print("\n" + "=" * 60)
    print("Testing Embedding Similarity")
    print("=" * 60)
    
    # Create mock embeddings for testing
    from src.embeddings import EmbeddingResult
    
    # Create normalized random vectors
    vec1 = np.random.randn(1536)
    vec1 = vec1 / np.linalg.norm(vec1)
    
    vec2 = np.random.randn(1536)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # Similar vector (small angle)
    vec3 = vec1 + 0.1 * np.random.randn(1536)
    vec3 = vec3 / np.linalg.norm(vec3)
    
    # Create embedding results
    emb1 = EmbeddingResult(
        text="Text 1",
        embedding=vec1.tolist(),
        model="text-embedding-3-small",
        dimensions=1536
    )
    
    emb2 = EmbeddingResult(
        text="Text 2",
        embedding=vec2.tolist(),
        model="text-embedding-3-small",
        dimensions=1536
    )
    
    emb3 = EmbeddingResult(
        text="Similar to Text 1",
        embedding=vec3.tolist(),
        model="text-embedding-3-small",
        dimensions=1536
    )
    
    # Test similarity computation
    generator = EmbeddingGenerator()
    
    sim_1_2 = generator.compute_similarity(emb1, emb2, metric="cosine")
    sim_1_3 = generator.compute_similarity(emb1, emb3, metric="cosine")
    
    print(f"Similarity between random vectors: {sim_1_2:.4f}")
    print(f"Similarity between similar vectors: {sim_1_3:.4f}")
    
    # Find similar embeddings
    candidates = [emb2, emb3]
    similar = generator.find_similar_embeddings(emb1, candidates, top_k=2)
    
    print("\nMost similar to Text 1:")
    for emb, score in similar:
        print(f"  {emb.text}: {score:.4f}")


def test_batch_processor():
    """Test batch processing with checkpointing"""
    print("\n" + "=" * 60)
    print("Testing Batch Processor")
    print("=" * 60)
    
    # Check if we have chunks from previous test
    chunk_file = "test_output/chunks.json"
    
    if os.path.exists(chunk_file):
        print(f"\nLoading chunks from {chunk_file}")
        chunks = load_chunks_from_json(chunk_file)
        print(f"Loaded {len(chunks)} chunks")
        
        # Show chunk preview
        print("\nChunk preview:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1} ({chunk.chunk_id}):")
            print(f"  Tokens: {chunk.token_count}")
            print(f"  Content: {chunk.content[:80]}...")
            
        # Demonstrate batch processor setup
        print("\n" + "-" * 40)
        print("Batch Processor Configuration:")
        print("-" * 40)
        
        print("To process chunks with embeddings:")
        print("1. Set OPENAI_API_KEY in .env file")
        print("2. Run the following code:")
        print()
        print("```python")
        print("from src.embeddings import EmbeddingGenerator, BatchEmbeddingProcessor")
        print()
        print("# Initialize")
        print("generator = EmbeddingGenerator()")
        print("processor = BatchEmbeddingProcessor(")
        print("    embedding_generator=generator,")
        print("    checkpoint_dir='embeddings_checkpoint'")
        print(")")
        print()
        print("# Process chunks")
        print("results = processor.process_chunks(")
        print("    chunks=chunks,")
        print("    batch_size=100,")
        print("    checkpoint_interval=500,")
        print("    progress_callback=lambda p, t: print(f'Progress: {p}/{t}')")
        print(")")
        print()
        print("# Save results")
        print("processor.save_embeddings_to_file(")
        print("    chunk_embedding_pairs=results,")
        print("    output_path='embeddings/is_456_embeddings.json'")
        print(")")
        print("```")
        
    else:
        print(f"\nNo chunks found at {chunk_file}")
        print("Run test_chunker.py first to generate chunks")


def demo_embedding_pipeline():
    """Demonstrate the complete embedding pipeline"""
    print("\n" + "=" * 60)
    print("Embedding Pipeline Demo")
    print("=" * 60)
    
    # Sample IS code content
    sample_content = """
## 5.2 Aggregates

### 5.2.1 General Requirements

All aggregates shall comply with IS 383:1970. The nominal maximum size of 
coarse aggregate should be as large as possible but should not exceed one-fourth 
of the minimum thickness of the member.

### 5.2.2 Size of Aggregates

The maximum size of aggregate should generally be restricted to:
- 20 mm for reinforced concrete work
- 40 mm for mass concrete work
- 10 mm for thin sections
"""
    
    print("1. Chunking document...")
    chunker = ISCodeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_by_clauses(sample_content, {"code": "IS 456", "version": 2000})
    print(f"   Created {len(chunks)} chunks")
    
    print("\n2. Embedding configuration:")
    print("   Model: text-embedding-3-small")
    print("   Dimensions: 1536")
    print("   Batch size: 100")
    
    print("\n3. Processing steps:")
    print("   - Generate embeddings for each chunk")
    print("   - Validate embedding dimensions")
    print("   - Save checkpoints for recovery")
    print("   - Store embeddings with metadata")
    
    print("\n4. Output formats:")
    print("   - JSON: Human-readable, includes metadata")
    print("   - Pickle: Binary format, faster loading")
    print("   - Direct to Pinecone: Vector database storage")


if __name__ == "__main__":
    # Run tests
    test_basic_embedding()
    test_embedding_similarity()
    test_batch_processor()
    demo_embedding_pipeline()
    
    print("\n✅ Embedding tests completed!")