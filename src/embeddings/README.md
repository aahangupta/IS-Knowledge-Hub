# Vector Embedding Generation Module

This module provides functionality for generating vector embeddings from text using OpenAI's embedding models, optimized for IS Code document processing.

## Features

### EmbeddingGenerator
- **OpenAI Integration**: Uses text-embedding-3-small model by default
- **Configurable Dimensions**: Support for different embedding sizes
- **Token Counting**: Accurate token counting using tiktoken
- **Batch Processing**: Efficient batch API calls
- **Similarity Search**: Built-in similarity computation

### BatchEmbeddingProcessor
- **Large-scale Processing**: Handle thousands of chunks efficiently
- **Checkpointing**: Resume from interruptions
- **Progress Tracking**: Real-time progress updates
- **Error Recovery**: Automatic retry and checkpoint on errors
- **Multiple Output Formats**: JSON and pickle formats

## Usage

### Basic Embedding Generation

```python
from src.embeddings import EmbeddingGenerator

# Initialize generator
generator = EmbeddingGenerator(
    model="text-embedding-3-small",
    dimensions=1536  # Optional, uses model default
)

# Generate single embedding
result = generator.generate_embedding(
    text="The characteristic compressive strength of concrete",
    chunk_id="chunk_1",
    metadata={"clause": "5.1", "section": "Materials"}
)

print(f"Embedding dimensions: {result.dimensions}")
print(f"First 10 values: {result.embedding[:10]}")
```

### Batch Processing with Chunks

```python
from src.embeddings import EmbeddingGenerator, BatchEmbeddingProcessor
from src.chunkers import ISCodeChunker

# Create chunks
chunker = ISCodeChunker()
chunks = chunker.chunk_markdown_document(content)

# Initialize processor
generator = EmbeddingGenerator()
processor = BatchEmbeddingProcessor(
    embedding_generator=generator,
    checkpoint_dir="./embeddings_checkpoint"
)

# Process chunks with progress tracking
def progress_callback(processed, total):
    print(f"Progress: {processed}/{total} ({processed/total*100:.1f}%)")

results = processor.process_chunks(
    chunks=chunks,
    batch_size=100,
    delay_ms=100,
    checkpoint_interval=500,
    progress_callback=progress_callback
)

# Save results
processor.save_embeddings_to_file(
    chunk_embedding_pairs=results,
    output_path="embeddings/is_456_embeddings.json",
    format="json"
)
```

### Similarity Search

```python
# Generate query embedding
query_text = "What is the minimum cement content for concrete?"
query_embedding = generator.generate_embedding(query_text)

# Load candidate embeddings
candidates = processor.load_embeddings_from_file(
    "embeddings/is_456_embeddings.json"
)

# Find similar chunks
candidate_embeddings = [emb for _, emb in candidates]
similar = generator.find_similar_embeddings(
    query_embedding=query_embedding,
    candidate_embeddings=candidate_embeddings,
    top_k=5,
    metric="cosine",
    threshold=0.7
)

# Display results
for embedding, score in similar:
    print(f"Score: {score:.3f}")
    print(f"Chunk: {embedding.chunk_id}")
    print(f"Text: {embedding.text[:100]}...")
    print()
```

## Configuration

### Model Selection

- **text-embedding-3-small** (Default)
  - Dimensions: 1536 (configurable down to 512)
  - Good balance of performance and cost
  - Recommended for most use cases

- **text-embedding-3-large**
  - Dimensions: 3072 (configurable down to 256)
  - Higher quality embeddings
  - Use for critical applications

- **text-embedding-ada-002**
  - Dimensions: 1536 (fixed)
  - Legacy model, still supported

### Batch Processing Parameters

- **batch_size**: Number of texts per API call (default: 100)
  - Larger batches are more efficient but may hit rate limits
  - Adjust based on your OpenAI tier

- **delay_ms**: Delay between batches (default: 100ms)
  - Helps avoid rate limiting
  - Increase for free tier usage

- **checkpoint_interval**: Save progress every N chunks (default: 500)
  - More frequent = better recovery but slower processing
  - Less frequent = faster but more work lost on interruption

## API Requirements

Requires OpenAI API key in environment:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or in `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## Output Formats

### JSON Format
```json
[
  {
    "chunk": {
      "content": "All aggregates shall comply with IS 383:1970",
      "metadata": {
        "clause_id": "5.2.1",
        "section": "5"
      },
      "chunk_id": "IS_456_7",
      "token_count": 89
    },
    "embedding": {
      "vector": [0.123, -0.456, ...],
      "model": "text-embedding-3-small",
      "dimensions": 1536
    }
  }
]
```

### Statistics

```python
# Compute embedding statistics
stats = processor.compute_embedding_statistics(embeddings)
print(f"Total embeddings: {stats['count']}")
print(f"Dimensions: {stats['dimensions']}")
print(f"Mean similarity: {stats['similarity_sample']['mean']:.3f}")
```

## Best Practices

1. **Chunk Size**: Keep chunks under 8191 tokens (model limit)
2. **Batch Size**: Use 100-200 for optimal throughput
3. **Checkpointing**: Enable for datasets > 1000 chunks
4. **Error Handling**: Always use try-except blocks for API calls
5. **Rate Limiting**: Monitor usage and adjust delays accordingly

## Performance Considerations

- **API Costs**: ~$0.00002 per 1K tokens for text-embedding-3-small
- **Processing Time**: ~1-2 seconds per 100 chunks (including delays)
- **Memory Usage**: ~6KB per embedding (1536 dimensions × 4 bytes)
- **Storage**: JSON ~2x larger than pickle format

## Troubleshooting

### Common Issues

1. **Rate Limiting**
   - Increase `delay_ms` parameter
   - Reduce `batch_size`
   - Upgrade OpenAI tier

2. **Memory Issues**
   - Process in smaller batches
   - Use pickle format for storage
   - Enable checkpointing

3. **Dimension Mismatch**
   - Ensure all chunks use same model
   - Validate before storing in vector DB
   - Check model configuration

4. **API Errors**
   - Verify API key is set
   - Check internet connection
   - Monitor OpenAI status page