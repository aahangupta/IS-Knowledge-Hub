# Document Chunking Module

This module provides intelligent chunking functionality for IS Code documents, designed to split parsed Markdown content into manageable chunks while preserving document structure and semantic meaning.

## Features

### MarkdownChunker
- **Token-based chunking** using tiktoken for accurate token counting
- **Header-aware splitting** that respects document hierarchy
- **Metadata preservation** from YAML frontmatter
- **Configurable chunk size and overlap**
- **Clause detection** with automatic metadata extraction

### ISCodeChunker (extends MarkdownChunker)
- **Table preservation** - keeps tables intact as single chunks
- **Equation preservation** - maintains LaTeX equations together
- **IS-specific metadata extraction**:
  - IS code number and year
  - Table references
  - Figure references
  - Equation references
- **Content type tagging** for specialized handling

## Usage

### Basic Chunking

```python
from src.chunkers import MarkdownChunker

# Initialize chunker
chunker = MarkdownChunker(
    chunk_size=1000,        # Target tokens per chunk
    chunk_overlap=200,      # Token overlap between chunks
    model_name="text-embedding-3-small"
)

# Chunk a markdown document
with open("IS_456_2000.md", "r") as f:
    content = f.read()
    
chunks = chunker.chunk_markdown_document(content, "IS_456_2000.md")

# Access chunk information
for chunk in chunks:
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Tokens: {chunk.token_count}")
    print(f"Metadata: {chunk.metadata}")
    print(f"Content: {chunk.content[:100]}...")
```

### IS Code Chunking

```python
from src.chunkers import ISCodeChunker

# Initialize IS code chunker
chunker = ISCodeChunker(
    chunk_size=1000,
    chunk_overlap=200,
    preserve_tables=True,    # Keep tables intact
    preserve_equations=True  # Keep equations intact
)

# Chunk IS code document
chunks = chunker.chunk_markdown_document(content, "IS_456_2000.md")

# Filter chunks by content type
table_chunks = [c for c in chunks if c.metadata.get('content_type') == 'table']
equation_chunks = [c for c in chunks if c.metadata.get('content_type') == 'equation']
```

## Chunk Structure

Each chunk is a `DocumentChunk` object with:

```python
@dataclass
class DocumentChunk:
    content: str              # The actual text content
    metadata: Dict[str, Any]  # Metadata including clause info, IS code details
    chunk_id: str            # Unique identifier
    token_count: int         # Number of tokens
    start_char: int          # Start position in original document
    end_char: int            # End position in original document
```

## Metadata Fields

Common metadata fields extracted:

- **From frontmatter**: `title`, `code`, `version`, `type`, `subject`
- **From content**: 
  - `clause_id` - e.g., "5.1.2"
  - `clause_title` - e.g., "Storage of Cement"
  - `section` - e.g., "5"
  - `is_number` - e.g., "456"
  - `year` - e.g., "2000"
- **Special chunks**:
  - `content_type` - "table", "equation", or "text"
  - `preserved` - True for protected content
  - `table_references` - List of table numbers referenced
  - `figure_references` - List of figure numbers referenced

## Configuration

### Chunk Size Guidelines

- **1000 tokens** - Standard for most use cases
- **500 tokens** - For detailed search and retrieval
- **1500 tokens** - For broader context understanding
- **2000 tokens** - Maximum for most embedding models

### Overlap Guidelines

- **10-20%** of chunk size - Standard overlap
- **200 tokens** - Good default for 1000 token chunks
- **0 tokens** - For non-overlapping chunks (not recommended)

## Utilities

The module includes utility functions in `src.utils.chunk_utils`:

```python
from src.utils.chunk_utils import (
    save_chunks_to_json,      # Save chunks to JSON
    load_chunks_from_json,    # Load chunks from JSON
    save_chunks_to_csv,       # Export chunk analysis
    analyze_chunks,           # Get chunk statistics
    merge_overlapping_chunks, # Merge small chunks
    filter_chunks_by_metadata # Filter chunks
)
```

## Best Practices

1. **Choose appropriate chunk size** based on your embedding model's context window
2. **Use overlap** to maintain context between chunks (10-20% of chunk size)
3. **Preserve tables and equations** for IS codes to maintain data integrity
4. **Review metadata extraction** to ensure all relevant information is captured
5. **Test different configurations** to find optimal settings for your use case

## Example Output

```json
{
  "content": "### 5.2.1 General Requirements\n\nAll aggregates shall comply with IS 383:1970.",
  "metadata": {
    "title": "IS 456:2000 - Plain and Reinforced Concrete",
    "code": "IS 456",
    "version": 2000,
    "clause_id": "5.2.1",
    "clause_title": "General Requirements",
    "section": "5",
    "is_number": "456",
    "year": "2000"
  },
  "chunk_id": "IS 456_7",
  "token_count": 89,
  "start_char": 1234,
  "end_char": 1456
}
```