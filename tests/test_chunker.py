"""
Test script for document chunking functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.chunkers import ISCodeChunker
from src.utils.chunk_utils import analyze_chunks, save_chunks_to_json, save_chunks_to_csv
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sample IS code markdown content
SAMPLE_MARKDOWN = """---
title: IS 456:2000 - Plain and Reinforced Concrete
code: IS 456
version: 2000
type: Indian Standard
subject: Concrete
---

# IS 456:2000 - PLAIN AND REINFORCED CONCRETE - CODE OF PRACTICE

## 1 SCOPE

This standard deals with the general structural use of plain and reinforced concrete.

## 2 REFERENCES

The Indian Standards listed below are necessary adjuncts to this standard:

| IS No. | Title |
|--------|-------|
| IS 269:1989 | Specification for ordinary Portland cement |
| IS 383:1970 | Specification for coarse and fine aggregates |
| IS 1489:1991 | Specification for Portland pozzolana cement |

## 3 TERMINOLOGY

### 3.1 Concrete

A mixture of cement, water, and aggregates, with or without admixtures.

### 3.2 Reinforced Concrete

Concrete with embedded steel reinforcement.

## 4 SYMBOLS

The following symbols are used in this standard:

$$f_{ck} = \\text{Characteristic compressive strength of concrete}$$

$$f_y = \\text{Characteristic strength of steel}$$

## 5 MATERIALS

### 5.1 Cement

#### 5.1.1 General

Cement used shall conform to one of the following Indian Standards:

- IS 269:1989 Ordinary Portland cement
- IS 8112:1989 43 grade ordinary Portland cement  
- IS 12269:1987 53 grade ordinary Portland cement

#### 5.1.2 Storage of Cement

Cement shall be stored in such a manner as to prevent deterioration due to moisture.

### 5.2 Aggregates

#### 5.2.1 General Requirements

All aggregates shall comply with IS 383:1970.

#### 5.2.2 Size of Aggregates

The nominal maximum size of coarse aggregate should be as large as possible.

| Type of Work | Maximum Size |
|--------------|--------------|
| Reinforced concrete | 20 mm |
| Mass concrete | 40 mm |
| Lightly reinforced slabs | 10 mm |

### 5.3 Water

#### 5.3.1 Quality of Mixing Water

Water used for mixing and curing shall be clean and free from injurious amounts of oils, acids, alkalis, salts, sugar, organic materials.

## 6 CONCRETE MIX PROPORTIONING

### 6.1 Mix Design

#### 6.1.1 General

The mix proportion shall be selected to achieve the following:

$$\\text{Target mean strength} = f_{ck} + 1.65 \\times s$$

where:
- $f_{ck}$ = characteristic compressive strength at 28 days
- $s$ = standard deviation

#### 6.1.2 Selection of Mix Proportions

The mix shall be designed to produce the grade of concrete having the required workability and characteristic strength.

## 7 DURABILITY OF CONCRETE

### 7.1 General

A durable concrete is one that performs satisfactorily in the working environment.

### 7.2 Requirements for Durability

| Exposure | Min Cement Content (kg/m³) | Max W/C Ratio | Min Grade |
|----------|---------------------------|---------------|-----------|
| Mild | 300 | 0.55 | M20 |
| Moderate | 300 | 0.50 | M25 |
| Severe | 320 | 0.45 | M30 |
| Very severe | 340 | 0.45 | M35 |
| Extreme | 360 | 0.40 | M40 |
"""

def test_is_code_chunker():
    """Test IS code chunking functionality"""
    print("Testing IS Code Chunker")
    print("=" * 60)
    
    # Initialize chunker
    chunker = ISCodeChunker(
        chunk_size=200,  # Small size for testing
        chunk_overlap=50,
        preserve_tables=True,
        preserve_equations=True
    )
    
    # Chunk the sample document
    chunks = chunker.chunk_markdown_document(SAMPLE_MARKDOWN, "IS_456_2000.md")
    
    # Analyze chunks
    stats = analyze_chunks(chunks)
    
    print(f"\nChunking Statistics:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Average tokens per chunk: {stats['avg_tokens']:.1f}")
    print(f"Min tokens: {stats['min_tokens']}")
    print(f"Max tokens: {stats['max_tokens']}")
    print(f"Total tokens: {stats['total_tokens']}")
    
    if 'chunks_by_section' in stats:
        print(f"\nChunks by section:")
        for section, count in stats['chunks_by_section'].items():
            print(f"  Section {section}: {count} chunks")
            
    if 'chunks_by_content_type' in stats:
        print(f"\nChunks by content type:")
        for ctype, count in stats['chunks_by_content_type'].items():
            print(f"  {ctype}: {count} chunks")
    
    print(f"\nMetadata keys: {', '.join(stats['metadata_keys'])}")
    
    # Show sample chunks
    print("\n" + "=" * 60)
    print("Sample Chunks:")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        print(f"\nChunk {i+1} (ID: {chunk.chunk_id}):")
        print(f"Tokens: {chunk.token_count}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content preview: {chunk.content[:150]}...")
        print("-" * 40)
        
    # Look for special chunks
    print("\n" + "=" * 60)
    print("Special Content Chunks:")
    print("=" * 60)
    
    for chunk in chunks:
        if chunk.metadata.get('content_type') == 'table':
            print(f"\nTable chunk (ID: {chunk.chunk_id}):")
            print(chunk.content)
            print("-" * 40)
            
        elif chunk.metadata.get('content_type') == 'equation':
            print(f"\nEquation chunk (ID: {chunk.chunk_id}):")
            print(chunk.content)
            print("-" * 40)
            
    # Save results
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    save_chunks_to_json(chunks, f"{output_dir}/chunks.json")
    save_chunks_to_csv(chunks, f"{output_dir}/chunks_analysis.csv")
    
    print(f"\nResults saved to {output_dir}/")
    
    return chunks


def test_token_counting():
    """Test token counting accuracy"""
    print("\n" + "=" * 60)
    print("Testing Token Counting")
    print("=" * 60)
    
    chunker = ISCodeChunker()
    
    test_texts = [
        "This is a simple test.",
        "IS 456:2000 deals with plain and reinforced concrete.",
        "The characteristic compressive strength $f_{ck}$ is measured in N/mm².",
        "| Column 1 | Column 2 |\n|----------|----------|\n| Value 1  | Value 2  |"
    ]
    
    for text in test_texts:
        tokens = chunker.count_tokens(text)
        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")
        

if __name__ == "__main__":
    # Run tests
    test_token_counting()
    chunks = test_is_code_chunker()
    
    print("\n✅ Chunking tests completed successfully!")