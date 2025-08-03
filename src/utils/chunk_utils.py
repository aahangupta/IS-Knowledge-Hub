"""
Utility functions for chunk processing
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from ..chunkers import DocumentChunk

logger = logging.getLogger(__name__)


def save_chunks_to_json(chunks: List[DocumentChunk], output_path: str) -> None:
    """
    Save chunks to JSON file
    
    Args:
        chunks: List of document chunks
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert chunks to dictionaries
    chunk_dicts = [chunk.to_dict() for chunk in chunks]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    

def load_chunks_from_json(json_path: str) -> List[DocumentChunk]:
    """
    Load chunks from JSON file
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        List of document chunks
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        chunk_dicts = json.load(f)
        
    chunks = []
    for chunk_dict in chunk_dicts:
        chunk = DocumentChunk(**chunk_dict)
        chunks.append(chunk)
        
    logger.info(f"Loaded {len(chunks)} chunks from {json_path}")
    return chunks
    

def save_chunks_to_csv(chunks: List[DocumentChunk], output_path: str) -> None:
    """
    Save chunks to CSV file for analysis
    
    Args:
        chunks: List of document chunks
        output_path: Path to save CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for DataFrame
    data = []
    for chunk in chunks:
        row = {
            'chunk_id': chunk.chunk_id,
            'token_count': chunk.token_count,
            'char_count': len(chunk.content),
            'start_char': chunk.start_char,
            'end_char': chunk.end_char,
            'content_preview': chunk.content[:100] + '...' if len(chunk.content) > 100 else chunk.content
        }
        
        # Add metadata fields
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                row[f'meta_{key}'] = value
            else:
                row[f'meta_{key}'] = str(value)
                
        data.append(row)
        
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"Saved chunk analysis to {output_path}")
    

def analyze_chunks(chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """
    Analyze chunk statistics
    
    Args:
        chunks: List of document chunks
        
    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {
            'total_chunks': 0,
            'avg_tokens': 0,
            'min_tokens': 0,
            'max_tokens': 0,
            'total_tokens': 0,
            'metadata_keys': []
        }
        
    token_counts = [chunk.token_count for chunk in chunks]
    
    # Collect all metadata keys
    metadata_keys = set()
    for chunk in chunks:
        metadata_keys.update(chunk.metadata.keys())
        
    stats = {
        'total_chunks': len(chunks),
        'avg_tokens': sum(token_counts) / len(token_counts),
        'min_tokens': min(token_counts),
        'max_tokens': max(token_counts),
        'total_tokens': sum(token_counts),
        'metadata_keys': sorted(list(metadata_keys))
    }
    
    # Analyze by metadata categories
    if 'section' in metadata_keys:
        sections = {}
        for chunk in chunks:
            section = chunk.metadata.get('section', 'Unknown')
            if section not in sections:
                sections[section] = 0
            sections[section] += 1
        stats['chunks_by_section'] = sections
        
    if 'content_type' in metadata_keys:
        content_types = {}
        for chunk in chunks:
            ctype = chunk.metadata.get('content_type', 'text')
            if ctype not in content_types:
                content_types[ctype] = 0
            content_types[ctype] += 1
        stats['chunks_by_content_type'] = content_types
        
    return stats
    

def merge_overlapping_chunks(chunks: List[DocumentChunk], max_tokens: int = 1500) -> List[DocumentChunk]:
    """
    Merge small chunks that are adjacent and related
    
    Args:
        chunks: List of document chunks
        max_tokens: Maximum tokens for merged chunk
        
    Returns:
        List of merged chunks
    """
    if not chunks:
        return []
        
    merged = []
    current_chunk = chunks[0]
    
    for next_chunk in chunks[1:]:
        # Check if chunks can be merged
        can_merge = (
            # Same metadata (excluding chunk-specific fields)
            _compare_metadata(current_chunk.metadata, next_chunk.metadata) and
            # Combined size within limit
            current_chunk.token_count + next_chunk.token_count <= max_tokens and
            # Adjacent chunks
            current_chunk.end_char == next_chunk.start_char
        )
        
        if can_merge:
            # Merge chunks
            current_chunk = DocumentChunk(
                content=current_chunk.content + next_chunk.content,
                metadata=current_chunk.metadata.copy(),
                chunk_id=f"{current_chunk.chunk_id}_merged",
                token_count=current_chunk.token_count + next_chunk.token_count,
                start_char=current_chunk.start_char,
                end_char=next_chunk.end_char
            )
        else:
            # Save current and start new
            merged.append(current_chunk)
            current_chunk = next_chunk
            
    # Add last chunk
    merged.append(current_chunk)
    
    logger.info(f"Merged {len(chunks)} chunks into {len(merged)} chunks")
    return merged
    

def _compare_metadata(meta1: Dict[str, Any], meta2: Dict[str, Any]) -> bool:
    """
    Compare metadata dictionaries, ignoring chunk-specific fields
    """
    ignore_fields = {'chunk_id', 'sub_chunk', 'partial_chunk', 'start_char', 'end_char'}
    
    # Get keys to compare
    keys1 = set(meta1.keys()) - ignore_fields
    keys2 = set(meta2.keys()) - ignore_fields
    
    if keys1 != keys2:
        return False
        
    # Compare values
    for key in keys1:
        if meta1[key] != meta2[key]:
            return False
            
    return True
    

def filter_chunks_by_metadata(
    chunks: List[DocumentChunk],
    filters: Dict[str, Any]
) -> List[DocumentChunk]:
    """
    Filter chunks based on metadata criteria
    
    Args:
        chunks: List of document chunks
        filters: Dictionary of metadata key-value pairs to filter by
        
    Returns:
        Filtered list of chunks
    """
    filtered = []
    
    for chunk in chunks:
        match = True
        for key, value in filters.items():
            chunk_value = chunk.metadata.get(key)
            
            # Handle list values (e.g., check if value in list)
            if isinstance(value, list):
                if chunk_value not in value:
                    match = False
                    break
            # Handle string pattern matching
            elif isinstance(value, str) and value.startswith('*') and value.endswith('*'):
                pattern = value[1:-1]
                if not (chunk_value and pattern in str(chunk_value)):
                    match = False
                    break
            # Exact match
            else:
                if chunk_value != value:
                    match = False
                    break
                    
        if match:
            filtered.append(chunk)
            
    logger.info(f"Filtered {len(chunks)} chunks to {len(filtered)} based on criteria")
    return filtered