"""
Markdown Document Chunker for IS Codes
Splits parsed markdown documents by clause headers while maintaining metadata
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import tiktoken
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    MarkdownHeaderTextSplitter
)
import yaml
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    Represents a chunk of an IS code document
    """
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    token_count: int
    start_char: int
    end_char: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return asdict(self)


class MarkdownChunker:
    """
    Chunks markdown documents with IS code-specific handling
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "text-embedding-3-small"
    ):
        """
        Initialize markdown chunker
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Token overlap between chunks
            model_name: Model name for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        # Initialize tiktoken encoder
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except:
            # Fallback to cl100k_base encoding if model not found
            self.encoder = tiktoken.get_encoding("cl100k_base")
            
        # Headers to split on for IS codes
        self.headers_to_split = [
            ("##", "Section"),
            ("###", "Subsection"),
            ("####", "Clause"),
        ]
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))
        
    def extract_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """
        Extract YAML frontmatter from markdown content
        
        Returns:
            Tuple of (metadata dict, content without frontmatter)
        """
        if content.startswith("---"):
            try:
                # Find the closing ---
                end_match = re.search(r'\n---\n', content[3:])
                if end_match:
                    yaml_content = content[3:end_match.start() + 3]
                    metadata = yaml.safe_load(yaml_content)
                    remaining_content = content[end_match.end() + 3:]
                    return metadata or {}, remaining_content
            except Exception as e:
                logger.warning(f"Failed to parse frontmatter: {e}")
                
        return {}, content
        
    def chunk_by_clauses(self, content: str, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Chunk markdown content by clause headers
        
        Args:
            content: Markdown content (without frontmatter)
            base_metadata: Base metadata from frontmatter
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        # First, use MarkdownHeaderTextSplitter to split by headers
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split,
            strip_headers=False
        )
        
        # Split by headers
        header_splits = header_splitter.split_text(content)
        
        # Then use RecursiveCharacterTextSplitter with token counting
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=self.model_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True
        )
        
        # Process each header split
        for idx, split in enumerate(header_splits):
            # Extract header metadata
            split_metadata = {**base_metadata}
            
            # Get the header metadata from the split
            if hasattr(split, 'metadata') and split.metadata:
                split_metadata.update(split.metadata)
                
            # Get content
            split_content = split.page_content if hasattr(split, 'page_content') else str(split)
            
            # Extract clause information from content
            clause_info = self._extract_clause_info(split_content)
            if clause_info:
                split_metadata.update(clause_info)
                
            # Check if this split needs further chunking
            token_count = self.count_tokens(split_content)
            
            if token_count <= self.chunk_size:
                # Single chunk for this section
                chunk = DocumentChunk(
                    content=split_content,
                    metadata=split_metadata,
                    chunk_id=f"{base_metadata.get('code', 'IS')}_{idx}",
                    token_count=token_count,
                    start_char=0,  # Will be updated later
                    end_char=len(split_content)
                )
                chunks.append(chunk)
            else:
                # Further split this section
                sub_splits = text_splitter.split_text(split_content)
                
                for sub_idx, sub_content in enumerate(sub_splits):
                    sub_token_count = self.count_tokens(sub_content)
                    chunk = DocumentChunk(
                        content=sub_content,
                        metadata={**split_metadata, "sub_chunk": sub_idx},
                        chunk_id=f"{base_metadata.get('code', 'IS')}_{idx}_{sub_idx}",
                        token_count=sub_token_count,
                        start_char=0,  # Will be updated later
                        end_char=len(sub_content)
                    )
                    chunks.append(chunk)
                    
        return chunks
        
    def _extract_clause_info(self, content: str) -> Optional[Dict[str, str]]:
        """
        Extract clause information from content
        
        Returns:
            Dictionary with clause_id, clause_title, section
        """
        clause_patterns = [
            # Clause 5.1.2 Title
            r'^#{2,4}\s*Clause\s+(\d+(?:\.\d+)*)\s*[:\-]?\s*(.+?)$',
            # 5.1.2 Title
            r'^#{2,4}\s*(\d+(?:\.\d+)+)\s+(.+?)$',
            # Section patterns
            r'^##\s*(\d+)\s+(.+?)$',
        ]
        
        lines = content.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            for pattern in clause_patterns:
                match = re.match(pattern, line.strip(), re.IGNORECASE)
                if match:
                    clause_id = match.group(1)
                    clause_title = match.group(2).strip()
                    
                    # Determine section from clause_id
                    section_match = re.match(r'^(\d+)', clause_id)
                    section = section_match.group(1) if section_match else None
                    
                    return {
                        "clause_id": clause_id,
                        "clause_title": clause_title,
                        "section": section
                    }
                    
        return None
        
    def chunk_markdown_document(self, content: str, filename: str = None) -> List[DocumentChunk]:
        """
        Chunk a complete markdown document
        
        Args:
            content: Complete markdown content including frontmatter
            filename: Optional filename for metadata
            
        Returns:
            List of document chunks
        """
        # Extract frontmatter
        metadata, markdown_content = self.extract_frontmatter(content)
        
        # Add filename to metadata if provided
        if filename:
            metadata['source_file'] = filename
            
        # Chunk by clauses
        chunks = self.chunk_by_clauses(markdown_content, metadata)
        
        # Update character positions
        current_pos = len(content) - len(markdown_content)  # Account for frontmatter
        for chunk in chunks:
            chunk.start_char = current_pos
            current_pos += len(chunk.content)
            chunk.end_char = current_pos
            
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
        
    def chunk_multiple_documents(self, documents: List[Dict[str, str]]) -> List[DocumentChunk]:
        """
        Chunk multiple markdown documents
        
        Args:
            documents: List of dicts with 'content' and 'filename' keys
            
        Returns:
            Combined list of chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            content = doc.get('content', '')
            filename = doc.get('filename', '')
            
            if content:
                chunks = self.chunk_markdown_document(content, filename)
                all_chunks.extend(chunks)
                
        return all_chunks