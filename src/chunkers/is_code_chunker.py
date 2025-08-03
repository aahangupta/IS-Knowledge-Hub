"""
Specialized chunker for IS Code documents
Handles tables, equations, and IS-specific structures
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from .markdown_chunker import MarkdownChunker, DocumentChunk

logger = logging.getLogger(__name__)


class ISCodeChunker(MarkdownChunker):
    """
    Specialized chunker for IS Code documents with enhanced handling
    for tables, equations, and IS-specific structures
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "text-embedding-3-small",
        preserve_tables: bool = True,
        preserve_equations: bool = True
    ):
        """
        Initialize IS Code chunker
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Token overlap between chunks
            model_name: Model name for token counting
            preserve_tables: Keep tables intact in chunks
            preserve_equations: Keep equations intact in chunks
        """
        super().__init__(chunk_size, chunk_overlap, model_name)
        self.preserve_tables = preserve_tables
        self.preserve_equations = preserve_equations
        
    def _is_table_line(self, line: str) -> bool:
        """Check if a line is part of a markdown table"""
        # Table separator line
        if re.match(r'^\s*\|?\s*[-:]+\s*\|', line):
            return True
        # Table content line
        if '|' in line and line.count('|') >= 2:
            return True
        return False
        
    def _is_equation_line(self, line: str) -> bool:
        """Check if a line contains LaTeX equations"""
        # Block equations
        if line.strip().startswith('$$') or line.strip().endswith('$$'):
            return True
        # Inline equations
        if re.search(r'\$[^$]+\$', line):
            return True
        return False
        
    def _extract_tables(self, content: str) -> List[Tuple[int, int, str]]:
        """
        Extract table positions from content
        
        Returns:
            List of (start_idx, end_idx, table_content) tuples
        """
        tables = []
        lines = content.split('\n')
        in_table = False
        table_start = 0
        table_lines = []
        
        for i, line in enumerate(lines):
            if self._is_table_line(line):
                if not in_table:
                    in_table = True
                    table_start = i
                    table_lines = [line]
                else:
                    table_lines.append(line)
            else:
                if in_table:
                    # Table ended
                    in_table = False
                    table_content = '\n'.join(table_lines)
                    tables.append((table_start, i-1, table_content))
                    table_lines = []
                    
        # Handle table at end of content
        if in_table and table_lines:
            table_content = '\n'.join(table_lines)
            tables.append((table_start, len(lines)-1, table_content))
            
        return tables
        
    def _extract_equations(self, content: str) -> List[Tuple[int, int, str]]:
        """
        Extract equation positions from content
        
        Returns:
            List of (start_idx, end_idx, equation_content) tuples
        """
        equations = []
        lines = content.split('\n')
        in_equation = False
        equation_start = 0
        equation_lines = []
        
        for i, line in enumerate(lines):
            # Check for block equation start/end
            if line.strip().startswith('$$'):
                if not in_equation:
                    in_equation = True
                    equation_start = i
                    equation_lines = [line]
                else:
                    # Equation ended
                    equation_lines.append(line)
                    equation_content = '\n'.join(equation_lines)
                    equations.append((equation_start, i, equation_content))
                    in_equation = False
                    equation_lines = []
            elif in_equation:
                equation_lines.append(line)
            elif self._is_equation_line(line) and not in_equation:
                # Single line equation
                equations.append((i, i, line))
                
        return equations
        
    def _get_is_code_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract IS code specific metadata from content
        """
        metadata = {}
        
        # Extract IS code number and year
        is_pattern = r'IS\s+(\d+)(?:\s*:\s*(\d{4}))?'
        match = re.search(is_pattern, content[:500])  # Check first 500 chars
        if match:
            metadata['is_number'] = match.group(1)
            if match.group(2):
                metadata['year'] = match.group(2)
                
        # Extract table references
        table_refs = re.findall(r'Table\s+(\d+(?:\.\d+)*)', content)
        if table_refs:
            metadata['table_references'] = list(set(table_refs))
            
        # Extract figure references
        figure_refs = re.findall(r'Fig(?:ure)?\s+(\d+(?:\.\d+)*)', content)
        if figure_refs:
            metadata['figure_references'] = list(set(figure_refs))
            
        # Extract equation references
        eq_refs = re.findall(r'Eq(?:uation)?\s*\((\d+(?:\.\d+)*)\)', content)
        if eq_refs:
            metadata['equation_references'] = list(set(eq_refs))
            
        return metadata
        
    def chunk_by_clauses(self, content: str, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Enhanced chunking that preserves tables and equations
        """
        # Add IS-specific metadata
        is_metadata = self._get_is_code_metadata(content)
        base_metadata.update(is_metadata)
        
        # Get base chunks
        chunks = super().chunk_by_clauses(content, base_metadata)
        
        if not self.preserve_tables and not self.preserve_equations:
            return chunks
            
        # Post-process chunks to preserve tables and equations
        processed_chunks = []
        
        for chunk in chunks:
            # Check if chunk contains tables or equations
            tables = self._extract_tables(chunk.content) if self.preserve_tables else []
            equations = self._extract_equations(chunk.content) if self.preserve_equations else []
            
            if not tables and not equations:
                processed_chunks.append(chunk)
                continue
                
            # Split chunk content preserving tables/equations
            protected_regions = []
            
            # Add tables to protected regions
            for start, end, table_content in tables:
                protected_regions.append({
                    'type': 'table',
                    'start': start,
                    'end': end,
                    'content': table_content
                })
                
            # Add equations to protected regions  
            for start, end, eq_content in equations:
                protected_regions.append({
                    'type': 'equation',
                    'start': start,
                    'end': end,
                    'content': eq_content
                })
                
            # Sort by start position
            protected_regions.sort(key=lambda x: x['start'])
            
            # Rebuild chunk respecting protected regions
            lines = chunk.content.split('\n')
            new_chunks = []
            current_chunk_lines = []
            current_tokens = 0
            
            i = 0
            while i < len(lines):
                # Check if we're at a protected region
                in_protected = False
                for region in protected_regions:
                    if i >= region['start'] and i <= region['end']:
                        # Add current chunk if it has content
                        if current_chunk_lines:
                            new_chunk_content = '\n'.join(current_chunk_lines)
                            new_chunk = DocumentChunk(
                                content=new_chunk_content,
                                metadata={
                                    **chunk.metadata,
                                    'partial_chunk': True
                                },
                                chunk_id=f"{chunk.chunk_id}_p{len(new_chunks)}",
                                token_count=self.count_tokens(new_chunk_content),
                                start_char=chunk.start_char,
                                end_char=chunk.start_char + len(new_chunk_content)
                            )
                            new_chunks.append(new_chunk)
                            current_chunk_lines = []
                            current_tokens = 0
                            
                        # Add protected region as separate chunk
                        protected_chunk = DocumentChunk(
                            content=region['content'],
                            metadata={
                                **chunk.metadata,
                                'content_type': region['type'],
                                'preserved': True
                            },
                            chunk_id=f"{chunk.chunk_id}_{region['type']}{len(new_chunks)}",
                            token_count=self.count_tokens(region['content']),
                            start_char=chunk.start_char,
                            end_char=chunk.start_char + len(region['content'])
                        )
                        new_chunks.append(protected_chunk)
                        
                        # Skip to end of protected region
                        i = region['end'] + 1
                        in_protected = True
                        break
                        
                if not in_protected:
                    # Regular line processing
                    line = lines[i]
                    line_tokens = self.count_tokens(line)
                    
                    if current_tokens + line_tokens > self.chunk_size and current_chunk_lines:
                        # Create new chunk
                        new_chunk_content = '\n'.join(current_chunk_lines)
                        new_chunk = DocumentChunk(
                            content=new_chunk_content,
                            metadata=chunk.metadata,
                            chunk_id=f"{chunk.chunk_id}_p{len(new_chunks)}",
                            token_count=self.count_tokens(new_chunk_content),
                            start_char=chunk.start_char,
                            end_char=chunk.start_char + len(new_chunk_content)
                        )
                        new_chunks.append(new_chunk)
                        current_chunk_lines = [line]
                        current_tokens = line_tokens
                    else:
                        current_chunk_lines.append(line)
                        current_tokens += line_tokens
                        
                    i += 1
                    
            # Add remaining content
            if current_chunk_lines:
                new_chunk_content = '\n'.join(current_chunk_lines)
                new_chunk = DocumentChunk(
                    content=new_chunk_content,
                    metadata=chunk.metadata,
                    chunk_id=f"{chunk.chunk_id}_p{len(new_chunks)}",
                    token_count=self.count_tokens(new_chunk_content),
                    start_char=chunk.start_char,
                    end_char=chunk.start_char + len(new_chunk_content)
                )
                new_chunks.append(new_chunk)
                
            # Use new chunks or original if no splitting occurred
            if len(new_chunks) > 1:
                processed_chunks.extend(new_chunks)
            else:
                processed_chunks.append(chunk)
                
        return processed_chunks