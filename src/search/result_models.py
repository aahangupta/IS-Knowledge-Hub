"""
Data models for search results
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

@dataclass
class SearchResult:
    """
    Represents a single search result
    """
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class ResultFormatter:
    """
    Formats search results for display
    """
    
    @staticmethod
    def format_to_string(results: List[SearchResult], with_content: bool = True) -> str:
        """
        Format search results to a simple string
        
        Args:
            results: List of search results
            with_content: Whether to include chunk content
            
        Returns:
            Formatted string representation of results
        """
        if not results:
            return "No results found."
            
        output = []
        for i, res in enumerate(results):
            output.append(f"Result {i+1} (Score: {res.score:.4f}):")
            output.append(f"  Chunk ID: {res.chunk_id}")
            
            # Extract key metadata
            is_code = res.metadata.get('is_code', 'N/A')
            clause = res.metadata.get('clause', 'N/A')
            page = res.metadata.get('page', 'N/A')
            
            output.append(f"  Source: {is_code}, Clause: {clause}, Page: {page}")
            
            if with_content:
                output.append(f"  Content: {res.content.strip()}")
                
            output.append("-" * 20)
            
        return "\n".join(output)
        
    @staticmethod
    def format_to_markdown(results: List[SearchResult]) -> str:
        """
        Format search results to Markdown
        
        Args:
            results: List of search results
            
        Returns:
            Formatted Markdown string
        """
        if not results:
            return "*No results found.*"
            
        output = ["### Search Results\n"]
        for res in results:
            output.append(f"**Score: {res.score:.4f}**")
            
            is_code = res.metadata.get('is_code', 'N/A')
            clause = res.metadata.get('clause', 'N/A')
            page = res.metadata.get('page', 'N/A')
            
            output.append(f"- **Source**: `{is_code}`")
            output.append(f"- **Clause**: `{clause}`")
            output.append(f"- **Page**: `{page}`")
            
            output.append("\n> ```")
            output.append(f"> {res.content.strip()}")
            output.append("> ```\n")
            
        return "\n".join(output)
        
    @staticmethod
    def format_for_rag(
        results: List[SearchResult],
        max_tokens: int = 4000,
        tokenizer: Optional[Any] = None
    ) -> str:
        """
        Format search results as context for RAG
        
        Args:
            results: List of search results
            max_tokens: Maximum number of tokens for context
            tokenizer: Optional tokenizer for accurate token counting
            
        Returns:
            Formatted context string
        """
        context = []
        total_tokens = 0
        
        for res in results:
            source_info = (
                f"[Source: {res.metadata.get('is_code', 'N/A')}, "
                f"Clause: {res.metadata.get('clause', 'N/A')}]"
            )
            content_with_source = f"{source_info}\n{res.content.strip()}"
            
            # Token counting
            if tokenizer:
                tokens = len(tokenizer.encode(content_with_source))
            else:
                tokens = len(content_with_source.split()) # Approximation
                
            if total_tokens + tokens > max_tokens:
                break
                
            context.append(content_with_source)
            total_tokens += tokens
            
        return "\n\n---\n\n".join(context)