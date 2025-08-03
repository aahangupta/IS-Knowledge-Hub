"""
Semantic Search Engine for IS Codes
Orchestrates embedding generation and vector search
"""

import logging
from typing import List, Dict, Any, Optional
from ..embeddings import EmbeddingGenerator
from ..database import PineconeManager
from .result_models import SearchResult
from config import settings

logger = logging.getLogger(__name__)

class SearchEngine:
    """
    Performs semantic search over IS code documents
    """
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        pinecone_manager: PineconeManager,
        index_name: Optional[str] = None
    ):
        """
        Initialize search engine
        
        Args:
            embedding_generator: Instance of EmbeddingGenerator
            pinecone_manager: Instance of PineconeManager
            index_name: Name of the Pinecone index to use
        """
        self.generator = embedding_generator
        self.pinecone = pinecone_manager
        self.index_name = index_name or settings.pinecone_index_name
        
        # Get Pinecone index
        self.index = self.pinecone.get_index(self.index_name)
        if self.index is None:
            logger.warning(
                f"Pinecone index '{self.index_name}' not found. "
                "Search functionality will be limited. Please create the index."
            )
            
    def search(
        self,
        query: str,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform a semantic search
        
        Args:
            query: User's search query
            top_k: Number of results to return
            namespace: Optional Pinecone namespace
            filter_metadata: Optional metadata filter for Pinecone
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            logger.error(f"Cannot perform search: Pinecone index '{self.index_name}' is not available.")
            return []
            
        logger.info(f"Performing search for query: '{query}'")
        
        # 1. Generate query embedding
        query_embedding = self.generator.generate_embedding(query)
        
        # 2. Query Pinecone
        try:
            query_results = self.pinecone.query(
                index=self.index,
                query_embedding=query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter_metadata=filter_metadata,
                include_metadata=True
            )
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            return []
            
        # 3. Format results
        search_results = []
        for match in query_results:
            result = SearchResult(
                chunk_id=match['id'],
                content=match.get('metadata', {}).get('content', ''),
                score=match['score'],
                metadata=match.get('metadata', {})
            )
            # Pinecone metadata often contains the original text content
            # We ensure we have it in our result.
            if not result.content and 'text' in result.metadata:
                 result.content = result.metadata['text']
            
            search_results.append(result)
            
        logger.info(f"Found {len(search_results)} results")
        return search_results
        
    def get_context_for_query(
        self,
        query: str,
        top_k: int = 5,
        max_tokens: int = 4000
    ) -> str:
        """
        Get formatted context for a query, suitable for RAG
        
        Args:
            query: User's search query
            top_k: Number of results to retrieve
            max_tokens: Maximum tokens for the context
            
        Returns:
            Formatted context string
        """
        from .result_models import ResultFormatter
        
        # Perform search
        search_results = self.search(query, top_k=top_k)
        
        # Format for RAG
        context = ResultFormatter.format_for_rag(
            results=search_results,
            max_tokens=max_tokens,
            tokenizer=self.generator.encoder
        )
        
        return context
        
    def set_index(self, index_name: str) -> bool:
        """
        Set or change the Pinecone index for the search engine
        
        Args:
            index_name: Name of the new index
            
        Returns:
            True if index was set successfully, False otherwise
        """
        index = self.pinecone.get_index(index_name)
        if index:
            self.index_name = index_name
            self.index = index
            logger.info(f"Search engine index set to '{index_name}'")
            return True
        else:
            logger.error(f"Failed to set index to '{index_name}': not found")
            return False