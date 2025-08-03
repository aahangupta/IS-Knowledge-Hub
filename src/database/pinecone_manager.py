"""
Pinecone Database Integration
Handles connection, index management, and vector operations for Pinecone
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pinecone
from pinecone import Pinecone, ServerlessSpec, PodSpec
from config import settings
from ..embeddings import EmbeddingResult
from ..chunkers import DocumentChunk

logger = logging.getLogger(__name__)


class PineconeManager:
    """
    Manages Pinecone database connection, index, and operations
    """
    
    def __init__(self, api_key: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialize Pinecone manager
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
        """
        self.api_key = api_key or settings.pinecone_api_key
        self.environment = environment or settings.pinecone_environment
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        logger.info(f"Initialized PineconeManager for environment: {self.environment}")
        
    def get_index(self, index_name: str):
        """
        Get a Pinecone index object
        
        Args:
            index_name: Name of the index
            
        Returns:
            Pinecone index object or None if not found
        """
        if index_name not in self.pc.list_indexes().names():
            logger.warning(f"Index '{index_name}' not found")
            return None
            
        return self.pc.Index(index_name)
        
    def create_index_if_not_exists(
        self,
        index_name: str,
        dimension: int,
        metric: str = 'cosine',
        spec: Optional[Dict[str, Any]] = None
    ):
        """
        Create a Pinecone index if it doesn't already exist
        
        Args:
            index_name: Name of the index
            dimension: Dimension of the vectors
            metric: Similarity metric
            spec: Index specification (e.g., ServerlessSpec or PodSpec)
            
        Returns:
            Pinecone index object
        """
        if index_name not in self.pc.list_indexes().names():
            logger.info(f"Creating index '{index_name}' with dimension {dimension}...")
            
            # Default to serverless spec if not provided
            if spec is None:
                spec = ServerlessSpec(cloud='aws', region='us-east-1')
            
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=spec
            )
            logger.info(f"Index '{index_name}' created successfully")
            
        else:
            logger.info(f"Index '{index_name}' already exists")
            
        return self.pc.Index(index_name)
        
    def delete_index(self, index_name: str) -> None:
        """
        Delete a Pinecone index
        
        Args:
            index_name: Name of the index to delete
        """
        if index_name in self.pc.list_indexes().names():
            logger.info(f"Deleting index '{index_name}'...")
            self.pc.delete_index(index_name)
            logger.info(f"Index '{index_name}' deleted successfully")
        else:
            logger.warning(f"Index '{index_name}' not found for deletion")
            
    def list_indexes(self) -> List[str]:
        """
        List all available indexes
        
        Returns:
            List of index names
        """
        return self.pc.list_indexes().names()
        
    def describe_index(self, index_name: str) -> Optional[Dict[str, Any]]:
        """
        Get description of an index
        
        Args:
            index_name: Name of the index
            
        Returns:
            Index description or None if not found
        """
        if index_name not in self.pc.list_indexes().names():
            logger.warning(f"Index '{index_name}' not found")
            return None
            
        return self.pc.describe_index(index_name)
        
    def upsert_embeddings(
        self,
        index,
        chunk_embedding_pairs: List[Tuple[DocumentChunk, EmbeddingResult]],
        namespace: Optional[str] = None,
        batch_size: int = 100
    ) -> int:
        """
        Upsert embeddings to Pinecone index
        
        Args:
            index: Pinecone index object
            chunk_embedding_pairs: List of (chunk, embedding) tuples
            namespace: Optional namespace
            batch_size: Number of vectors per batch
            
        Returns:
            Number of upserted vectors
        """
        vectors_to_upsert = []
        for chunk, embedding in chunk_embedding_pairs:
            vector = {
                'id': chunk.chunk_id,
                'values': embedding.embedding,
                'metadata': chunk.metadata
            }
            vectors_to_upsert.append(vector)
            
        total_upserted = 0
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            upsert_response = index.upsert(
                vectors=batch,
                namespace=namespace
            )
            total_upserted += upsert_response['upserted_count']
            
            logger.info(f"Upserted batch {i//batch_size + 1}, total upserted: {total_upserted}")
            
        return total_upserted
        
    def query(
        self,
        index,
        query_embedding: EmbeddingResult,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone index
        
        Args:
            index: Pinecone index object
            query_embedding: Query embedding
            top_k: Number of top results
            namespace: Optional namespace
            filter_metadata: Optional metadata filter
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of query results
        """

        query_response = index.query(
            vector=query_embedding.embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter_metadata,
            include_metadata=include_metadata
        )
        
        logger.info(f"Query returned {len(query_response['matches'])} results")
        return query_response['matches']
        
    def delete_vectors(
        self,
        index,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> None:
        """
        Delete vectors from index by ID
        
        Args:
            index: Pinecone index object
            ids: List of vector IDs to delete
            namespace: Optional namespace
        """
        delete_response = index.delete(ids=ids, namespace=namespace)
        logger.info(f"Delete response: {delete_response}")
        
    def get_index_stats(self, index) -> Dict[str, Any]:
        """
        Get statistics about an index
        
        Args:
            index: Pinecone index object
            
        Returns:
            Dictionary with index statistics
        """
        stats = index.describe_index_stats()
        return {
            'dimension': stats.dimension,
            'index_fullness': stats.index_fullness,
            'total_vector_count': stats.total_vector_count,
            'namespaces': {
                ns: {
                    'vector_count': details.vector_count
                } for ns, details in stats.namespaces.items()
            }
        }