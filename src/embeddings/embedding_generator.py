"""
Embedding Generator Module
Generates vector embeddings using OpenAI's text-embedding-3-small model
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from openai import OpenAI
from openai.types import CreateEmbeddingResponse, Embedding
import tiktoken
from dataclasses import dataclass, asdict
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """
    Represents an embedding result with metadata
    """
    text: str
    embedding: List[float]
    model: str
    dimensions: int
    chunk_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_numpy(self) -> np.ndarray:
        """Convert embedding to numpy array"""
        return np.array(self.embedding)


class EmbeddingGenerator:
    """
    Generates embeddings using OpenAI's embedding models
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize embedding generator
        
        Args:
            model: OpenAI embedding model to use
            dimensions: Optional dimensions for embedding (None uses model default)
            api_key: OpenAI API key (uses settings if not provided)
        """
        self.model = model
        self.dimensions = dimensions
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        
        # Initialize tokenizer for the model
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except:
            # Fallback to cl100k_base encoding
            self.encoder = tiktoken.get_encoding("cl100k_base")
            
        # Model dimension defaults
        self.default_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        # Set actual dimensions
        if self.dimensions is None:
            self.dimensions = self.default_dimensions.get(model, 1536)
            
        logger.info(f"Initialized EmbeddingGenerator with model: {model}, dimensions: {self.dimensions}")
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))
        
    def generate_embedding(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            chunk_id: Optional chunk identifier
            metadata: Optional metadata to attach
            
        Returns:
            EmbeddingResult object
        """
        try:
            # Create embedding
            response: CreateEmbeddingResponse = self.client.embeddings.create(
                input=text,
                model=self.model,
                dimensions=self.dimensions if self.model.startswith("text-embedding-3") else None
            )
            
            # Extract embedding
            embedding_data: Embedding = response.data[0]
            embedding_vector = embedding_data.embedding
            
            # Create result
            result = EmbeddingResult(
                text=text,
                embedding=embedding_vector,
                model=self.model,
                dimensions=len(embedding_vector),
                chunk_id=chunk_id,
                metadata=metadata
            )
            
            logger.debug(f"Generated embedding for chunk {chunk_id} with {len(embedding_vector)} dimensions")
            return result
            
        except Exception as e:
            logger.error(f"Error generating embedding for chunk {chunk_id}: {e}")
            raise
            
    def generate_embeddings_batch(
        self,
        texts: List[str],
        chunk_ids: Optional[List[str]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
        delay_ms: int = 100
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts in batches
        
        Args:
            texts: List of texts to embed
            chunk_ids: Optional list of chunk identifiers
            metadata_list: Optional list of metadata dictionaries
            batch_size: Number of texts per API call
            delay_ms: Delay between batches in milliseconds
            
        Returns:
            List of EmbeddingResult objects
        """
        results = []
        
        # Prepare chunk_ids and metadata
        if chunk_ids is None:
            chunk_ids = [None] * len(texts)
        if metadata_list is None:
            metadata_list = [None] * len(texts)
            
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = chunk_ids[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size]
            
            try:
                # Create embeddings for batch
                response: CreateEmbeddingResponse = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model,
                    dimensions=self.dimensions if self.model.startswith("text-embedding-3") else None
                )
                
                # Process each embedding
                for j, embedding_data in enumerate(response.data):
                    result = EmbeddingResult(
                        text=batch_texts[j],
                        embedding=embedding_data.embedding,
                        model=self.model,
                        dimensions=len(embedding_data.embedding),
                        chunk_id=batch_ids[j],
                        metadata=batch_metadata[j]
                    )
                    results.append(result)
                    
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                # Delay between batches to avoid rate limits
                if i + batch_size < len(texts):
                    time.sleep(delay_ms / 1000.0)
                    
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                raise
                
        return results
        
    def generate_embeddings_for_chunks(
        self,
        chunks: List[Any],  # DocumentChunk objects
        batch_size: int = 100,
        delay_ms: int = 100
    ) -> List[Tuple[Any, EmbeddingResult]]:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of DocumentChunk objects
            batch_size: Number of chunks per API call
            delay_ms: Delay between batches in milliseconds
            
        Returns:
            List of tuples (chunk, embedding_result)
        """
        # Extract texts and metadata
        texts = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        metadata_list = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings
        embedding_results = self.generate_embeddings_batch(
            texts=texts,
            chunk_ids=chunk_ids,
            metadata_list=metadata_list,
            batch_size=batch_size,
            delay_ms=delay_ms
        )
        
        # Combine chunks with embeddings
        chunk_embedding_pairs = list(zip(chunks, embedding_results))
        
        return chunk_embedding_pairs
        
    def validate_embedding_dimensions(self, embeddings: List[EmbeddingResult]) -> bool:
        """
        Validate that all embeddings have the same dimensions
        
        Args:
            embeddings: List of embedding results
            
        Returns:
            True if all embeddings have the same dimensions
        """
        if not embeddings:
            return True
            
        expected_dim = embeddings[0].dimensions
        
        for embedding in embeddings:
            if embedding.dimensions != expected_dim:
                logger.error(
                    f"Dimension mismatch: expected {expected_dim}, got {embedding.dimensions} "
                    f"for chunk {embedding.chunk_id}"
                )
                return False
                
        return True
        
    def compute_similarity(
        self,
        embedding1: EmbeddingResult,
        embedding2: EmbeddingResult,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric ("cosine" or "dot")
            
        Returns:
            Similarity score
        """
        vec1 = np.array(embedding1.embedding)
        vec2 = np.array(embedding2.embedding)
        
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
            
        elif metric == "dot":
            # Dot product
            return np.dot(vec1, vec2)
            
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
    def find_similar_embeddings(
        self,
        query_embedding: EmbeddingResult,
        candidate_embeddings: List[EmbeddingResult],
        top_k: int = 10,
        metric: str = "cosine",
        threshold: Optional[float] = None
    ) -> List[Tuple[EmbeddingResult, float]]:
        """
        Find most similar embeddings to a query
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            metric: Similarity metric
            threshold: Optional similarity threshold
            
        Returns:
            List of (embedding, similarity_score) tuples
        """
        similarities = []
        
        for candidate in candidate_embeddings:
            score = self.compute_similarity(query_embedding, candidate, metric)
            
            if threshold is None or score >= threshold:
                similarities.append((candidate, score))
                
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return similarities[:top_k]