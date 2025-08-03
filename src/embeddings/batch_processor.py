"""
Batch Embedding Processor
Handles large-scale embedding generation with progress tracking and error recovery
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import time
from datetime import datetime
import pickle
from .embedding_generator import EmbeddingGenerator, EmbeddingResult
from ..chunkers import DocumentChunk

logger = logging.getLogger(__name__)


class BatchEmbeddingProcessor:
    """
    Processes document chunks in batches to generate embeddings efficiently
    """
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize batch processor
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            checkpoint_dir: Directory to save progress checkpoints
        """
        self.generator = embedding_generator
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
    def process_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 100,
        delay_ms: int = 100,
        checkpoint_interval: int = 500,
        resume_from_checkpoint: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Tuple[DocumentChunk, EmbeddingResult]]:
        """
        Process chunks to generate embeddings with checkpointing
        
        Args:
            chunks: List of document chunks
            batch_size: Number of chunks per API call
            delay_ms: Delay between batches in milliseconds
            checkpoint_interval: Save checkpoint every N chunks
            resume_from_checkpoint: Resume from last checkpoint if available
            progress_callback: Optional callback function(processed, total)
            
        Returns:
            List of (chunk, embedding) tuples
        """
        total_chunks = len(chunks)
        processed_chunks = 0
        results = []
        
        # Try to resume from checkpoint
        checkpoint_data = None
        if resume_from_checkpoint and self.checkpoint_dir:
            checkpoint_data = self._load_latest_checkpoint()
            
        if checkpoint_data:
            results = checkpoint_data['results']
            processed_chunks = checkpoint_data['processed']
            logger.info(f"Resumed from checkpoint: {processed_chunks}/{total_chunks} chunks processed")
        
        # Process remaining chunks
        start_time = time.time()
        
        for i in range(processed_chunks, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            try:
                # Generate embeddings for batch
                batch_results = self.generator.generate_embeddings_for_chunks(
                    chunks=batch_chunks,
                    batch_size=batch_size,
                    delay_ms=0  # We handle delay here
                )
                
                results.extend(batch_results)
                processed_chunks += len(batch_chunks)
                
                # Progress callback
                if progress_callback:
                    progress_callback(processed_chunks, total_chunks)
                    
                # Log progress
                elapsed = time.time() - start_time
                rate = processed_chunks / elapsed if elapsed > 0 else 0
                eta = (total_chunks - processed_chunks) / rate if rate > 0 else 0
                
                logger.info(
                    f"Progress: {processed_chunks}/{total_chunks} chunks "
                    f"({processed_chunks/total_chunks*100:.1f}%) | "
                    f"Rate: {rate:.1f} chunks/s | ETA: {eta:.0f}s"
                )
                
                # Save checkpoint
                if self.checkpoint_dir and processed_chunks % checkpoint_interval == 0:
                    self._save_checkpoint(results, processed_chunks)
                    
                # Delay between batches
                if i + batch_size < total_chunks:
                    time.sleep(delay_ms / 1000.0)
                    
            except Exception as e:
                logger.error(f"Error processing batch at index {i}: {e}")
                
                # Save checkpoint on error
                if self.checkpoint_dir:
                    self._save_checkpoint(results, processed_chunks, error=str(e))
                    
                raise
                
        # Final checkpoint
        if self.checkpoint_dir:
            self._save_checkpoint(results, processed_chunks, completed=True)
            
        # Validate embeddings
        embedding_results = [result for _, result in results]
        if not self.generator.validate_embedding_dimensions(embedding_results):
            logger.warning("Embedding dimension validation failed")
            
        logger.info(f"Completed processing {total_chunks} chunks in {time.time() - start_time:.1f}s")
        return results
        
    def _save_checkpoint(
        self,
        results: List[Tuple[DocumentChunk, EmbeddingResult]],
        processed: int,
        error: Optional[str] = None,
        completed: bool = False
    ) -> None:
        """Save processing checkpoint"""
        if not self.checkpoint_dir:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{timestamp}.pkl"
        
        checkpoint_data = {
            'results': results,
            'processed': processed,
            'timestamp': timestamp,
            'error': error,
            'completed': completed
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        logger.info(f"Saved checkpoint: {processed} chunks processed")
        
        # Keep only latest checkpoints
        self._cleanup_old_checkpoints(keep=3)
        
    def _load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint"""
        if not self.checkpoint_dir:
            return None
            
        checkpoint_files = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        
        if not checkpoint_files:
            return None
            
        latest_checkpoint = checkpoint_files[-1]
        
        try:
            with open(latest_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)
                
            logger.info(f"Loaded checkpoint from {latest_checkpoint}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
            
    def _cleanup_old_checkpoints(self, keep: int = 3) -> None:
        """Remove old checkpoint files"""
        if not self.checkpoint_dir:
            return
            
        checkpoint_files = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        
        if len(checkpoint_files) > keep:
            for old_file in checkpoint_files[:-keep]:
                old_file.unlink()
                logger.debug(f"Removed old checkpoint: {old_file}")
                
    def save_embeddings_to_file(
        self,
        chunk_embedding_pairs: List[Tuple[DocumentChunk, EmbeddingResult]],
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        Save embeddings to file
        
        Args:
            chunk_embedding_pairs: List of (chunk, embedding) tuples
            output_path: Output file path
            format: Output format ("json" or "pickle")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            # Convert to JSON-serializable format
            data = []
            for chunk, embedding in chunk_embedding_pairs:
                entry = {
                    'chunk': chunk.to_dict(),
                    'embedding': {
                        'vector': embedding.embedding,
                        'model': embedding.model,
                        'dimensions': embedding.dimensions
                    }
                }
                data.append(entry)
                
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        elif format == "pickle":
            # Save as pickle for faster loading
            with open(output_path, 'wb') as f:
                pickle.dump(chunk_embedding_pairs, f)
                
        else:
            raise ValueError(f"Unknown format: {format}")
            
        logger.info(f"Saved {len(chunk_embedding_pairs)} embeddings to {output_path}")
        
    def load_embeddings_from_file(
        self,
        file_path: str,
        format: str = "json"
    ) -> List[Tuple[DocumentChunk, EmbeddingResult]]:
        """
        Load embeddings from file
        
        Args:
            file_path: Input file path
            format: Input format ("json" or "pickle")
            
        Returns:
            List of (chunk, embedding) tuples
        """
        file_path = Path(file_path)
        
        if format == "json":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Reconstruct objects
            chunk_embedding_pairs = []
            for entry in data:
                chunk = DocumentChunk(**entry['chunk'])
                embedding = EmbeddingResult(
                    text=chunk.content,
                    embedding=entry['embedding']['vector'],
                    model=entry['embedding']['model'],
                    dimensions=entry['embedding']['dimensions'],
                    chunk_id=chunk.chunk_id,
                    metadata=chunk.metadata
                )
                chunk_embedding_pairs.append((chunk, embedding))
                
        elif format == "pickle":
            with open(file_path, 'rb') as f:
                chunk_embedding_pairs = pickle.load(f)
                
        else:
            raise ValueError(f"Unknown format: {format}")
            
        logger.info(f"Loaded {len(chunk_embedding_pairs)} embeddings from {file_path}")
        return chunk_embedding_pairs
        
    def compute_embedding_statistics(
        self,
        embeddings: List[EmbeddingResult]
    ) -> Dict[str, Any]:
        """
        Compute statistics about embeddings
        
        Args:
            embeddings: List of embedding results
            
        Returns:
            Dictionary with statistics
        """
        import numpy as np
        
        if not embeddings:
            return {
                'count': 0,
                'dimensions': 0
            }
            
        # Convert to numpy array
        embedding_matrix = np.array([e.embedding for e in embeddings])
        
        stats = {
            'count': len(embeddings),
            'dimensions': embeddings[0].dimensions,
            'model': embeddings[0].model,
            'mean_norm': float(np.mean(np.linalg.norm(embedding_matrix, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embedding_matrix, axis=1))),
            'mean_values': {
                'min': float(np.min(np.mean(embedding_matrix, axis=0))),
                'max': float(np.max(np.mean(embedding_matrix, axis=0))),
                'mean': float(np.mean(np.mean(embedding_matrix, axis=0)))
            }
        }
        
        # Compute pairwise similarities sample
        if len(embeddings) > 1:
            sample_size = min(100, len(embeddings))
            sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
            
            similarities = []
            for i in range(sample_size):
                for j in range(i + 1, sample_size):
                    sim = self.generator.compute_similarity(
                        embeddings[sample_indices[i]],
                        embeddings[sample_indices[j]]
                    )
                    similarities.append(sim)
                    
            stats['similarity_sample'] = {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities))
            }
            
        return stats