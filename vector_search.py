"""
Vector search engine for the Synapse Tracker.

Uses FAISS for fast similarity search over memory shard embeddings.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class VectorSearchEngine:
    """
    Fast vector similarity search using FAISS.
    
    Stores embeddings for each memory shard and enables semantic search
    to find the most relevant shards for a query.
    """
    
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "flat",
        index_path: Optional[str] = None,
    ):
        """
        Initialize the vector search engine.
        
        Args:
            dimension: Embedding dimension (768 for Nomic Embed)
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            index_path: Path to save/load index
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index_path = Path(index_path) if index_path else Path("./vector_index.faiss")
        
        self.index = None
        self.info_hashes = []  # Maps index position to info_hash
        
        self._init_index()
        logger.info(f"VectorSearchEngine initialized: {index_type}, dim={dimension}")
    
    def _init_index(self):
        """Initialize or load FAISS index."""
        try:
            import faiss
        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu")
            raise RuntimeError("FAISS required for vector search")
        
        # Try to load existing index
        if self.index_path.exists():
            self.load_index()
        else:
            # Create new index
            if self.index_type == "flat":
                # Exact search, no compression
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
            elif self.index_type == "ivf":
                # Inverted file index (faster, approximate)
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
                self.index.nprobe = 10  # Search 10 clusters
            elif self.index_type == "hnsw":
                # Hierarchical Navigable Small World (fast, accurate)
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # M=32
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
            
            logger.info(f"Created new {self.index_type} index")
    
    def add_embedding(self, info_hash: str, embedding: np.ndarray) -> bool:
        """
        Add an embedding to the index.
        
        Args:
            info_hash: Unique identifier for the shard
            embedding: 768-dim embedding vector
            
        Returns:
            True if successful
        """
        if embedding.shape != (self.dimension,):
            logger.error(f"Invalid embedding dimension: {embedding.shape}")
            return False
        
        # Normalize embedding for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        # Check if already exists
        if info_hash in self.info_hashes:
            logger.warning(f"Embedding already exists: {info_hash}")
            return False
        
        # Add to index
        self.index.add(embedding.reshape(1, -1).astype('float32'))
        self.info_hashes.append(info_hash)
        
        logger.debug(f"Added embedding for {info_hash}")
        return True
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of (info_hash, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        k = min(k, self.index.ntotal)
        similarities, indices = self.index.search(query_embedding, k)
        
        # Convert to results
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx >= 0 and idx < len(self.info_hashes):
                # Convert inner product back to similarity
                similarity = float(similarity)
                
                if similarity >= min_similarity:
                    results.append((self.info_hashes[idx], similarity))
        
        logger.debug(f"Search returned {len(results)} results")
        return results
    
    def remove_embedding(self, info_hash: str) -> bool:
        """
        Remove an embedding from the index.
        
        Note: FAISS doesn't support efficient removal, so we rebuild the index.
        For production, consider using a flag-based soft delete.
        
        Args:
            info_hash: The info hash to remove
            
        Returns:
            True if successful
        """
        if info_hash not in self.info_hashes:
            return False
        
        # Remove from mapping
        self.info_hashes.remove(info_hash)
        
        # For production: implement soft delete or periodic rebuild
        logger.warning("Embedding removal requires index rebuild (TODO)")
        return True
    
    def save_index(self, path: Optional[str] = None):
        """
        Save index and mappings to disk.
        
        Args:
            path: Optional custom path
        """
        try:
            import faiss
            
            save_path = Path(path) if path else self.index_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(save_path))
            
            # Save info_hash mappings
            mapping_path = save_path.with_suffix('.pkl')
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.info_hashes, f)
            
            logger.info(f"Index saved to {save_path}")
        
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load_index(self, path: Optional[str] = None):
        """
        Load index and mappings from disk.
        
        Args:
            path: Optional custom path
        """
        try:
            import faiss
            
            load_path = Path(path) if path else self.index_path
            
            if not load_path.exists():
                logger.warning(f"Index file not found: {load_path}")
                return
            
            # Load FAISS index
            self.index = faiss.read_index(str(load_path))
            
            # Load info_hash mappings
            mapping_path = load_path.with_suffix('.pkl')
            if mapping_path.exists():
                with open(mapping_path, 'rb') as f:
                    self.info_hashes = pickle.load(f)
            
            logger.info(f"Index loaded: {self.index.ntotal} embeddings")
        
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self._init_index()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_embeddings": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "index_size_mb": self.index_path.stat().st_size / (1024 * 1024) if self.index_path.exists() else 0,
        }
    
    def rebuild_index(self, embeddings: List[Tuple[str, np.ndarray]]):
        """
        Rebuild the entire index from scratch.
        
        Useful after removing many embeddings or changing index type.
        
        Args:
            embeddings: List of (info_hash, embedding) tuples
        """
        logger.info(f"Rebuilding index with {len(embeddings)} embeddings")
        
        # Reset index
        self._init_index()
        self.info_hashes = []
        
        # Add all embeddings
        for info_hash, embedding in embeddings:
            self.add_embedding(info_hash, embedding)
        
        # Save
        self.save_index()
        
        logger.info("Index rebuild complete")


def create_search_engine(dimension: int = 768, index_type: str = "flat") -> VectorSearchEngine:
    """
    Factory function to create a VectorSearchEngine.
    
    Args:
        dimension: Embedding dimension
        index_type: FAISS index type
        
    Returns:
        Configured VectorSearchEngine instance
    """
    return VectorSearchEngine(
        dimension=dimension,
        index_type=index_type,
        index_path="./vector_index.faiss",
    )


# Example usage
if __name__ == "__main__":
    # Test the search engine
    engine = create_search_engine()
    
    # Add some test embeddings
    test_embeddings = [
        ("hash1", np.random.randn(768)),
        ("hash2", np.random.randn(768)),
        ("hash3", np.random.randn(768)),
    ]
    
    for info_hash, embedding in test_embeddings:
        engine.add_embedding(info_hash, embedding)
    
    # Search
    query = np.random.randn(768)
    results = engine.search(query, k=3)
    
    print(f"Search results: {results}")
    print(f"Statistics: {engine.get_statistics()}")
