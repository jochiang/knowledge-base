"""
Knowledge Harness - Embeddings Module

Local embedding using sentence-transformers with MiniLM.
Embeddings are stored in SQLite for persistence.
"""

import numpy as np
from typing import Optional, Callable
from pathlib import Path

from schema import KnowledgeDB, Chunk


# Default model - small, fast, good quality
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class LocalEmbedder:
    """
    Local embedding using sentence-transformers.

    Manages model loading, embedding computation, and caching to SQLite.
    """

    def __init__(
        self,
        db: KnowledgeDB,
        model_name: str = DEFAULT_MODEL,
        device: str = None,  # None = auto-detect (CUDA if available)
    ):
        self.db = db
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = None

    @property
    def model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            # Load model to get dimension
            _ = self.model
        return self._dimension

    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )

        import sys
        # Use stderr for logging since stdout is used for MCP protocol
        print(f"Loading embedding model: {self.model_name}...", file=sys.stderr)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        print(f"Model loaded. Dimension: {self._dimension}", file=sys.stderr)

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Returns a list of floats (the embedding vector).
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embed multiple texts efficiently in batches.

        Returns list of embedding vectors.
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.tolist()

    def embed_chunk(self, chunk: Chunk, store: bool = True) -> list[float]:
        """
        Embed a chunk, optionally storing to DB.

        Checks cache first.
        """
        # Check if we already have it
        existing = self.db.get_embedding(chunk.id)
        if existing is not None:
            return existing

        # Compute embedding
        # Use summary + content for richer representation
        text = chunk.content
        if chunk.summary:
            text = f"{chunk.summary}\n\n{text}"

        embedding = self.embed(text)

        if store:
            self.db.store_embedding(chunk.id, embedding, self.model_name)

        return embedding

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 32,
        skip_existing: bool = True
    ) -> dict[str, list[float]]:
        """
        Embed multiple chunks efficiently.

        Returns dict mapping chunk_id -> embedding.
        """
        results = {}
        to_embed = []
        to_embed_ids = []

        for chunk in chunks:
            if skip_existing:
                existing = self.db.get_embedding(chunk.id)
                if existing is not None:
                    results[chunk.id] = existing
                    continue

            # Prepare text
            text = chunk.content
            if chunk.summary:
                text = f"{chunk.summary}\n\n{text}"

            to_embed.append(text)
            to_embed_ids.append(chunk.id)

        if to_embed:
            import sys
            print(f"Embedding {len(to_embed)} chunks...", file=sys.stderr)
            embeddings = self.embed_batch(to_embed, batch_size)

            for chunk_id, embedding in zip(to_embed_ids, embeddings):
                self.db.store_embedding(chunk_id, embedding, self.model_name)
                results[chunk_id] = embedding

        return results

    def embed_missing(self, batch_size: int = 32) -> int:
        """
        Embed all chunks that don't have embeddings yet.

        Returns count of newly embedded chunks.
        """
        missing = self.db.get_chunks_without_embeddings(self.model_name)
        if not missing:
            return 0

        self.embed_chunks(missing, batch_size, skip_existing=False)
        return len(missing)

    def get_embedding_fn(self) -> Callable[[str], list[float]]:
        """
        Return a simple embedding function for use with Retriever.

        This doesn't cache - use embed_chunk for cached embeddings.
        """
        return self.embed

    def similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def find_similar(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> list[tuple[str, float]]:
        """
        Find chunks most similar to a query.

        Returns list of (chunk_id, similarity_score) tuples.
        """
        query_embedding = self.embed(query)

        # Get all stored embeddings
        all_embeddings = self.db.get_all_embeddings(self.model_name)

        if not all_embeddings:
            return []

        # Compute similarities
        query_vec = np.array(query_embedding)

        scores = []
        for chunk_id, embedding in all_embeddings:
            chunk_vec = np.array(embedding)
            sim = float(np.dot(query_vec, chunk_vec) /
                       (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)))
            if sim >= min_similarity:
                scores.append((chunk_id, sim))

        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


# ============================================================================
# Convenience functions
# ============================================================================

def create_embedder(db: KnowledgeDB, model_name: str = DEFAULT_MODEL) -> LocalEmbedder:
    """Create an embedder instance."""
    return LocalEmbedder(db, model_name)


def embed_all_chunks(db: KnowledgeDB, model_name: str = DEFAULT_MODEL) -> int:
    """Embed all chunks that don't have embeddings."""
    embedder = LocalEmbedder(db, model_name)
    return embedder.embed_missing()


def semantic_search(
    db: KnowledgeDB,
    query: str,
    top_k: int = 10,
    model_name: str = DEFAULT_MODEL
) -> list[tuple[Chunk, float]]:
    """
    Simple semantic search.

    Returns list of (Chunk, similarity_score) tuples.
    """
    embedder = LocalEmbedder(db, model_name)
    results = embedder.find_similar(query, top_k)

    return [
        (db.get_chunk(chunk_id), score)
        for chunk_id, score in results
        if db.get_chunk(chunk_id) is not None
    ]


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    from schema import init_db, Document, ContentType
    from ingest import quick_ingest

    print("Initializing in-memory database...")
    db = init_db(":memory:")

    # Ingest some content
    print("\nIngesting sample content...")
    quick_ingest(db, "ml_intro.md", """
# Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables
computers to learn from data without being explicitly programmed.

## Supervised Learning

In supervised learning, models learn from labeled examples. The algorithm
learns a mapping from inputs to outputs based on training data.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data, such as clustering
similar items together or reducing dimensionality.
""")

    quick_ingest(db, "neural_nets.md", """
# Neural Networks

Neural networks are computing systems inspired by biological neural networks.
They consist of layers of interconnected nodes that process information.

## Deep Learning

Deep learning uses neural networks with many layers (deep networks) to
learn hierarchical representations of data.

## Backpropagation

Backpropagation is the algorithm used to train neural networks by
computing gradients of the loss function with respect to weights.
""")

    print(f"\nStats: {db.stats()}")

    # Create embedder and embed all chunks
    print("\nCreating embedder...")
    embedder = LocalEmbedder(db)

    count = embedder.embed_missing()
    print(f"Embedded {count} chunks")
    print(f"Updated stats: {db.stats()}")

    # Test semantic search
    print("\n" + "="*60)
    print("Testing semantic search...")
    print("="*60)

    queries = [
        "How do neural networks learn?",
        "What is the difference between supervised and unsupervised?",
        "deep learning architecture"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = embedder.find_similar(query, top_k=3)
        for chunk_id, score in results:
            chunk = db.get_chunk(chunk_id)
            preview = chunk.content[:80].replace('\n', ' ')
            print(f"  [{score:.3f}] {preview}...")

    db.close()
    print("\nDemo complete!")
