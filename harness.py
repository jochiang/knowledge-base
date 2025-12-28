"""
Knowledge Harness - Main Entry Point

This is the primary interface for using the knowledge harness.
Designed for use with Claude Code.

Example usage:
    from harness import KnowledgeHarness
    
    harness = KnowledgeHarness("./my_knowledge.db")
    
    # Ingest content
    harness.ingest_file("./documents/article.md")
    harness.ingest_text("Some text content", title="Quick Note")
    
    # Search
    results = harness.search("machine learning optimization")
    for r in results:
        print(r.chunk.content)
    
    # Record usage
    harness.record_win(chunk_ids, "Helped with ML project")
"""

import sys
from pathlib import Path
from typing import Optional, Callable
import yaml

# Add src to path if needed
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from schema import (
    KnowledgeDB, Document, Chunk, Concept,
    ContentType, UsageOutcome, TaskType, init_db
)
from ingest import IngestPipeline, IngestResult, quick_ingest, ingest_file
from retrieve import (
    Retriever, RetrievalResult, RetrievalConfig, RetrievalStrategy,
    GraphRetriever, format_results
)
from record import SessionRecorder, record_usage, chunk_report

# Optional: embeddings (requires sentence-transformers)
try:
    from embeddings import LocalEmbedder, DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    LocalEmbedder = None
    DEFAULT_EMBEDDING_MODEL = None


class KnowledgeHarness:
    """
    Unified interface to the knowledge management system.
    
    This class provides a clean API for all operations:
    - Ingestion
    - Retrieval
    - Usage recording
    - Maintenance
    
    LLM-powered features (summarization, concept extraction) can be
    plugged in via callbacks.
    """
    
    def __init__(
        self,
        db_path: str = "./knowledge.db",
        config_path: str = None,
        # LLM callbacks (optional)
        summarize_chunk: Callable[[str], str] = None,
        summarize_document: Callable[[str, list[str]], str] = None,
        extract_concepts: Callable[[str], list[dict]] = None,
        # Embedding options
        enable_embeddings: bool = True,  # Auto-enable if sentence-transformers available
        embedding_model: str = None,  # Uses DEFAULT_EMBEDDING_MODEL if None
    ):
        """
        Initialize the knowledge harness.

        Args:
            db_path: Path to SQLite database (created if doesn't exist)
            config_path: Path to config.yaml (optional)
            summarize_chunk: Function to summarize a chunk -> str
            summarize_document: Function to summarize doc (content, chunk_summaries) -> str
            extract_concepts: Function to extract concepts -> list of {name, type, description}
            enable_embeddings: Enable local MiniLM embeddings (requires sentence-transformers)
            embedding_model: Override embedding model name (default: all-MiniLM-L6-v2)
        """
        self.db_path = Path(db_path)
        self.db = init_db(self.db_path)

        # Load config if provided
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)

        # Store callbacks
        self._summarize_chunk = summarize_chunk
        self._summarize_document = summarize_document
        self._extract_concepts = extract_concepts

        # Initialize embedder if enabled and available
        self._embedder = None
        self._embed_fn = None
        if enable_embeddings and EMBEDDINGS_AVAILABLE:
            model = embedding_model or DEFAULT_EMBEDDING_MODEL
            self._embedder = LocalEmbedder(self.db, model_name=model)
            self._embed_fn = self._embedder.embed
        elif enable_embeddings and not EMBEDDINGS_AVAILABLE:
            import sys
            print("Warning: sentence-transformers not installed. Semantic search disabled.", file=sys.stderr)
            print("Install with: pip install sentence-transformers", file=sys.stderr)

        # Initialize components
        self._init_components()

        # Session management
        self._current_session: Optional[SessionRecorder] = None
    
    def _init_components(self):
        """Initialize retriever and pipeline with current settings."""
        retrieval_config = RetrievalConfig()
        if "retrieval" in self.config:
            rc = self.config["retrieval"]
            if "weights" in rc:
                retrieval_config.weight_keyword = rc["weights"].get("keyword", 0.2)
                retrieval_config.weight_semantic = rc["weights"].get("semantic", 0.3)
                retrieval_config.weight_concept = rc["weights"].get("concept", 0.2)
                retrieval_config.weight_usage = rc["weights"].get("usage", 0.2)
                retrieval_config.weight_recency = rc["weights"].get("recency", 0.1)
            if "recency_halflife_days" in rc:
                retrieval_config.recency_halflife_days = rc["recency_halflife_days"]

        self.retriever = Retriever(
            db=self.db,
            config=retrieval_config,
            embed_fn=self._embed_fn,
            use_stored_embeddings=True  # Use embeddings from DB
        )

        self.graph_retriever = GraphRetriever(self.db)

        self.pipeline = IngestPipeline(
            db=self.db,
            summarize_chunk=self._summarize_chunk,
            summarize_document=self._summarize_document,
            extract_concepts=self._extract_concepts,
            embedder=self._embedder  # Pass embedder for eager embedding on ingest
        )
    
    # ========================================================================
    # Ingestion
    # ========================================================================
    
    def ingest_file(self, filepath: str | Path, title: str = None) -> IngestResult:
        """
        Ingest a file from disk.
        
        Supports: .md, .txt, .py, .js, .ts, and other text files.
        """
        path = Path(filepath)
        content = path.read_text(encoding='utf-8')
        return self.pipeline.ingest(
            source=str(path),
            content=content,
            title=title or path.stem
        )
    
    def ingest_text(
        self,
        content: str,
        title: str,
        source: str = "manual",
        content_type: ContentType = None
    ) -> IngestResult:
        """Ingest raw text content."""
        return self.pipeline.ingest(
            source=source,
            content=content,
            title=title,
            content_type=content_type
        )
    
    def ingest_url(self, url: str, content: str, title: str = None) -> IngestResult:
        """
        Ingest content from a URL.
        
        Note: You need to fetch the content yourself (e.g., via web_fetch).
        """
        return self.pipeline.ingest(
            source=url,
            content=content,
            title=title
        )
    
    def quick_ingest(self, source: str, content: str, title: str = None) -> IngestResult:
        """
        Fast ingestion without LLM features.
        
        Use this for bulk imports or when you'll add metadata later.
        """
        return quick_ingest(self.db, source, content, title)
    
    # ========================================================================
    # Retrieval
    # ========================================================================
    
    def search(
        self,
        query: str,
        limit: int = 10,
        strategies: list[str] = None,
        task_type: str | TaskType = None
    ) -> list[RetrievalResult]:
        """
        Search the knowledge base.

        Args:
            query: Search query
            limit: Maximum results
            strategies: List of strategies to use. Options:
                       "keyword", "semantic", "concept", "usage", "recency"
                       Default: all applicable strategies
            task_type: Optional task type for context-aware retrieval.
                      If provided, boosts chunks that performed well for this task type.
                      Options: "factual_lookup", "implementation_howto",
                              "conceptual_understanding", "debugging",
                              "decision_support", "exploratory_research", "other"

        Returns:
            List of RetrievalResult with chunk, document, scores, etc.
        """
        strategy_enums = None
        if strategies:
            strategy_enums = [RetrievalStrategy(s) for s in strategies]

        # Convert task_type string to enum if needed
        task_type_enum = None
        if task_type:
            if isinstance(task_type, str):
                task_type_enum = TaskType(task_type)
            else:
                task_type_enum = task_type

        return self.retriever.retrieve(
            query,
            strategies=strategy_enums,
            limit=limit,
            task_type=task_type_enum
        )
    
    def search_formatted(self, query: str, limit: int = 10, verbose: bool = False) -> str:
        """Search and return formatted string output."""
        results = self.search(query, limit)
        return format_results(results, verbose=verbose)
    
    def find_related(self, chunk_id: str, limit: int = 10) -> list[tuple[Chunk, list[str]]]:
        """Find chunks related to a given chunk via shared concepts."""
        return self.graph_retriever.find_related_via_concepts(chunk_id, limit)
    
    def traverse(
        self,
        seed_chunk_ids: list[str],
        max_hops: int = 2
    ) -> list[tuple[str, float, list[str]]]:
        """
        Traverse the knowledge graph from seed chunks.
        
        Returns: [(chunk_id, score, path), ...]
        """
        return self.graph_retriever.traverse_from_chunks(
            seed_chunk_ids, max_hops=max_hops
        )
    
    # ========================================================================
    # Usage Recording
    # ========================================================================
    
    def start_session(self, session_id: str = None) -> SessionRecorder:
        """Start a new recording session."""
        self._current_session = SessionRecorder(self.db, session_id)
        return self._current_session
    
    @property
    def session(self) -> Optional[SessionRecorder]:
        """Get current session (if any)."""
        return self._current_session
    
    def record_win(self, chunk_ids: list[str], context: str, notes: str = None) -> str:
        """Record successful use of chunks."""
        return record_usage(
            self.db, chunk_ids, context, "win",
            session_id=self._current_session.session_id if self._current_session else None,
            notes=notes
        )
    
    def record_partial(self, chunk_ids: list[str], context: str, notes: str = None) -> str:
        """Record partial success."""
        return record_usage(
            self.db, chunk_ids, context, "partial",
            session_id=self._current_session.session_id if self._current_session else None,
            notes=notes
        )
    
    def record_miss(self, chunk_ids: list[str], context: str, notes: str = None) -> str:
        """Record that chunks were not useful."""
        return record_usage(
            self.db, chunk_ids, context, "miss",
            session_id=self._current_session.session_id if self._current_session else None,
            notes=notes
        )
    
    def record_misleading(self, chunk_ids: list[str], context: str, notes: str = None) -> str:
        """Record that chunks were actively misleading."""
        return record_usage(
            self.db, chunk_ids, context, "misleading",
            session_id=self._current_session.session_id if self._current_session else None,
            notes=notes
        )
    
    # ========================================================================
    # Queries and Reports
    # ========================================================================
    
    def stats(self) -> dict:
        """Get database statistics."""
        return self.db.stats()
    
    def chunk_info(self, chunk_id: str) -> dict:
        """Get detailed info about a chunk including usage history."""
        return chunk_report(self.db, chunk_id)
    
    def list_documents(self, limit: int = 50) -> list[Document]:
        """List all documents."""
        rows = self.db.conn.execute(
            "SELECT * FROM documents ORDER BY last_accessed DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [self.db._row_to_document(row) for row in rows]
    
    def list_concepts(self) -> list[Concept]:
        """List all concepts."""
        return self.db.get_all_concepts()
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.db.get_document(doc_id)
    
    def get_chunks(self, doc_id: str) -> list[Chunk]:
        """Get all chunks for a document."""
        return self.db.get_chunks_for_document(doc_id)
    
    # ========================================================================
    # Maintenance
    # ========================================================================
    
    def find_stale(self, days: int = 90) -> list[Document]:
        """Find documents not accessed in the given number of days."""
        from datetime import datetime, timedelta
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        rows = self.db.conn.execute(
            "SELECT * FROM documents WHERE last_accessed < ? AND status = 'active'",
            (cutoff,)
        ).fetchall()
        return [self.db._row_to_document(row) for row in rows]
    
    def find_unused(self) -> list[Chunk]:
        """Find chunks that have never been used in a trace."""
        rows = self.db.conn.execute("""
            SELECT c.* FROM chunks c
            LEFT JOIN usage_traces ut ON ut.chunk_ids LIKE '%' || c.id || '%'
            WHERE ut.id IS NULL
        """).fetchall()
        return [self.db._row_to_chunk(row) for row in rows]
    
    def archive_document(self, doc_id: str):
        """Mark a document as archived."""
        self.db.conn.execute(
            "UPDATE documents SET status = 'archived' WHERE id = ?",
            (doc_id,)
        )
        self.db.conn.commit()

    # ========================================================================
    # Embeddings
    # ========================================================================

    @property
    def embeddings_enabled(self) -> bool:
        """Check if embeddings are available."""
        return self._embedder is not None

    def embed_missing(self) -> int:
        """Embed any chunks that don't have embeddings yet."""
        if not self._embedder:
            return 0
        return self._embedder.embed_missing()

    def semantic_search(self, query: str, limit: int = 10) -> list[tuple[Chunk, float]]:
        """
        Direct semantic search (bypasses multi-strategy retrieval).

        Returns list of (chunk, similarity_score) tuples.
        """
        if not self._embedder:
            raise RuntimeError("Embeddings not enabled. Install sentence-transformers.")
        results = self._embedder.find_similar(query, top_k=limit)
        return [(self.db.get_chunk(cid), score) for cid, score in results]

    def close(self):
        """Close database connection."""
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================================
# Quick usage functions
# ============================================================================

def demo():
    """Run a quick demo of the harness."""
    print("Initializing harness...")
    harness = KnowledgeHarness(":memory:")

    print(f"Embeddings enabled: {harness.embeddings_enabled}")

    # Ingest some content
    print("\nIngesting sample content...")

    harness.ingest_text("""
# Introduction to Retrieval-Augmented Generation

RAG combines retrieval systems with generative models. Instead of relying
solely on parametric knowledge, RAG systems fetch relevant documents and
use them to ground their responses.

## Key Components

1. **Retriever**: Finds relevant documents from a corpus
2. **Generator**: Produces output conditioned on retrieved context
3. **Knowledge Base**: The corpus of documents to search

## Advantages

- Reduces hallucination by grounding in sources
- Allows easy knowledge updates without retraining
- Provides citations and transparency
""", title="RAG Introduction", source="notes")

    harness.ingest_text("""
# Vector Embeddings

Vector embeddings are dense numerical representations of data.
They map items (words, sentences, images) to points in high-dimensional space
where geometric proximity reflects semantic similarity.

## Popular Models

- Word2Vec, GloVe for word embeddings
- BERT, Sentence-BERT for sentence embeddings
- CLIP for multimodal embeddings

## Applications

- Semantic search
- Clustering and classification
- Recommendation systems
""", title="Vector Embeddings", source="notes")

    print(f"\nStats: {harness.stats()}")

    # Multi-strategy search
    print("\n" + "="*60)
    print("Multi-strategy search for 'semantic search retrieval'...")
    print("="*60)
    results = harness.search("semantic search retrieval", limit=3)
    print(harness.search_formatted("semantic search retrieval", limit=3, verbose=True))

    # Direct semantic search (if embeddings enabled)
    if harness.embeddings_enabled:
        print("\n" + "="*60)
        print("Direct semantic search for 'how to find similar documents'...")
        print("="*60)
        sem_results = harness.semantic_search("how to find similar documents", limit=3)
        for chunk, score in sem_results:
            preview = chunk.content[:100].replace('\n', ' ')
            print(f"[{score:.3f}] {preview}...")

    # Record usage
    if results:
        harness.record_win(
            [results[0].chunk.id],
            "Learning about retrieval systems",
            notes="Good overview of RAG"
        )
        print(f"\nRecorded usage. Chunk stats: {harness.chunk_info(results[0].chunk.id)['usage']}")

    harness.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    demo()
