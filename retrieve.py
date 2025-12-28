"""
Knowledge Harness - Retrieval System

Multi-strategy retrieval with ranking and fusion.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum

from schema import (
    KnowledgeDB, Document, Chunk, Concept, UsageTrace,
    EntityType, UsageOutcome
)


class RetrievalStrategy(Enum):
    KEYWORD = "keyword"           # Simple text matching
    SEMANTIC = "semantic"         # Embedding similarity (requires external embeddings)
    CONCEPT = "concept"           # Via shared concepts
    USAGE = "usage"              # Based on past usage patterns
    GRAPH = "graph"              # Graph traversal from seed nodes
    RECENCY = "recency"          # Time-weighted


@dataclass
class RetrievalResult:
    """A single retrieval result with provenance."""
    chunk: Chunk
    document: Document
    score: float
    strategies: list[str]           # Which strategies found this
    strategy_scores: dict           # Score breakdown by strategy
    usage_stats: Optional[dict]     # Success rate info if available
    related_concepts: list[str]     # Concept names linked to this chunk
    context_snippets: list[str]     # Relevant usage contexts from traces


@dataclass
class RetrievalConfig:
    """Tunable parameters for retrieval."""
    # Strategy weights (should sum to ~1.0 for interpretability)
    weight_keyword: float = 0.2
    weight_semantic: float = 0.3
    weight_concept: float = 0.2
    weight_usage: float = 0.2
    weight_recency: float = 0.1
    
    # Recency decay (half-life in days)
    recency_halflife_days: float = 30.0
    
    # Minimum scores to include
    min_keyword_score: float = 0.1
    min_concept_overlap: int = 1
    
    # Usage success rate thresholds
    usage_boost_threshold: float = 0.7    # Boost chunks above this success rate
    usage_penalty_threshold: float = 0.3  # Penalize chunks below this


class Retriever:
    """
    Multi-strategy retrieval engine.

    For semantic retrieval, you'll need to provide an embedding function.
    Without it, the system falls back to keyword + concept + usage strategies.
    """

    def __init__(
        self,
        db: KnowledgeDB,
        config: RetrievalConfig = None,
        embed_fn: callable = None,  # (text) -> list[float]
        similarity_fn: callable = None,  # (vec1, vec2) -> float
        use_stored_embeddings: bool = True,  # Use embeddings from DB
    ):
        self.db = db
        self.config = config or RetrievalConfig()
        self.embed_fn = embed_fn
        self.similarity_fn = similarity_fn or self._cosine_similarity
        self.use_stored_embeddings = use_stored_embeddings

        # Cache for embeddings (chunk_id -> embedding) - used when not using stored
        self._embedding_cache: dict[str, list[float]] = {}
    
    def retrieve(
        self,
        query: str,
        strategies: list[RetrievalStrategy] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """
        Main retrieval entry point.
        
        Runs specified strategies (or all applicable ones), fuses results,
        and returns ranked list.
        """
        if strategies is None:
            # Use all strategies that we can actually run
            strategies = [RetrievalStrategy.KEYWORD, RetrievalStrategy.CONCEPT, 
                         RetrievalStrategy.USAGE, RetrievalStrategy.RECENCY]
            if self.embed_fn:
                strategies.append(RetrievalStrategy.SEMANTIC)
        
        # Collect results from each strategy
        # Each strategy returns: {chunk_id: score}
        strategy_results: dict[str, dict[str, float]] = {}
        
        if RetrievalStrategy.KEYWORD in strategies:
            strategy_results["keyword"] = self._keyword_search(query)
        
        if RetrievalStrategy.SEMANTIC in strategies and self.embed_fn:
            strategy_results["semantic"] = self._semantic_search(query)
        
        if RetrievalStrategy.CONCEPT in strategies:
            strategy_results["concept"] = self._concept_search(query)
        
        if RetrievalStrategy.USAGE in strategies:
            strategy_results["usage"] = self._usage_search(query)
        
        if RetrievalStrategy.RECENCY in strategies:
            strategy_results["recency"] = self._recency_scores()
        
        # Fuse results
        fused = self._fuse_results(strategy_results)
        
        # Filter and limit
        fused = {k: v for k, v in fused.items() if v["final_score"] >= min_score}
        sorted_ids = sorted(fused.keys(), key=lambda x: fused[x]["final_score"], reverse=True)[:limit]
        
        # Build full result objects
        results = []
        for chunk_id in sorted_ids:
            chunk = self.db.get_chunk(chunk_id)
            if not chunk:
                continue
            
            doc = self.db.get_document(chunk.document_id)
            usage_stats = self.db.get_chunk_success_rate(chunk_id)
            concepts = [c.name for c, _ in self.db.get_concepts_for_chunk(chunk_id)]
            
            # Get relevant usage contexts
            traces = self.db.get_usage_traces_for_chunk(chunk_id)
            context_snippets = [
                t.context_summary for t in traces 
                if t.outcome in (UsageOutcome.WIN, UsageOutcome.PARTIAL)
            ][:3]  # Limit to 3 most relevant
            
            results.append(RetrievalResult(
                chunk=chunk,
                document=doc,
                score=fused[chunk_id]["final_score"],
                strategies=fused[chunk_id]["strategies"],
                strategy_scores=fused[chunk_id]["scores"],
                usage_stats=usage_stats,
                related_concepts=concepts,
                context_snippets=context_snippets
            ))
        
        return results
    
    def _keyword_search(self, query: str) -> dict[str, float]:
        """Simple keyword/term matching."""
        results = {}
        
        # Tokenize query
        query_terms = set(re.findall(r'\w+', query.lower()))
        if not query_terms:
            return results
        
        # Search all chunks (in practice, you'd want an index)
        all_chunks = self.db.conn.execute("SELECT id, content, summary FROM chunks").fetchall()
        
        for row in all_chunks:
            chunk_id = row["id"]
            text = (row["content"] + " " + (row["summary"] or "")).lower()
            chunk_terms = set(re.findall(r'\w+', text))
            
            # Jaccard-ish overlap
            overlap = len(query_terms & chunk_terms)
            if overlap > 0:
                score = overlap / len(query_terms)
                if score >= self.config.min_keyword_score:
                    results[chunk_id] = score
        
        return results
    
    def _semantic_search(self, query: str) -> dict[str, float]:
        """Embedding-based similarity search."""
        if not self.embed_fn:
            return {}

        results = {}
        query_embedding = self.embed_fn(query)

        if self.use_stored_embeddings:
            # Use pre-computed embeddings from DB
            all_embeddings = self.db.get_all_embeddings()

            for chunk_id, chunk_embedding in all_embeddings:
                similarity = self.similarity_fn(query_embedding, chunk_embedding)
                if similarity > 0:
                    results[chunk_id] = similarity
        else:
            # Compute embeddings on the fly (with in-memory cache)
            all_chunks = self.db.conn.execute("SELECT id, content FROM chunks").fetchall()

            for row in all_chunks:
                chunk_id = row["id"]

                # Get or compute embedding
                if chunk_id not in self._embedding_cache:
                    self._embedding_cache[chunk_id] = self.embed_fn(row["content"])

                chunk_embedding = self._embedding_cache[chunk_id]
                similarity = self.similarity_fn(query_embedding, chunk_embedding)

                if similarity > 0:
                    results[chunk_id] = similarity

        return results
    
    def _concept_search(self, query: str) -> dict[str, float]:
        """Find chunks that share concepts with the query."""
        results = {}
        
        # Extract concept-like terms from query (simplified: just use significant words)
        query_terms = set(re.findall(r'\b[a-z]{4,}\b', query.lower()))
        
        # Find matching concepts
        matching_concepts = []
        for term in query_terms:
            # Check if term matches any concept name or alias
            concept = self.db.get_concept_by_name(term)
            if concept:
                matching_concepts.append(concept)
            else:
                # Check aliases (simplified)
                rows = self.db.conn.execute(
                    "SELECT * FROM concepts WHERE aliases LIKE ?",
                    (f'%{term}%',)
                ).fetchall()
                for row in rows:
                    matching_concepts.append(self.db._row_to_concept(row))
        
        if not matching_concepts:
            return results
        
        # Get chunks linked to these concepts
        for concept in matching_concepts:
            chunks_with_weights = self.db.get_chunks_for_concept(concept.id)
            for chunk, weight in chunks_with_weights:
                if chunk.id in results:
                    results[chunk.id] += weight
                else:
                    results[chunk.id] = weight
        
        # Normalize
        if results:
            max_score = max(results.values())
            results = {k: v / max_score for k, v in results.items()}
        
        return results
    
    def _usage_search(self, query: str) -> dict[str, float]:
        """Find chunks that were useful in similar contexts."""
        results = {}
        
        # Get all usage traces
        traces = self.db.conn.execute(
            "SELECT * FROM usage_traces"
        ).fetchall()
        
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        for row in traces:
            trace = self.db._row_to_usage_trace(row)
            
            # Check if trace context is similar to query
            context_terms = set(re.findall(r'\w+', trace.context_summary.lower()))
            if trace.query:
                context_terms.update(re.findall(r'\w+', trace.query.lower()))
            
            overlap = len(query_terms & context_terms)
            if overlap < 2:  # Require some meaningful overlap
                continue
            
            context_similarity = overlap / max(len(query_terms), 1)
            
            # Weight by outcome
            outcome_weight = {
                UsageOutcome.WIN: 1.0,
                UsageOutcome.PARTIAL: 0.5,
                UsageOutcome.MISS: 0.0,
                UsageOutcome.MISLEADING: -0.5,
            }.get(trace.outcome, 0)
            
            score = context_similarity * outcome_weight
            
            # Apply to all chunks in this trace
            for chunk_id in trace.chunk_ids:
                if chunk_id in results:
                    results[chunk_id] = max(results[chunk_id], score)
                else:
                    results[chunk_id] = score
        
        # Remove negative scores and normalize
        results = {k: v for k, v in results.items() if v > 0}
        if results:
            max_score = max(results.values())
            if max_score > 0:
                results = {k: v / max_score for k, v in results.items()}
        
        return results
    
    def _recency_scores(self) -> dict[str, float]:
        """Score chunks by document recency."""
        results = {}
        now = datetime.utcnow()
        halflife = timedelta(days=self.config.recency_halflife_days)
        
        # Get all chunks with their document's last_accessed time
        rows = self.db.conn.execute("""
            SELECT c.id, d.last_accessed FROM chunks c
            JOIN documents d ON c.document_id = d.id
        """).fetchall()
        
        for row in rows:
            last_accessed = datetime.fromisoformat(row["last_accessed"])
            age = now - last_accessed
            
            # Exponential decay
            decay = 0.5 ** (age / halflife)
            results[row["id"]] = decay
        
        return results
    
    def _fuse_results(
        self, 
        strategy_results: dict[str, dict[str, float]]
    ) -> dict[str, dict]:
        """
        Combine results from multiple strategies into final rankings.
        
        Returns: {chunk_id: {"final_score": float, "strategies": list, "scores": dict}}
        """
        weights = {
            "keyword": self.config.weight_keyword,
            "semantic": self.config.weight_semantic,
            "concept": self.config.weight_concept,
            "usage": self.config.weight_usage,
            "recency": self.config.weight_recency,
        }
        
        # Collect all chunk IDs
        all_chunk_ids = set()
        for results in strategy_results.values():
            all_chunk_ids.update(results.keys())
        
        # Calculate fused scores
        fused = {}
        for chunk_id in all_chunk_ids:
            scores = {}
            strategies = []
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for strategy_name, results in strategy_results.items():
                if chunk_id in results:
                    score = results[chunk_id]
                    scores[strategy_name] = score
                    strategies.append(strategy_name)
                    weighted_sum += score * weights.get(strategy_name, 0.1)
                    weight_sum += weights.get(strategy_name, 0.1)
            
            # Normalize by weights actually used
            final_score = weighted_sum / weight_sum if weight_sum > 0 else 0
            
            # Boost/penalize based on historical usage success
            usage_stats = self.db.get_chunk_success_rate(chunk_id)
            if usage_stats["total"] >= 3:  # Only adjust if we have enough data
                rate = usage_stats["success_rate"]
                if rate >= self.config.usage_boost_threshold:
                    final_score *= 1.2  # 20% boost
                elif rate <= self.config.usage_penalty_threshold:
                    final_score *= 0.8  # 20% penalty
            
            fused[chunk_id] = {
                "final_score": final_score,
                "strategies": strategies,
                "scores": scores
            }
        
        return fused
    
    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)


# ============================================================================
# Graph-based retrieval
# ============================================================================

class GraphRetriever:
    """
    Retrieval via graph traversal.
    
    Given seed nodes (chunks, concepts, or documents), walk the link graph
    to find related content.
    """
    
    def __init__(self, db: KnowledgeDB):
        self.db = db
    
    def traverse_from_chunks(
        self,
        seed_chunk_ids: list[str],
        max_hops: int = 2,
        max_results: int = 20,
        relation_filter: list[str] = None,
    ) -> list[tuple[str, float, list[str]]]:
        """
        Find chunks reachable from seeds via graph links.
        
        Returns: [(chunk_id, score, path), ...]
        where score decays with hop distance and path shows how we got there.
        """
        visited = set(seed_chunk_ids)
        results = []  # (chunk_id, score, path)
        
        # BFS with decay
        frontier = [(cid, 1.0, [cid]) for cid in seed_chunk_ids]
        
        for hop in range(max_hops):
            next_frontier = []
            decay = 0.5 ** hop
            
            for chunk_id, score, path in frontier:
                # Get links from this chunk
                links = self.db.get_links_from(EntityType.CHUNK, chunk_id)
                
                for link in links:
                    if relation_filter and link.relation not in relation_filter:
                        continue
                    
                    if link.target_type == EntityType.CHUNK:
                        target_id = link.target_id
                        if target_id not in visited:
                            visited.add(target_id)
                            new_score = score * link.weight * decay
                            new_path = path + [f"--{link.relation}-->", target_id]
                            results.append((target_id, new_score, new_path))
                            next_frontier.append((target_id, new_score, new_path))
                    
                    elif link.target_type == EntityType.CONCEPT:
                        # Go through concept to find other chunks
                        concept_id = link.target_id
                        related_chunks = self.db.get_chunks_for_concept(concept_id)
                        for chunk, weight in related_chunks:
                            if chunk.id not in visited:
                                visited.add(chunk.id)
                                concept = self.db.conn.execute(
                                    "SELECT name FROM concepts WHERE id = ?",
                                    (concept_id,)
                                ).fetchone()
                                concept_name = concept["name"] if concept else "concept"
                                new_score = score * link.weight * weight * decay * 0.5
                                new_path = path + [f"--{link.relation}-->[{concept_name}]-->", chunk.id]
                                results.append((chunk.id, new_score, new_path))
                                next_frontier.append((chunk.id, new_score, new_path))
            
            frontier = next_frontier
        
        # Sort by score, limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def find_related_via_concepts(
        self,
        chunk_id: str,
        limit: int = 10
    ) -> list[tuple[Chunk, list[str]]]:
        """
        Find chunks that share concepts with the given chunk.
        
        Returns: [(chunk, shared_concepts), ...]
        """
        # Get concepts for this chunk
        chunk_concepts = self.db.get_concepts_for_chunk(chunk_id)
        if not chunk_concepts:
            return []
        
        concept_ids = {c.id for c, _ in chunk_concepts}
        concept_names = {c.id: c.name for c, _ in chunk_concepts}
        
        # Find other chunks with these concepts
        related = {}  # chunk_id -> (chunk, shared_concept_names)
        
        for concept, _ in chunk_concepts:
            chunks_with_concept = self.db.get_chunks_for_concept(concept.id)
            for chunk, weight in chunks_with_concept:
                if chunk.id == chunk_id:
                    continue
                if chunk.id not in related:
                    related[chunk.id] = (chunk, [])
                related[chunk.id][1].append(concept.name)
        
        # Sort by number of shared concepts
        sorted_related = sorted(
            related.values(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        return sorted_related[:limit]


# ============================================================================
# Convenience functions
# ============================================================================

def quick_search(db: KnowledgeDB, query: str, limit: int = 5) -> list[RetrievalResult]:
    """Simple search using keyword + recency strategies."""
    retriever = Retriever(db)
    return retriever.retrieve(
        query,
        strategies=[RetrievalStrategy.KEYWORD, RetrievalStrategy.RECENCY],
        limit=limit
    )


def search_with_usage(db: KnowledgeDB, query: str, limit: int = 10) -> list[RetrievalResult]:
    """Search using all non-semantic strategies."""
    retriever = Retriever(db)
    return retriever.retrieve(
        query,
        strategies=[
            RetrievalStrategy.KEYWORD,
            RetrievalStrategy.CONCEPT,
            RetrievalStrategy.USAGE,
            RetrievalStrategy.RECENCY
        ],
        limit=limit
    )


def format_results(results: list[RetrievalResult], verbose: bool = False) -> str:
    """Format retrieval results for display."""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"\n{'='*60}")
        lines.append(f"Result {i}: {r.document.title} (score: {r.score:.3f})")
        lines.append(f"Strategies: {', '.join(r.strategies)}")
        if r.related_concepts:
            lines.append(f"Concepts: {', '.join(r.related_concepts)}")
        if r.usage_stats and r.usage_stats["total"] > 0:
            lines.append(f"Usage: {r.usage_stats['total']} traces, {r.usage_stats['success_rate']:.1%} success")
        lines.append(f"\n{r.chunk.summary or r.chunk.content[:200]}...")
        
        if verbose and r.context_snippets:
            lines.append(f"\nPast useful contexts:")
            for ctx in r.context_snippets:
                lines.append(f"  - {ctx}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Demo
    from schema import init_db
    from ingest import quick_ingest
    
    db = init_db(":memory:")
    
    # Ingest some sample content
    quick_ingest(db, "ml_basics.md", """
# Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

## Supervised Learning
In supervised learning, models learn from labeled examples. Common algorithms include:
- Linear regression for continuous outputs
- Logistic regression for classification
- Decision trees and random forests
- Neural networks

## Unsupervised Learning  
Unsupervised learning finds patterns in unlabeled data. Key techniques:
- Clustering (k-means, hierarchical)
- Dimensionality reduction (PCA, t-SNE)
- Anomaly detection
""")
    
    quick_ingest(db, "vectors.md", """
# Vector Representations

Vectors are fundamental to modern ML. Embedding models convert data into dense vectors.

## Word Embeddings
Word2Vec and GloVe create word vectors where similar words are close together.
These embeddings capture semantic relationships.

## Sentence Embeddings
Models like BERT and sentence transformers create embeddings for entire sentences.
Useful for semantic search and similarity comparisons.
""")
    
    # Add a concept manually
    from schema import Concept, ConceptType
    ml_concept = Concept.create("machine learning", concept_type=ConceptType.TOPIC, 
                                description="Field of AI focused on learning from data")
    db.insert_concept(ml_concept)
    
    # Search
    print("Searching for 'learning from data'...")
    results = quick_search(db, "learning from data", limit=3)
    print(format_results(results))
    
    db.close()
