"""
Knowledge Harness - Consolidation

This is where episodic memory becomes semantic memory.

Individual usage traces are like remembering "that time I used this and it worked."
Functional profiles are like knowing "this source is good for practical debugging."

Consolidation runs periodically (not every session) and:
1. Generates/updates functional profiles for chunks with enough traces
2. Generates/updates functional profiles for sources
3. Updates concept profiles and identifies gaps
4. Discovers co-retrieval patterns (chunks that succeed together)
5. Surfaces contradictions, stale content, and unused content

All prose generation is LLM-powered via callbacks.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable
from collections import defaultdict
import json

from schema import (
    KnowledgeDB, Chunk, Document, Concept, SourceProfile, UsageTrace, Link,
    TaskType, UsageOutcome, EntityType, LinkCreator
)


# ============================================================================
# Prompt Templates
# ============================================================================

CHUNK_PROFILE_PROMPT = """You are analyzing a chunk of content to understand what it's good for and what it's not good for.

CHUNK CONTENT:
{content}

CHUNK SUMMARY:
{summary}

SOURCE: {source}
SOURCE TYPE: {source_type}

USAGE HISTORY:
{usage_history}

Based on this usage history, write a brief functional profile (2-4 sentences) that captures:
- What tasks/contexts this chunk is GOOD for
- What tasks/contexts this chunk is NOT good for or can be misleading
- Any caveats about how to use it (e.g., "verify before adopting", "anecdotal not authoritative")

Write in a direct, practical tone. This will be shown to help decide whether to use this chunk for future tasks.

FUNCTIONAL PROFILE:"""


SOURCE_PROFILE_PROMPT = """You are analyzing a source to understand its strengths and weaknesses across different task types.

SOURCE DOMAIN: {domain}
SOURCE TYPE: {source_type}

AGGREGATED USAGE STATISTICS:
{usage_stats}

SAMPLE SUCCESSFUL USES:
{sample_wins}

SAMPLE FAILURES/MISLEADING USES:
{sample_failures}

Based on this data, write a brief functional profile (2-4 sentences) that captures:
- What this source is GOOD for (which task types, what kind of information)
- What this source is NOT good for or where to be careful
- Any general caveats about using content from this source

Also list:
- STRENGTHS: 2-4 short phrases (e.g., "practical implementation advice", "community sentiment")
- WEAKNESSES: 2-4 short phrases (e.g., "factual accuracy", "outdated information")

FORMAT:
PROFILE: [your prose profile]
STRENGTHS: [comma-separated list]
WEAKNESSES: [comma-separated list]"""


CONCEPT_PROFILE_PROMPT = """You are analyzing our knowledge coverage for a concept.

CONCEPT: {name}
DESCRIPTION: {description}

CHUNKS COVERING THIS CONCEPT: {chunk_count}

USAGE STATISTICS FOR THESE CHUNKS:
{usage_stats}

SAMPLE CONTENT SUMMARIES:
{sample_summaries}

TASK TYPES WHERE WE'VE USED THIS KNOWLEDGE:
{task_types_used}

Based on this, write:
1. A FUNCTIONAL PROFILE (2-3 sentences): What aspects of this concept do we cover well? What angles are we strong on?
2. GAP NOTES (1-2 sentences): What's missing? What aspects aren't covered or are weak?

FORMAT:
PROFILE: [prose assessment of our coverage]
GAPS: [what's missing or weak]"""


CO_RETRIEVAL_ANALYSIS_PROMPT = """You are analyzing chunks that are frequently retrieved and used successfully together.

CHUNK A:
{chunk_a_summary}
Source: {chunk_a_source}

CHUNK B:
{chunk_b_summary}
Source: {chunk_b_source}

These chunks have been used together successfully {co_success_count} times in these contexts:
{contexts}

Why might these chunks complement each other? Write a brief explanation (1-2 sentences) of how they work together.

RELATIONSHIP:"""


# ============================================================================
# Consolidation Results
# ============================================================================

@dataclass
class ConsolidationReport:
    """Results from a consolidation run."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Profiles generated/updated
    chunks_profiled: int = 0
    sources_profiled: int = 0
    concepts_profiled: int = 0
    
    # Links created
    complement_links_created: int = 0
    
    # Issues surfaced
    contradictions: list = field(default_factory=list)
    stale_content: list = field(default_factory=list)
    unused_content: list = field(default_factory=list)
    gaps_identified: list = field(default_factory=list)
    
    # Traces archived
    traces_archived: int = 0
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "chunks_profiled": self.chunks_profiled,
            "sources_profiled": self.sources_profiled,
            "concepts_profiled": self.concepts_profiled,
            "complement_links_created": self.complement_links_created,
            "contradictions": self.contradictions,
            "stale_content": [(d.id, d.title) for d in self.stale_content],
            "unused_content": [(c.id, c.content[:50]) for c in self.unused_content],
            "gaps_identified": self.gaps_identified,
            "traces_archived": self.traces_archived
        }
    
    def summary(self) -> str:
        lines = [
            f"=== Consolidation Report ({self.timestamp.strftime('%Y-%m-%d %H:%M')}) ===",
            f"Profiles updated: {self.chunks_profiled} chunks, {self.sources_profiled} sources, {self.concepts_profiled} concepts",
            f"Complement links created: {self.complement_links_created}",
            f"Traces archived: {self.traces_archived}",
        ]
        if self.contradictions:
            lines.append(f"Contradictions found: {len(self.contradictions)}")
        if self.stale_content:
            lines.append(f"Stale documents: {len(self.stale_content)}")
        if self.unused_content:
            lines.append(f"Unused chunks: {len(self.unused_content)}")
        if self.gaps_identified:
            lines.append(f"Knowledge gaps: {len(self.gaps_identified)}")
        return "\n".join(lines)


# ============================================================================
# Consolidator
# ============================================================================

@dataclass
class ConsolidationConfig:
    """Configuration for consolidation behavior."""
    # Minimum traces before generating a profile
    min_traces_for_chunk_profile: int = 5
    min_traces_for_source_profile: int = 10
    min_chunks_for_concept_profile: int = 3
    
    # Co-retrieval detection
    min_co_retrievals_for_link: int = 3
    min_co_success_rate: float = 0.7
    
    # Stale content thresholds
    stale_days: int = 90
    
    # Archive traces after consolidation
    archive_traces_after_profile: bool = False  # Keep traces by default for now


class Consolidator:
    """
    Runs the consolidation process.
    
    LLM-powered operations are provided as callbacks.
    """
    
    def __init__(
        self,
        db: KnowledgeDB,
        config: ConsolidationConfig = None,
        # LLM callbacks
        generate_chunk_profile: Callable[[str], str] = None,
        generate_source_profile: Callable[[str], dict] = None,  # Returns {"profile": str, "strengths": list, "weaknesses": list}
        generate_concept_profile: Callable[[str], dict] = None,  # Returns {"profile": str, "gaps": str}
        analyze_co_retrieval: Callable[[str], str] = None,
    ):
        self.db = db
        self.config = config or ConsolidationConfig()
        
        # LLM callbacks (if None, profiles won't be generated but stats will be updated)
        self._generate_chunk_profile = generate_chunk_profile
        self._generate_source_profile = generate_source_profile
        self._generate_concept_profile = generate_concept_profile
        self._analyze_co_retrieval = analyze_co_retrieval
    
    def run(self) -> ConsolidationReport:
        """Run full consolidation pass."""
        report = ConsolidationReport()
        
        # 1. Profile chunks with enough traces
        report.chunks_profiled = self._consolidate_chunks()
        
        # 2. Profile sources
        report.sources_profiled = self._consolidate_sources()
        
        # 3. Profile concepts and identify gaps
        concepts_result = self._consolidate_concepts()
        report.concepts_profiled = concepts_result["profiled"]
        report.gaps_identified = concepts_result["gaps"]
        
        # 4. Discover co-retrieval patterns
        report.complement_links_created = self._discover_co_retrieval_patterns()
        
        # 5. Surface issues
        report.stale_content = self._find_stale_content()
        report.unused_content = self._find_unused_content()
        report.contradictions = self._find_contradictions()
        
        return report
    
    def _consolidate_chunks(self) -> int:
        """Generate functional profiles for chunks with enough usage data."""
        count = 0
        
        # Find chunks with enough traces
        rows = self.db.conn.execute("""
            SELECT c.id, c.content, c.summary, c.document_id, c.functional_profile,
                   d.source, d.title
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.functional_profile IS NULL
        """).fetchall()
        
        for row in rows:
            chunk_id = row["id"]
            traces = self.db.get_usage_traces_for_chunk(chunk_id)
            
            if len(traces) < self.config.min_traces_for_chunk_profile:
                continue
            
            # Format usage history for prompt
            usage_history = self._format_traces_for_prompt(traces)
            
            # Get source info
            doc = self.db.get_document(row["document_id"])
            source_profile = None
            if doc.source_profile_id:
                source_profile = self.db.conn.execute(
                    "SELECT * FROM source_profiles WHERE id = ?",
                    (doc.source_profile_id,)
                ).fetchone()
            
            source_type = source_profile["source_type"] if source_profile else "unknown"
            
            if self._generate_chunk_profile:
                prompt = CHUNK_PROFILE_PROMPT.format(
                    content=row["content"][:1500],  # Limit content length
                    summary=row["summary"] or "No summary",
                    source=row["source"],
                    source_type=source_type,
                    usage_history=usage_history
                )
                
                profile = self._generate_chunk_profile(prompt)
                if profile:
                    self.db.update_chunk_functional_profile(chunk_id, profile.strip())
                    count += 1
            else:
                # No LLM callback - generate a basic stats-based profile
                stats = self.db.get_chunk_success_rate_by_task(chunk_id)
                profile = self._generate_stats_based_profile(stats)
                self.db.update_chunk_functional_profile(chunk_id, profile)
                count += 1
        
        return count
    
    def _consolidate_sources(self) -> int:
        """Generate functional profiles for sources with enough usage data."""
        count = 0
        
        # Get all source profiles
        rows = self.db.conn.execute("SELECT * FROM source_profiles").fetchall()
        
        for row in rows:
            profile = self.db._row_to_source_profile(row)
            
            # Aggregate traces across all chunks from this source
            trace_data = self._aggregate_source_traces(profile.id)
            
            if trace_data["total"] < self.config.min_traces_for_source_profile:
                continue
            
            if self._generate_source_profile:
                prompt = SOURCE_PROFILE_PROMPT.format(
                    domain=profile.domain,
                    source_type=profile.source_type.value,
                    usage_stats=json.dumps(trace_data["by_task"], indent=2),
                    sample_wins="\n".join(trace_data["sample_wins"][:5]),
                    sample_failures="\n".join(trace_data["sample_failures"][:5])
                )
                
                result = self._generate_source_profile(prompt)
                if result:
                    profile.functional_profile = result.get("profile", "")
                    profile.strengths = result.get("strengths", [])
                    profile.weaknesses = result.get("weaknesses", [])
                    profile.trace_counts_by_task = trace_data["by_task"]
                    self.db.update_source_profile(profile)
                    count += 1
            else:
                # Stats-only update
                profile.trace_counts_by_task = trace_data["by_task"]
                self.db.update_source_profile(profile)
                count += 1
        
        return count
    
    def _consolidate_concepts(self) -> dict:
        """Generate functional profiles for concepts and identify gaps."""
        profiled = 0
        gaps = []
        
        concepts = self.db.get_all_concepts()
        
        for concept in concepts:
            # Get chunks for this concept
            chunks_with_weights = self.db.get_chunks_for_concept(concept.id)
            
            if len(chunks_with_weights) < self.config.min_chunks_for_concept_profile:
                continue
            
            # Aggregate usage stats across chunks
            all_traces = []
            task_types_used = set()
            chunk_summaries = []
            
            for chunk, weight in chunks_with_weights:
                traces = self.db.get_usage_traces_for_chunk(chunk.id)
                all_traces.extend(traces)
                for t in traces:
                    task_types_used.add(t.task_type.value)
                if chunk.summary:
                    chunk_summaries.append(chunk.summary)
            
            if not all_traces:
                continue
            
            usage_stats = self._aggregate_traces_by_task(all_traces)
            
            if self._generate_concept_profile:
                prompt = CONCEPT_PROFILE_PROMPT.format(
                    name=concept.name,
                    description=concept.description or "No description",
                    chunk_count=len(chunks_with_weights),
                    usage_stats=json.dumps(usage_stats, indent=2),
                    sample_summaries="\n".join(chunk_summaries[:5]),
                    task_types_used=", ".join(task_types_used)
                )
                
                result = self._generate_concept_profile(prompt)
                if result:
                    self.db.update_concept_functional_profile(
                        concept.id,
                        result.get("profile", ""),
                        result.get("gaps", "")
                    )
                    profiled += 1
                    
                    if result.get("gaps"):
                        gaps.append({
                            "concept": concept.name,
                            "gap": result["gaps"]
                        })
        
        return {"profiled": profiled, "gaps": gaps}
    
    def _discover_co_retrieval_patterns(self) -> int:
        """Find chunks that are frequently used successfully together."""
        links_created = 0
        
        # Get all sessions with multiple successful chunk uses
        rows = self.db.conn.execute("""
            SELECT session_id, chunk_ids, context_summary
            FROM usage_traces
            WHERE outcome IN ('win', 'partial')
        """).fetchall()
        
        # Count co-occurrences
        co_occurrences = defaultdict(lambda: {"count": 0, "contexts": []})
        
        for row in rows:
            chunk_ids = json.loads(row["chunk_ids"])
            if len(chunk_ids) < 2:
                continue
            
            # Count all pairs
            for i, id1 in enumerate(chunk_ids):
                for id2 in chunk_ids[i+1:]:
                    pair = tuple(sorted([id1, id2]))
                    co_occurrences[pair]["count"] += 1
                    co_occurrences[pair]["contexts"].append(row["context_summary"])
        
        # Create links for frequent co-occurrences
        for (id1, id2), data in co_occurrences.items():
            if data["count"] < self.config.min_co_retrievals_for_link:
                continue
            
            # Check if link already exists
            existing = self.db.conn.execute("""
                SELECT id FROM links
                WHERE source_type = 'chunk' AND source_id = ?
                AND target_type = 'chunk' AND target_id = ?
                AND relation = 'complements'
            """, (id1, id2)).fetchone()
            
            if existing:
                continue
            
            # Create the link
            link = Link.create(
                source_type=EntityType.CHUNK,
                source_id=id1,
                target_type=EntityType.CHUNK,
                target_id=id2,
                relation="complements",
                weight=min(1.0, data["count"] / 10),  # Cap at 1.0
                created_by=LinkCreator.AUTO
            )
            link.created_by = LinkCreator(LinkCreator.AUTO.value.replace("auto", "consolidated"))
            self.db.insert_link(link)
            links_created += 1
        
        return links_created
    
    def _find_stale_content(self) -> list[Document]:
        """Find documents not accessed in a long time."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.config.stale_days)).isoformat()
        
        rows = self.db.conn.execute("""
            SELECT * FROM documents
            WHERE last_accessed < ? AND status = 'active'
        """, (cutoff,)).fetchall()
        
        return [self.db._row_to_document(row) for row in rows]
    
    def _find_unused_content(self) -> list[Chunk]:
        """Find chunks that have never been used in a trace."""
        rows = self.db.conn.execute("""
            SELECT c.* FROM chunks c
            WHERE NOT EXISTS (
                SELECT 1 FROM usage_traces ut
                WHERE ut.chunk_ids LIKE '%' || c.id || '%'
            )
        """).fetchall()
        
        return [self.db._row_to_chunk(row) for row in rows]
    
    def _find_contradictions(self) -> list[dict]:
        """Find potential contradictions in the knowledge base."""
        # Look for chunks with 'supports' links to different claims
        # that might contradict each other
        # This is a simplified version - real implementation would need
        # more sophisticated analysis
        
        contradictions = []
        
        # Find concepts with conflicting chunk outcomes
        concepts = self.db.get_all_concepts()
        
        for concept in concepts:
            chunks_with_weights = self.db.get_chunks_for_concept(concept.id)
            
            # Look for chunks where one has high misleading rate
            # and another has high success rate
            chunk_stats = []
            for chunk, _ in chunks_with_weights:
                stats = self.db.get_chunk_success_rate(chunk.id)
                if stats["total"] >= 3:
                    chunk_stats.append((chunk, stats))
            
            for i, (chunk1, stats1) in enumerate(chunk_stats):
                for chunk2, stats2 in chunk_stats[i+1:]:
                    # If one is mostly wins and other is mostly misleading
                    if (stats1.get("success_rate", 0.5) > 0.8 and 
                        stats2.get("success_rate", 0.5) < 0.3):
                        contradictions.append({
                            "concept": concept.name,
                            "chunk_high": chunk1.id,
                            "chunk_low": chunk2.id,
                            "note": "Conflicting success rates suggest potential contradiction"
                        })
        
        return contradictions
    
    # ========================================================================
    # Helper methods
    # ========================================================================
    
    def _format_traces_for_prompt(self, traces: list[UsageTrace]) -> str:
        """Format traces for inclusion in a prompt."""
        lines = []
        
        # Group by task type
        by_task = defaultdict(list)
        for t in traces:
            by_task[t.task_type.value].append(t)
        
        for task_type, task_traces in by_task.items():
            outcomes = defaultdict(int)
            notes = []
            for t in task_traces:
                outcomes[t.outcome.value] += 1
                if t.notes:
                    notes.append(f"  - [{t.outcome.value}] {t.notes}")
            
            outcome_str = ", ".join(f"{v} {k}" for k, v in outcomes.items())
            lines.append(f"\n{task_type}: {outcome_str}")
            for note in notes[:3]:  # Limit notes per task type
                lines.append(note)
        
        return "\n".join(lines)
    
    def _generate_stats_based_profile(self, stats: dict) -> str:
        """Generate a basic profile from stats when no LLM is available."""
        if not stats.get("by_task"):
            return "Insufficient usage data."
        
        lines = []
        for task_type, task_stats in stats["by_task"].items():
            rate = task_stats.get("success_rate", 0.5)
            total = task_stats.get("total", 0)
            if rate >= 0.7:
                lines.append(f"Good for {task_type} ({rate:.0%} success, {total} uses)")
            elif rate <= 0.3:
                lines.append(f"Not recommended for {task_type} ({rate:.0%} success)")
        
        return ". ".join(lines) if lines else "Mixed results across task types."
    
    def _aggregate_source_traces(self, source_profile_id: str) -> dict:
        """Aggregate all traces for chunks from a source."""
        # Get all documents for this source
        doc_rows = self.db.conn.execute(
            "SELECT id FROM documents WHERE source_profile_id = ?",
            (source_profile_id,)
        ).fetchall()
        
        doc_ids = [row["id"] for row in doc_rows]
        if not doc_ids:
            return {"total": 0, "by_task": {}, "sample_wins": [], "sample_failures": []}
        
        # Get all chunks for these documents
        placeholders = ",".join("?" * len(doc_ids))
        chunk_rows = self.db.conn.execute(
            f"SELECT id FROM chunks WHERE document_id IN ({placeholders})",
            doc_ids
        ).fetchall()
        
        chunk_ids = [row["id"] for row in chunk_rows]
        
        # Get all traces for these chunks
        all_traces = []
        for chunk_id in chunk_ids:
            traces = self.db.get_usage_traces_for_chunk(chunk_id)
            all_traces.extend(traces)
        
        by_task = self._aggregate_traces_by_task(all_traces)
        
        # Sample wins and failures
        sample_wins = []
        sample_failures = []
        for t in all_traces:
            if t.outcome == UsageOutcome.WIN and t.notes:
                sample_wins.append(f"[{t.task_type.value}] {t.notes}")
            elif t.outcome in (UsageOutcome.MISS, UsageOutcome.MISLEADING) and t.notes:
                sample_failures.append(f"[{t.task_type.value}] {t.notes}")
        
        return {
            "total": len(all_traces),
            "by_task": by_task,
            "sample_wins": sample_wins,
            "sample_failures": sample_failures
        }
    
    def _aggregate_traces_by_task(self, traces: list[UsageTrace]) -> dict:
        """Aggregate traces into stats by task type."""
        by_task = defaultdict(lambda: {"win": 0, "partial": 0, "miss": 0, "misleading": 0})
        
        for t in traces:
            by_task[t.task_type.value][t.outcome.value] += 1
        
        # Calculate success rates
        for task_type, counts in by_task.items():
            total = sum(counts.values())
            success = counts["win"] + 0.5 * counts["partial"]
            penalty = counts["miss"] + 2 * counts["misleading"]
            by_task[task_type]["total"] = total
            by_task[task_type]["success_rate"] = success / (success + penalty) if (success + penalty) > 0 else 0.5
        
        return dict(by_task)


# ============================================================================
# Convenience functions
# ============================================================================

def run_consolidation(db: KnowledgeDB, config: ConsolidationConfig = None) -> ConsolidationReport:
    """Run consolidation without LLM (stats-only profiles)."""
    consolidator = Consolidator(db, config)
    return consolidator.run()


def run_consolidation_with_llm(
    db: KnowledgeDB,
    llm_call: Callable[[str], str],  # Generic LLM call function
    config: ConsolidationConfig = None
) -> ConsolidationReport:
    """
    Run consolidation with LLM-generated profiles.
    
    The llm_call function should take a prompt string and return a response string.
    """
    
    def generate_chunk_profile(prompt: str) -> str:
        return llm_call(prompt)
    
    def generate_source_profile(prompt: str) -> dict:
        response = llm_call(prompt)
        # Parse the structured response
        result = {"profile": "", "strengths": [], "weaknesses": []}
        
        lines = response.split("\n")
        for line in lines:
            if line.startswith("PROFILE:"):
                result["profile"] = line[8:].strip()
            elif line.startswith("STRENGTHS:"):
                result["strengths"] = [s.strip() for s in line[10:].split(",")]
            elif line.startswith("WEAKNESSES:"):
                result["weaknesses"] = [s.strip() for s in line[11:].split(",")]
        
        # If structured parsing failed, use the whole response as profile
        if not result["profile"]:
            result["profile"] = response.strip()
        
        return result
    
    def generate_concept_profile(prompt: str) -> dict:
        response = llm_call(prompt)
        result = {"profile": "", "gaps": ""}
        
        lines = response.split("\n")
        for line in lines:
            if line.startswith("PROFILE:"):
                result["profile"] = line[8:].strip()
            elif line.startswith("GAPS:"):
                result["gaps"] = line[5:].strip()
        
        if not result["profile"]:
            result["profile"] = response.strip()
        
        return result
    
    consolidator = Consolidator(
        db, config,
        generate_chunk_profile=generate_chunk_profile,
        generate_source_profile=generate_source_profile,
        generate_concept_profile=generate_concept_profile
    )
    
    return consolidator.run()


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    from schema import init_db, Document, Chunk, ContentType, ChunkType, SourceProfile, SourceType
    from record import record_usage
    from datetime import timezone
    
    db = init_db(":memory:")
    
    # Create a source profile
    source = SourceProfile.create("reddit.com", SourceType.FORUM)
    db.insert_source_profile(source)
    
    # Create documents and chunks
    doc = Document.create(
        source="https://reddit.com/r/python/comments/abc",
        content_type=ContentType.ARTICLE,
        title="Async Python Tips",
        raw_content="Tips for async..."
    )
    doc.source_profile_id = source.id
    db.insert_document(doc)
    
    chunk = Chunk.create(
        document_id=doc.id,
        content="Use asyncio.create_task() to avoid deadlocks in callbacks.",
        position=0,
        summary="Async callback deadlock fix"
    )
    db.insert_chunk(chunk)
    
    # Add many usage traces to trigger consolidation
    for i in range(6):
        record_usage(db, [chunk.id], f"Debugging async issue #{i}", "win",
                    task_type="debugging", notes=f"Fixed callback issue #{i}")
    
    record_usage(db, [chunk.id], "Understanding event loop", "miss",
                task_type="conceptual_understanding", notes="Too practical")
    record_usage(db, [chunk.id], "Explaining async to colleague", "partial",
                task_type="conceptual_understanding", notes="Good example but needed more theory")
    
    # Run consolidation (without LLM - stats only)
    print("Running consolidation...")
    report = run_consolidation(db)
    
    print(report.summary())
    print()
    
    # Check the generated profile
    updated_chunk = db.get_chunk(chunk.id)
    print(f"Chunk functional profile: {updated_chunk.functional_profile}")
    
    # Check for unused content
    print(f"\nUnused chunks: {len(report.unused_content)}")
    
    db.close()
