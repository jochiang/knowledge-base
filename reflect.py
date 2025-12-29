"""
Knowledge Harness - Reflection

Higher-order synthesis that looks at the knowledge base as a whole.

While consolidation updates individual profiles (chunk by chunk, source by source),
reflection asks systemic questions:

- Where are we strong vs weak?
- What patterns have emerged?
- What should we acquire more knowledge about?
- What's working and what isn't?

This runs occasionally (weekly? monthly?) and produces actionable insights.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable
from collections import defaultdict
import json

from schema import (
    KnowledgeDB, Document, Chunk, Concept, SourceProfile, UsageTrace,
    TaskType, UsageOutcome, ContentType, SourceType
)


# ============================================================================
# Reflection Prompts
# ============================================================================

KNOWLEDGE_GAPS_PROMPT = """You are analyzing a knowledge base to identify gaps and opportunities.

CONCEPTS WE COVER:
{concepts_list}

TASK TYPES AND SUCCESS RATES:
{task_type_performance}

FAILED QUERIES (searches that returned poor results):
{failed_queries}

FREQUENTLY ACCESSED BUT LOW SUCCESS:
{high_access_low_success}

Based on this analysis:
1. What knowledge GAPS should we prioritize filling?
2. What TYPES of content should we seek out?
3. What SOURCES might help fill these gaps?

Be specific and actionable."""


SYSTEM_HEALTH_PROMPT = """You are analyzing the health of a knowledge management system.

OVERALL STATISTICS:
{stats}

SOURCE PERFORMANCE:
{source_performance}

CONTENT AGE DISTRIBUTION:
{age_distribution}

USAGE PATTERNS:
{usage_patterns}

POTENTIAL ISSUES:
{issues}

Provide a brief health assessment:
1. What's working well?
2. What needs attention?
3. Recommended actions?"""


LEARNING_SUMMARY_PROMPT = """You are summarizing what has been learned about knowledge quality.

SOURCES RANKED BY RELIABILITY (by task type):
{source_rankings}

CHUNKS WITH STRONGEST FUNCTIONAL PROFILES:
{top_chunks}

CHUNKS THAT CONSISTENTLY MISLEAD:
{problematic_chunks}

EMERGING PATTERNS:
{patterns}

Write a brief "lessons learned" summary that captures:
1. Key insights about source quality
2. Patterns in what works vs what doesn't  
3. Recommendations for future knowledge acquisition"""


# ============================================================================
# Reflection Results
# ============================================================================

@dataclass
class KnowledgeGap:
    """A identified gap in the knowledge base."""
    area: str  # What's missing
    severity: str  # "critical", "moderate", "minor"
    evidence: str  # Why we think this is a gap
    suggestions: list[str]  # How to fill it


@dataclass
class SourceAssessment:
    """Assessment of a source's overall value."""
    domain: str
    overall_value: str  # "high", "medium", "low", "negative"
    best_for: list[str]  # Task types where it excels
    avoid_for: list[str]  # Task types where it fails
    notes: str


@dataclass
class ReflectionReport:
    """Complete reflection output."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # System health
    total_documents: int = 0
    total_chunks: int = 0
    total_traces: int = 0
    overall_success_rate: float = 0.0
    
    # Performance by dimension
    task_type_performance: dict = field(default_factory=dict)
    source_assessments: list = field(default_factory=list)
    content_type_performance: dict = field(default_factory=dict)
    
    # Gaps and opportunities
    knowledge_gaps: list = field(default_factory=list)
    underutilized_content: list = field(default_factory=list)
    
    # Problem areas
    consistently_misleading: list = field(default_factory=list)
    stale_but_accessed: list = field(default_factory=list)
    
    # Patterns
    successful_patterns: list = field(default_factory=list)
    co_retrieval_clusters: list = field(default_factory=list)
    
    # LLM-generated summaries (if available)
    health_summary: Optional[str] = None
    gaps_analysis: Optional[str] = None
    lessons_learned: Optional[str] = None
    
    def format_report(self) -> str:
        """Format as readable text report."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"KNOWLEDGE BASE REFLECTION - {self.timestamp.strftime('%Y-%m-%d')}")
        lines.append("=" * 70)
        
        # Health overview
        lines.append("\n## SYSTEM HEALTH\n")
        lines.append(f"Documents: {self.total_documents}")
        lines.append(f"Chunks: {self.total_chunks}")
        lines.append(f"Usage traces: {self.total_traces}")
        lines.append(f"Overall success rate: {self.overall_success_rate:.1%}")
        
        if self.health_summary:
            lines.append(f"\n{self.health_summary}")
        
        # Task type performance
        lines.append("\n## PERFORMANCE BY TASK TYPE\n")
        for task_type, stats in sorted(
            self.task_type_performance.items(),
            key=lambda x: x[1].get("success_rate", 0),
            reverse=True
        ):
            rate = stats.get("success_rate", 0)
            total = stats.get("total", 0)
            emoji = "✓" if rate >= 0.7 else "~" if rate >= 0.4 else "✗"
            lines.append(f"  {emoji} {task_type}: {rate:.0%} success ({total} uses)")
        
        # Source assessments
        if self.source_assessments:
            lines.append("\n## SOURCE ASSESSMENTS\n")
            for sa in self.source_assessments:
                lines.append(f"  [{sa.overall_value.upper()}] {sa.domain}")
                if sa.best_for:
                    lines.append(f"       Best for: {', '.join(sa.best_for)}")
                if sa.avoid_for:
                    lines.append(f"       Avoid for: {', '.join(sa.avoid_for)}")
                if sa.notes:
                    lines.append(f"       Note: {sa.notes}")
        
        # Knowledge gaps
        if self.knowledge_gaps:
            lines.append("\n## KNOWLEDGE GAPS\n")
            for gap in self.knowledge_gaps:
                severity_emoji = {"critical": "🔴", "moderate": "🟡", "minor": "🟢"}.get(gap.severity, "⚪")
                lines.append(f"  {severity_emoji} {gap.area}")
                lines.append(f"       Evidence: {gap.evidence}")
                if gap.suggestions:
                    lines.append(f"       Suggestions: {', '.join(gap.suggestions)}")
        
        if self.gaps_analysis:
            lines.append(f"\n{self.gaps_analysis}")
        
        # Problem areas
        if self.consistently_misleading:
            lines.append("\n## PROBLEMATIC CONTENT (Consider Removing)\n")
            for chunk_id, info in self.consistently_misleading[:5]:
                lines.append(f"  ⚠ {info['title']}: {info['misleading_count']} misleading uses")
                lines.append(f"       Preview: {info['preview']}")
        
        # Underutilized
        if self.underutilized_content:
            lines.append(f"\n## UNDERUTILIZED CONTENT\n")
            lines.append(f"  {len(self.underutilized_content)} chunks have never been used")
        
        # Successful patterns
        if self.successful_patterns:
            lines.append("\n## SUCCESSFUL PATTERNS\n")
            for pattern in self.successful_patterns[:5]:
                lines.append(f"  • {pattern}")
        
        # Lessons learned
        if self.lessons_learned:
            lines.append("\n## LESSONS LEARNED\n")
            lines.append(self.lessons_learned)
        
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


# ============================================================================
# Reflector
# ============================================================================

class Reflector:
    """
    Performs high-level analysis of the knowledge base.
    """
    
    def __init__(
        self,
        db: KnowledgeDB,
        # LLM callbacks for generating summaries
        analyze_gaps: Callable[[str], str] = None,
        analyze_health: Callable[[str], str] = None,
        summarize_lessons: Callable[[str], str] = None,
    ):
        self.db = db
        self._analyze_gaps = analyze_gaps
        self._analyze_health = analyze_health
        self._summarize_lessons = summarize_lessons
    
    def reflect(self) -> ReflectionReport:
        """Run complete reflection analysis."""
        report = ReflectionReport()
        
        # Basic stats
        stats = self.db.stats()
        report.total_documents = stats["documents"]
        report.total_chunks = stats["chunks"]
        report.total_traces = stats["usage_traces"]
        
        # Calculate overall success rate
        all_traces = self._get_all_traces()
        if all_traces:
            wins = sum(1 for t in all_traces if t.outcome == UsageOutcome.WIN)
            partials = sum(1 for t in all_traces if t.outcome == UsageOutcome.PARTIAL)
            report.overall_success_rate = (wins + 0.5 * partials) / len(all_traces)
        
        # Performance by task type
        report.task_type_performance = self._analyze_task_type_performance(all_traces)
        
        # Source assessments
        report.source_assessments = self._assess_sources()
        
        # Content type performance
        report.content_type_performance = self._analyze_content_type_performance()
        
        # Knowledge gaps
        report.knowledge_gaps = self._identify_gaps()
        
        # Underutilized content
        report.underutilized_content = self._find_underutilized()
        
        # Problematic content
        report.consistently_misleading = self._find_consistently_misleading()
        
        # Stale but still accessed
        report.stale_but_accessed = self._find_stale_but_accessed()
        
        # Successful patterns
        report.successful_patterns = self._identify_successful_patterns()
        
        # Co-retrieval clusters
        report.co_retrieval_clusters = self._find_co_retrieval_clusters()
        
        # LLM-generated summaries
        if self._analyze_health:
            report.health_summary = self._generate_health_summary(report)
        
        if self._analyze_gaps:
            report.gaps_analysis = self._generate_gaps_analysis(report)
        
        if self._summarize_lessons:
            report.lessons_learned = self._generate_lessons_learned(report)
        
        return report
    
    # ========================================================================
    # Analysis methods
    # ========================================================================
    
    def _get_all_traces(self) -> list[UsageTrace]:
        """Get all usage traces."""
        rows = self.db.conn.execute("SELECT * FROM usage_traces").fetchall()
        return [self.db._row_to_usage_trace(row) for row in rows]
    
    def _analyze_task_type_performance(self, traces: list[UsageTrace]) -> dict:
        """Analyze success rates by task type."""
        by_task = defaultdict(lambda: {"win": 0, "partial": 0, "miss": 0, "misleading": 0})
        
        for t in traces:
            by_task[t.task_type.value][t.outcome.value] += 1
        
        result = {}
        for task_type, counts in by_task.items():
            total = sum(counts.values())
            success = counts["win"] + 0.5 * counts["partial"]
            penalty = counts["miss"] + 2 * counts["misleading"]
            
            result[task_type] = {
                "total": total,
                "counts": dict(counts),
                "success_rate": success / (success + penalty) if (success + penalty) > 0 else 0.5
            }
        
        return result
    
    def _assess_sources(self) -> list[SourceAssessment]:
        """Assess all sources."""
        assessments = []
        
        rows = self.db.conn.execute("SELECT * FROM source_profiles").fetchall()
        
        for row in rows:
            profile = self.db._row_to_source_profile(row)
            
            # Get trace stats for this source
            trace_counts = profile.trace_counts_by_task or {}
            
            if not trace_counts:
                continue
            
            # Determine best/worst task types
            best_for = []
            avoid_for = []
            
            for task_type, stats in trace_counts.items():
                rate = stats.get("success_rate", 0.5)
                if rate >= 0.7:
                    best_for.append(task_type)
                elif rate <= 0.3:
                    avoid_for.append(task_type)
            
            # Overall value assessment
            total_success = sum(
                s.get("win", 0) + 0.5 * s.get("partial", 0)
                for s in trace_counts.values()
            )
            total_failure = sum(
                s.get("miss", 0) + 2 * s.get("misleading", 0)
                for s in trace_counts.values()
            )
            
            if total_failure > total_success * 2:
                overall = "negative"
            elif total_success > total_failure * 2:
                overall = "high"
            elif total_success > total_failure:
                overall = "medium"
            else:
                overall = "low"
            
            assessments.append(SourceAssessment(
                domain=profile.domain,
                overall_value=overall,
                best_for=best_for,
                avoid_for=avoid_for,
                notes=profile.functional_profile[:100] if profile.functional_profile else ""
            ))
        
        # Sort by overall value
        value_order = {"high": 0, "medium": 1, "low": 2, "negative": 3}
        assessments.sort(key=lambda x: value_order.get(x.overall_value, 4))
        
        return assessments
    
    def _analyze_content_type_performance(self) -> dict:
        """Analyze success rates by content type."""
        # Get all chunks with their document's content type
        rows = self.db.conn.execute("""
            SELECT c.id, d.content_type
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
        """).fetchall()
        
        chunk_content_types = {row["id"]: row["content_type"] for row in rows}
        
        # Get all traces and group by content type
        by_content_type = defaultdict(lambda: {"win": 0, "partial": 0, "miss": 0, "misleading": 0})
        
        traces = self._get_all_traces()
        for trace in traces:
            for chunk_id in trace.chunk_ids:
                content_type = chunk_content_types.get(chunk_id, "unknown")
                by_content_type[content_type][trace.outcome.value] += 1
        
        result = {}
        for content_type, counts in by_content_type.items():
            total = sum(counts.values())
            success = counts["win"] + 0.5 * counts["partial"]
            penalty = counts["miss"] + 2 * counts["misleading"]
            
            result[content_type] = {
                "total": total,
                "success_rate": success / (success + penalty) if (success + penalty) > 0 else 0.5
            }
        
        return result
    
    def _identify_gaps(self) -> list[KnowledgeGap]:
        """Identify knowledge gaps."""
        gaps = []
        
        # Gap 1: Task types with low success rates but high usage
        task_perf = self._analyze_task_type_performance(self._get_all_traces())
        
        for task_type, stats in task_perf.items():
            if stats["total"] >= 5 and stats["success_rate"] < 0.4:
                gaps.append(KnowledgeGap(
                    area=f"Content for {task_type} tasks",
                    severity="critical" if stats["success_rate"] < 0.2 else "moderate",
                    evidence=f"{stats['success_rate']:.0%} success rate over {stats['total']} uses",
                    suggestions=[
                        f"Acquire more content specifically suited for {task_type}",
                        f"Review and potentially remove low-quality {task_type} content"
                    ]
                ))
        
        # Gap 2: Concepts with many chunks but poor performance
        concepts = self.db.get_all_concepts()
        for concept in concepts:
            chunks = self.db.get_chunks_for_concept(concept.id)
            if len(chunks) < 3:
                continue
            
            # Aggregate performance
            all_traces = []
            for chunk, _ in chunks:
                all_traces.extend(self.db.get_usage_traces_for_chunk(chunk.id))
            
            if len(all_traces) < 5:
                continue
            
            wins = sum(1 for t in all_traces if t.outcome == UsageOutcome.WIN)
            rate = wins / len(all_traces)
            
            if rate < 0.3:
                gaps.append(KnowledgeGap(
                    area=f"Quality content about '{concept.name}'",
                    severity="moderate",
                    evidence=f"{len(chunks)} chunks but only {rate:.0%} success rate",
                    suggestions=[
                        f"Find better sources for {concept.name}",
                        "Consider if existing content is outdated"
                    ]
                ))
        
        # Gap 3: Concepts from gap_notes
        for concept in concepts:
            if concept.gap_notes:
                gaps.append(KnowledgeGap(
                    area=f"'{concept.name}': {concept.gap_notes}",
                    severity="minor",
                    evidence="Identified during consolidation",
                    suggestions=[]
                ))
        
        return gaps
    
    def _find_underutilized(self) -> list[Chunk]:
        """Find chunks that have never been retrieved/used."""
        rows = self.db.conn.execute("""
            SELECT c.* FROM chunks c
            WHERE NOT EXISTS (
                SELECT 1 FROM usage_traces ut
                WHERE ut.chunk_ids LIKE '%' || c.id || '%'
            )
        """).fetchall()
        
        return [self.db._row_to_chunk(row) for row in rows]
    
    def _find_consistently_misleading(self) -> list[tuple[str, dict]]:
        """Find chunks that consistently mislead."""
        results = []
        
        # Get all chunks with their traces
        chunk_rows = self.db.conn.execute("SELECT id, content, document_id FROM chunks").fetchall()
        
        for row in chunk_rows:
            chunk_id = row["id"]
            traces = self.db.get_usage_traces_for_chunk(chunk_id)
            
            if len(traces) < 3:
                continue
            
            misleading_count = sum(1 for t in traces if t.outcome == UsageOutcome.MISLEADING)
            
            if misleading_count >= 2 and misleading_count / len(traces) >= 0.3:
                doc = self.db.get_document(row["document_id"])
                results.append((chunk_id, {
                    "title": doc.title if doc else "Unknown",
                    "preview": row["content"][:100],
                    "misleading_count": misleading_count,
                    "total_uses": len(traces)
                }))
        
        # Sort by misleading count
        results.sort(key=lambda x: x[1]["misleading_count"], reverse=True)
        return results
    
    def _find_stale_but_accessed(self) -> list[Document]:
        """Find old documents that are still being accessed (might need refresh)."""
        # Documents ingested >90 days ago but accessed in last 30 days
        ninety_days_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        
        rows = self.db.conn.execute("""
            SELECT * FROM documents
            WHERE ingested_at < ? AND last_accessed > ?
            AND status = 'active'
        """, (ninety_days_ago, thirty_days_ago)).fetchall()
        
        return [self.db._row_to_document(row) for row in rows]
    
    def _identify_successful_patterns(self) -> list[str]:
        """Identify patterns in successful retrievals."""
        patterns = []
        
        # Pattern: Source + task type combinations that work well
        source_task_success = defaultdict(lambda: {"wins": 0, "total": 0})
        
        # Get traces with source info
        rows = self.db.conn.execute("""
            SELECT ut.*, sp.domain
            FROM usage_traces ut
            JOIN chunks c ON ut.chunk_ids LIKE '%' || c.id || '%'
            JOIN documents d ON c.document_id = d.id
            LEFT JOIN source_profiles sp ON d.source_profile_id = sp.id
        """).fetchall()
        
        for row in rows:
            domain = row["domain"] or "unknown"
            task = row["task_type"]
            key = (domain, task)
            source_task_success[key]["total"] += 1
            if row["outcome"] == "win":
                source_task_success[key]["wins"] += 1
        
        for (domain, task), stats in source_task_success.items():
            if stats["total"] >= 5:
                rate = stats["wins"] / stats["total"]
                if rate >= 0.8:
                    patterns.append(f"{domain} excels for {task} tasks ({rate:.0%} win rate)")
        
        # Pattern: Time of day patterns (if we had that data)
        # Pattern: Query complexity patterns
        # etc.
        
        return patterns
    
    def _find_co_retrieval_clusters(self) -> list[dict]:
        """Find clusters of chunks that are frequently used together."""
        # Get complement links
        rows = self.db.conn.execute("""
            SELECT source_id, target_id, weight
            FROM links
            WHERE relation = 'complements'
            ORDER BY weight DESC
            LIMIT 20
        """).fetchall()
        
        clusters = []
        for row in rows:
            source_chunk = self.db.get_chunk(row["source_id"])
            target_chunk = self.db.get_chunk(row["target_id"])
            
            if source_chunk and target_chunk:
                clusters.append({
                    "chunks": [
                        source_chunk.summary or source_chunk.content[:50],
                        target_chunk.summary or target_chunk.content[:50]
                    ],
                    "strength": row["weight"]
                })
        
        return clusters
    
    # ========================================================================
    # LLM-powered summaries
    # ========================================================================
    
    def _generate_health_summary(self, report: ReflectionReport) -> str:
        """Generate health summary using LLM."""
        prompt = SYSTEM_HEALTH_PROMPT.format(
            stats=json.dumps({
                "documents": report.total_documents,
                "chunks": report.total_chunks,
                "traces": report.total_traces,
                "overall_success_rate": f"{report.overall_success_rate:.1%}"
            }, indent=2),
            source_performance="\n".join(
                f"  {sa.domain}: {sa.overall_value}"
                for sa in report.source_assessments[:10]
            ),
            age_distribution="(not yet implemented)",
            usage_patterns=json.dumps(report.task_type_performance, indent=2),
            issues="\n".join([
                f"  - {len(report.consistently_misleading)} chunks consistently mislead",
                f"  - {len(report.underutilized_content)} chunks never used",
                f"  - {len(report.knowledge_gaps)} knowledge gaps identified"
            ])
        )
        
        return self._analyze_health(prompt)
    
    def _generate_gaps_analysis(self, report: ReflectionReport) -> str:
        """Generate gaps analysis using LLM."""
        concepts = self.db.get_all_concepts()
        
        prompt = KNOWLEDGE_GAPS_PROMPT.format(
            concepts_list=", ".join(c.name for c in concepts[:30]),
            task_type_performance=json.dumps(report.task_type_performance, indent=2),
            failed_queries="(tracking not yet implemented)",
            high_access_low_success="\n".join(
                f"  - {gap.area}: {gap.evidence}"
                for gap in report.knowledge_gaps[:5]
            )
        )
        
        return self._analyze_gaps(prompt)
    
    def _generate_lessons_learned(self, report: ReflectionReport) -> str:
        """Generate lessons learned summary using LLM."""
        # Format source rankings
        source_rankings = []
        for sa in report.source_assessments:
            source_rankings.append(
                f"{sa.domain}: {sa.overall_value} (best for: {', '.join(sa.best_for) or 'n/a'})"
            )
        
        # Get top performing chunks
        top_chunks = []
        rows = self.db.conn.execute("""
            SELECT c.id, c.summary, c.functional_profile
            FROM chunks c
            WHERE c.functional_profile IS NOT NULL
            LIMIT 5
        """).fetchall()
        for row in rows:
            if row["functional_profile"]:
                top_chunks.append(f"  - {row['summary'] or 'Untitled'}: {row['functional_profile'][:100]}")
        
        # Problematic chunks
        problematic = [
            f"  - {info['title']}: {info['misleading_count']} misleading uses"
            for _, info in report.consistently_misleading[:5]
        ]
        
        prompt = LEARNING_SUMMARY_PROMPT.format(
            source_rankings="\n".join(source_rankings[:10]),
            top_chunks="\n".join(top_chunks) or "(none yet)",
            problematic_chunks="\n".join(problematic) or "(none)",
            patterns="\n".join(f"  - {p}" for p in report.successful_patterns[:5]) or "(none yet)"
        )
        
        return self._summarize_lessons(prompt)


# ============================================================================
# Convenience functions
# ============================================================================

def run_reflection(db: KnowledgeDB) -> ReflectionReport:
    """Run reflection without LLM summaries."""
    reflector = Reflector(db)
    return reflector.reflect()


def run_reflection_with_llm(
    db: KnowledgeDB,
    llm_call: Callable[[str], str]
) -> ReflectionReport:
    """Run reflection with LLM-generated summaries."""
    reflector = Reflector(
        db,
        analyze_gaps=llm_call,
        analyze_health=llm_call,
        summarize_lessons=llm_call
    )
    return reflector.reflect()


# ============================================================================
# Quick insights (for frequent use without full reflection)
# ============================================================================

def quick_insights(db: KnowledgeDB) -> dict:
    """Get quick insights without running full reflection."""
    stats = db.stats()
    
    # Get recent traces
    rows = db.conn.execute("""
        SELECT task_type, outcome, COUNT(*) as count
        FROM usage_traces
        GROUP BY task_type, outcome
    """).fetchall()
    
    task_outcomes = defaultdict(lambda: defaultdict(int))
    for row in rows:
        task_outcomes[row["task_type"]][row["outcome"]] = row["count"]
    
    # Calculate quick stats
    insights = {
        "knowledge_base_size": {
            "documents": stats["documents"],
            "chunks": stats["chunks"],
            "concepts": stats["concepts"]
        },
        "activity": {
            "total_usage_traces": stats["usage_traces"]
        },
        "task_performance": {}
    }
    
    for task, outcomes in task_outcomes.items():
        total = sum(outcomes.values())
        wins = outcomes.get("win", 0)
        misleading = outcomes.get("misleading", 0)
        
        insights["task_performance"][task] = {
            "uses": total,
            "win_rate": f"{wins/total:.0%}" if total > 0 else "n/a",
            "misleading_rate": f"{misleading/total:.0%}" if total > 0 else "n/a"
        }
    
    # Flag issues
    insights["issues"] = []
    
    for task, perf in insights["task_performance"].items():
        if perf["uses"] >= 5 and float(perf["win_rate"].rstrip("%")) < 40:
            insights["issues"].append(f"Low success rate for {task} tasks")
    
    # Check for unused content
    unused_count = db.conn.execute("""
        SELECT COUNT(*) FROM chunks c
        WHERE NOT EXISTS (
            SELECT 1 FROM usage_traces ut
            WHERE ut.chunk_ids LIKE '%' || c.id || '%'
        )
    """).fetchone()[0]

    if unused_count > stats["chunks"] * 0.5:
        insights["issues"].append(f"{unused_count} chunks have never been used")

    # Check consolidation status - are there chunks ready for profiling?
    chunks_ready = db.conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT c.id
            FROM chunks c
            JOIN usage_traces ut ON ut.chunk_ids LIKE '%' || c.id || '%'
            WHERE c.functional_profile IS NULL
            GROUP BY c.id
            HAVING COUNT(*) >= 5
        )
    """).fetchone()[0]

    sources_ready = db.conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT sp.id
            FROM source_profiles sp
            JOIN documents d ON d.source_profile_id = sp.id
            JOIN chunks c ON c.document_id = d.id
            JOIN usage_traces ut ON ut.chunk_ids LIKE '%' || c.id || '%'
            WHERE sp.functional_profile IS NULL
            GROUP BY sp.id
            HAVING COUNT(*) >= 10
        )
    """).fetchone()[0]

    consolidation_needed = chunks_ready > 0 or sources_ready > 0
    insights["consolidation"] = {
        "status": "recommended" if consolidation_needed else "not_needed",
        "chunks_ready_for_profiling": chunks_ready,
        "sources_ready_for_profiling": sources_ready
    }

    if consolidation_needed:
        insights["issues"].append(
            f"Consolidation recommended: {chunks_ready} chunks and {sources_ready} sources ready for profiling"
        )

    return insights


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    from schema import init_db, Document, Chunk, ContentType, ChunkType, SourceProfile, SourceType
    from record import record_usage
    from consolidate import run_consolidation
    
    db = init_db(":memory:")
    
    # Create some test data
    source1 = SourceProfile.create("stackoverflow.com", SourceType.FORUM)
    db.insert_source_profile(source1)
    source1.trace_counts_by_task = {
        "debugging": {"win": 15, "miss": 2, "success_rate": 0.88},
        "conceptual_understanding": {"win": 3, "miss": 8, "success_rate": 0.27}
    }
    db.update_source_profile(source1)
    
    source2 = SourceProfile.create("arxiv.org", SourceType.ACADEMIC)
    db.insert_source_profile(source2)
    source2.trace_counts_by_task = {
        "conceptual_understanding": {"win": 12, "miss": 1, "success_rate": 0.92},
        "implementation_howto": {"win": 2, "miss": 6, "success_rate": 0.25}
    }
    db.update_source_profile(source2)
    
    # Create documents and chunks
    doc1 = Document.create(
        source="https://stackoverflow.com/q/123",
        content_type=ContentType.ARTICLE,
        title="Debug async Python",
        raw_content="Debug tip..."
    )
    doc1.source_profile_id = source1.id
    db.insert_document(doc1)
    
    chunk1 = Chunk.create(document_id=doc1.id, content="Use breakpoint()...", position=0)
    db.insert_chunk(chunk1)
    
    doc2 = Document.create(
        source="https://arxiv.org/abs/1234",
        content_type=ContentType.PAPER,
        title="Async Theory",
        raw_content="Theory..."
    )
    doc2.source_profile_id = source2.id
    db.insert_document(doc2)
    
    chunk2 = Chunk.create(document_id=doc2.id, content="Event loop theory...", position=0)
    db.insert_chunk(chunk2)
    
    # Add usage traces
    for i in range(8):
        record_usage(db, [chunk1.id], f"Debug task {i}", "win", task_type="debugging")
    for i in range(3):
        record_usage(db, [chunk1.id], f"Theory task {i}", "miss", task_type="conceptual_understanding")
    
    for i in range(5):
        record_usage(db, [chunk2.id], f"Theory task {i}", "win", task_type="conceptual_understanding")
    record_usage(db, [chunk2.id], "Impl task", "miss", task_type="implementation_howto")
    record_usage(db, [chunk2.id], "Impl task 2", "misleading", task_type="implementation_howto", 
                notes="Too abstract, led me astray")
    
    # Run consolidation first
    run_consolidation(db)
    
    # Run reflection
    print("Running reflection...")
    report = run_reflection(db)
    
    print(report.format_report())
    
    print("\n\n=== QUICK INSIGHTS ===")
    insights = quick_insights(db)
    print(json.dumps(insights, indent=2))
    
    db.close()
