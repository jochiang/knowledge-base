"""
Knowledge Harness - Context Assembly

Assembles "chunk dossiers" - the full context about a piece of knowledge
that gets surfaced for LLM reasoning during retrieval.

The key insight: rather than pre-computing a reliability score, we give
the LLM the usage history and let it reason about trust and applicability.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from schema import (
    KnowledgeDB, Chunk, Document, SourceProfile, UsageTrace,
    TaskType, UsageOutcome
)


@dataclass
class ChunkDossier:
    """
    Everything the LLM needs to reason about whether to trust/use a chunk.
    """
    # The content itself
    chunk: Chunk
    document: Document
    
    # Source context
    source_profile: Optional[SourceProfile]
    
    # Usage history - the learning signal
    usage_history: list[dict]  # Recent traces formatted for reading
    usage_stats: dict          # Aggregate stats
    usage_stats_by_task: dict  # Stats broken down by task type
    
    # Functional profile (if consolidated)
    functional_profile: Optional[str]  # Chunk-level
    source_functional_profile: Optional[str]  # Source-level
    
    # Related concepts
    concepts: list[str]
    
    def format_for_context(self, current_task_type: TaskType = None, verbose: bool = False) -> str:
        """
        Format the dossier as text for inclusion in LLM context.
        
        This is what the LLM sees when reasoning about whether to use this chunk.
        """
        lines = []
        
        # Header
        lines.append(f"=== CHUNK: {self.document.title} ===")
        lines.append(f"Source: {self.document.source}")
        if self.source_profile:
            lines.append(f"Source type: {self.source_profile.source_type.value}")
        lines.append("")
        
        # Content
        lines.append("CONTENT:")
        lines.append(self.chunk.content)
        lines.append("")
        
        # Chunk summary if available
        if self.chunk.summary:
            lines.append(f"SUMMARY: {self.chunk.summary}")
            lines.append("")
        
        # Functional profiles - the consolidated learning
        if self.functional_profile:
            lines.append("LEARNED PROFILE (this chunk):")
            lines.append(self.functional_profile)
            lines.append("")
        
        if self.source_functional_profile:
            lines.append(f"SOURCE PROFILE ({self.source_profile.domain}):")
            lines.append(self.source_functional_profile)
            lines.append("")
        
        # Usage history - the raw signal
        if self.usage_history:
            lines.append("USAGE HISTORY:")
            
            # If we have a current task type, show relevant history first
            if current_task_type:
                relevant = [h for h in self.usage_history 
                           if h["task_type"] == current_task_type.value]
                other = [h for h in self.usage_history 
                        if h["task_type"] != current_task_type.value]
                
                if relevant:
                    lines.append(f"  For {current_task_type.value} tasks:")
                    for h in relevant[:3]:
                        outcome_symbol = {"win": "✓", "partial": "~", "miss": "✗", "misleading": "⚠"}
                        symbol = outcome_symbol.get(h["outcome"], "?")
                        lines.append(f"    {h['timestamp']} [{symbol} {h['outcome']}]: {h['notes'] or h['context']}")
                    lines.append("")
                
                if other and verbose:
                    lines.append("  Other tasks:")
                    for h in other[:3]:
                        outcome_symbol = {"win": "✓", "partial": "~", "miss": "✗", "misleading": "⚠"}
                        symbol = outcome_symbol.get(h["outcome"], "?")
                        lines.append(f"    {h['timestamp']} [{h['task_type']}] [{symbol}]: {h['notes'] or h['context']}")
            else:
                for h in self.usage_history[:5]:
                    outcome_symbol = {"win": "✓", "partial": "~", "miss": "✗", "misleading": "⚠"}
                    symbol = outcome_symbol.get(h["outcome"], "?")
                    lines.append(f"  {h['timestamp']} [{h['task_type']}] [{symbol}]: {h['notes'] or h['context']}")
            lines.append("")
        
        # Aggregate stats
        if self.usage_stats["total"] > 0:
            lines.append("USAGE STATS:")
            counts = self.usage_stats["counts"]
            lines.append(f"  Overall: {counts['win']} wins, {counts['partial']} partial, "
                        f"{counts['miss']} misses, {counts['misleading']} misleading")
            
            if current_task_type and current_task_type.value in self.usage_stats_by_task.get("by_task", {}):
                task_stats = self.usage_stats_by_task["by_task"][current_task_type.value]
                lines.append(f"  For {current_task_type.value}: {task_stats['win']} wins, "
                           f"{task_stats['miss']} misses "
                           f"(success rate: {task_stats['success_rate']:.0%})")
            lines.append("")
        
        # Concepts
        if self.concepts:
            lines.append(f"CONCEPTS: {', '.join(self.concepts)}")
            lines.append("")
        
        # Rhetorical mode warning
        if self.chunk.rhetorical_mode.value in ("sarcastic", "argumentative", "anecdotal"):
            lines.append(f"⚠ RHETORICAL MODE: {self.chunk.rhetorical_mode.value}")
            lines.append("")
        
        return "\n".join(lines)
    
    def format_compact(self, current_task_type: TaskType = None) -> str:
        """
        Compact format for when context is limited.
        Uses functional profile if available, falls back to stats.
        """
        lines = []
        lines.append(f"[{self.document.title}]")
        
        # Content preview
        preview = self.chunk.content[:200] + "..." if len(self.chunk.content) > 200 else self.chunk.content
        lines.append(preview)
        
        # Functional profile or stats
        if self.functional_profile:
            lines.append(f"Profile: {self.functional_profile}")
        elif self.usage_stats["total"] > 0:
            if current_task_type and current_task_type.value in self.usage_stats_by_task.get("by_task", {}):
                task_stats = self.usage_stats_by_task["by_task"][current_task_type.value]
                lines.append(f"For {current_task_type.value}: {task_stats['success_rate']:.0%} success "
                           f"({task_stats['total']} uses)")
            else:
                lines.append(f"Overall: {self.usage_stats['success_rate']:.0%} success "
                           f"({self.usage_stats['total']} uses)")
        
        return "\n".join(lines)


class ContextAssembler:
    """
    Assembles chunk dossiers for retrieval results.
    """
    
    def __init__(self, db: KnowledgeDB):
        self.db = db
    
    def assemble_dossier(
        self,
        chunk: Chunk,
        max_history: int = 10
    ) -> ChunkDossier:
        """
        Build a complete dossier for a chunk.
        """
        # Get document
        document = self.db.get_document(chunk.document_id)
        
        # Get source profile
        source_profile = None
        if document.source_profile_id:
            row = self.db.conn.execute(
                "SELECT * FROM source_profiles WHERE id = ?",
                (document.source_profile_id,)
            ).fetchone()
            if row:
                source_profile = self.db._row_to_source_profile(row)
        else:
            # Try to infer from document source
            domain = self._extract_domain(document.source)
            if domain:
                source_profile = self.db.get_source_profile_by_domain(domain)
        
        # Get usage history
        usage_history = self.db.get_chunk_usage_history(chunk.id, limit=max_history)
        usage_stats = self.db.get_chunk_success_rate(chunk.id)
        usage_stats_by_task = self.db.get_chunk_success_rate_by_task(chunk.id)
        
        # Get concepts
        concept_tuples = self.db.get_concepts_for_chunk(chunk.id)
        concepts = [c.name for c, _ in concept_tuples]
        
        return ChunkDossier(
            chunk=chunk,
            document=document,
            source_profile=source_profile,
            usage_history=usage_history,
            usage_stats=usage_stats,
            usage_stats_by_task=usage_stats_by_task,
            functional_profile=chunk.functional_profile,
            source_functional_profile=source_profile.functional_profile if source_profile else None,
            concepts=concepts
        )
    
    def assemble_batch(
        self,
        chunks: list[Chunk],
        current_task_type: TaskType = None,
        max_history_per_chunk: int = 5,
        format: str = "full"  # "full", "compact", or "minimal"
    ) -> str:
        """
        Assemble dossiers for multiple chunks and format for context.
        
        This is the main entry point for retrieval → LLM reasoning.
        """
        dossiers = [self.assemble_dossier(c, max_history_per_chunk) for c in chunks]
        
        if format == "full":
            sections = [d.format_for_context(current_task_type, verbose=True) for d in dossiers]
            return "\n\n".join(sections)
        elif format == "compact":
            sections = [d.format_for_context(current_task_type, verbose=False) for d in dossiers]
            return "\n\n".join(sections)
        else:  # minimal
            sections = [d.format_compact(current_task_type) for d in dossiers]
            return "\n---\n".join(sections)
    
    def _extract_domain(self, source: str) -> Optional[str]:
        """Extract domain from a URL or filepath."""
        import re
        
        # URL pattern
        url_match = re.search(r'https?://([^/]+)', source)
        if url_match:
            return url_match.group(1).lower()
        
        # Could add more patterns for different source types
        return None


def format_usage_history_for_prompt(
    traces: list[UsageTrace],
    current_task_type: TaskType = None,
    max_traces: int = 5
) -> str:
    """
    Format usage traces as a readable history for LLM prompts.
    
    Prioritizes traces matching the current task type.
    """
    if not traces:
        return "No usage history available."
    
    lines = []
    
    # Sort by relevance to current task, then by recency
    def sort_key(t):
        task_match = 1 if current_task_type and t.task_type == current_task_type else 0
        return (-task_match, -t.timestamp.timestamp())
    
    sorted_traces = sorted(traces, key=sort_key)[:max_traces]
    
    for t in sorted_traces:
        outcome_symbol = {
            UsageOutcome.WIN: "✓ WIN",
            UsageOutcome.PARTIAL: "~ PARTIAL",
            UsageOutcome.MISS: "✗ MISS",
            UsageOutcome.MISLEADING: "⚠ MISLEADING"
        }
        symbol = outcome_symbol.get(t.outcome, "?")
        
        date_str = t.timestamp.strftime("%Y-%m-%d")
        task_str = t.task_type.value
        
        line = f"- {date_str} [{task_str}] {symbol}"
        if t.notes:
            line += f": {t.notes}"
        elif t.context_summary:
            line += f": {t.context_summary}"
        
        lines.append(line)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Demo
    from schema import init_db, Document, Chunk, ContentType, ChunkType
    from record import record_usage
    
    db = init_db(":memory:")
    
    # Create a document and chunk
    doc = Document.create(
        source="https://reddit.com/r/python/comments/abc123",
        content_type=ContentType.ARTICLE,
        title="Async Python Gotcha",
        raw_content="When you call await inside a sync callback..."
    )
    db.insert_document(doc)
    
    chunk = Chunk.create(
        document_id=doc.id,
        content="When you call await inside a sync callback, you get a deadlock. Use asyncio.create_task() instead.",
        position=0,
        chunk_type=ChunkType.EXAMPLE
    )
    db.insert_chunk(chunk)
    
    # Record some usage
    record_usage(db, [chunk.id], "Debugging async deadlock", "win", 
                 notes="Exactly described my issue")
    record_usage(db, [chunk.id], "Understanding event loop theory", "miss",
                 notes="Too practical, needed conceptual explanation")
    
    # Assemble dossier
    assembler = ContextAssembler(db)
    dossier = assembler.assemble_dossier(chunk)
    
    print("=== FULL FORMAT ===")
    print(dossier.format_for_context())
    
    print("\n=== COMPACT FORMAT ===")
    print(dossier.format_compact())
    
    db.close()
