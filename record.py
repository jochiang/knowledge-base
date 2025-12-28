"""
Knowledge Harness - Usage Recording

Records how knowledge was used and whether it helped.
This is the feedback loop that makes the system learn.
"""

import uuid
from datetime import datetime
from typing import Optional

from schema import (
    KnowledgeDB, UsageTrace, UsageOutcome
)


class SessionRecorder:
    """
    Tracks usage within a work session.
    
    Typical flow:
    1. Start a session
    2. As you retrieve and use chunks, log them
    3. At the end, record outcomes
    """
    
    def __init__(self, db: KnowledgeDB, session_id: str = None):
        self.db = db
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.retrieved_chunks: list[str] = []  # chunk IDs retrieved this session
        self.pending_traces: list[dict] = []   # traces waiting to be recorded
    
    def log_retrieval(self, chunk_ids: list[str], query: str = None, context: str = None):
        """
        Log that these chunks were retrieved for potential use.
        
        Call this right after retrieval, before you know the outcome.
        """
        self.retrieved_chunks.extend(chunk_ids)
        self.pending_traces.append({
            "chunk_ids": chunk_ids,
            "query": query,
            "context": context,
            "timestamp": datetime.utcnow()
        })
    
    def record_outcome(
        self,
        chunk_ids: list[str],
        context_summary: str,
        outcome: UsageOutcome,
        query: str = None,
        notes: str = None,
    ) -> str:
        """
        Record the outcome of using specific chunks.
        
        Args:
            chunk_ids: Which chunks were actually used
            context_summary: What were you trying to accomplish
            outcome: How well did it work
            query: The original query (if applicable)
            notes: Freeform notes on why it worked/failed
        
        Returns:
            trace_id
        """
        trace = UsageTrace.create(
            chunk_ids=chunk_ids,
            session_id=self.session_id,
            context_summary=context_summary,
            query=query,
            outcome=outcome,
            notes=notes
        )
        
        trace_id = self.db.insert_usage_trace(trace)
        
        # Update document access times for chunks that were used
        doc_ids_updated = set()
        for chunk_id in chunk_ids:
            chunk = self.db.get_chunk(chunk_id)
            if chunk and chunk.document_id not in doc_ids_updated:
                self.db.update_document_access(chunk.document_id)
                doc_ids_updated.add(chunk.document_id)
        
        return trace_id
    
    def quick_win(self, chunk_ids: list[str], context: str, notes: str = None) -> str:
        """Shorthand for recording a successful usage."""
        return self.record_outcome(
            chunk_ids=chunk_ids,
            context_summary=context,
            outcome=UsageOutcome.WIN,
            notes=notes
        )
    
    def quick_miss(self, chunk_ids: list[str], context: str, notes: str = None) -> str:
        """Shorthand for recording a miss (retrieved but not useful)."""
        return self.record_outcome(
            chunk_ids=chunk_ids,
            context_summary=context,
            outcome=UsageOutcome.MISS,
            notes=notes
        )
    
    def quick_misleading(self, chunk_ids: list[str], context: str, notes: str = None) -> str:
        """Shorthand for recording misleading content."""
        return self.record_outcome(
            chunk_ids=chunk_ids,
            context_summary=context,
            outcome=UsageOutcome.MISLEADING,
            notes=notes
        )
    
    def finalize_pending(self, default_outcome: UsageOutcome = UsageOutcome.PARTIAL):
        """
        Record all pending traces that weren't explicitly resolved.
        
        Useful at end of session to ensure everything is logged.
        """
        for pending in self.pending_traces:
            # Check if these chunks already have a trace this session
            existing = self.db.conn.execute("""
                SELECT id FROM usage_traces 
                WHERE session_id = ? AND chunk_ids LIKE ?
            """, (self.session_id, f'%{pending["chunk_ids"][0]}%')).fetchone()
            
            if not existing:
                self.record_outcome(
                    chunk_ids=pending["chunk_ids"],
                    context_summary=pending.get("context", "Session retrieval"),
                    outcome=default_outcome,
                    query=pending.get("query")
                )
        
        self.pending_traces = []
    
    def session_summary(self) -> dict:
        """Get statistics for this session."""
        traces = self.db.conn.execute(
            "SELECT * FROM usage_traces WHERE session_id = ?",
            (self.session_id,)
        ).fetchall()
        
        outcomes = {"win": 0, "partial": 0, "miss": 0, "misleading": 0}
        total_chunks = set()
        
        for row in traces:
            outcomes[row["outcome"]] += 1
            import json
            chunk_ids = json.loads(row["chunk_ids"])
            total_chunks.update(chunk_ids)
        
        return {
            "session_id": self.session_id,
            "total_traces": len(traces),
            "unique_chunks_used": len(total_chunks),
            "outcomes": outcomes
        }


# ============================================================================
# Convenience functions
# ============================================================================

def record_usage(
    db: KnowledgeDB,
    chunk_ids: list[str],
    context: str,
    outcome: str,  # "win", "partial", "miss", "misleading"
    task_type: str = "other",  # See TaskType enum
    session_id: str = None,
    query: str = None,
    notes: str = None,
) -> str:
    """
    One-shot usage recording without managing a session.
    
    Useful for quick logging.
    """
    from schema import TaskType
    outcome_enum = UsageOutcome(outcome)
    task_type_enum = TaskType(task_type)
    trace = UsageTrace.create(
        chunk_ids=chunk_ids,
        session_id=session_id or "ad-hoc",
        context_summary=context,
        task_type=task_type_enum,
        query=query,
        outcome=outcome_enum,
        notes=notes
    )
    return db.insert_usage_trace(trace)


def get_usage_history(db: KnowledgeDB, chunk_id: str) -> list[dict]:
    """
    Get the full usage history for a chunk.
    
    Returns list of dicts with trace info.
    """
    traces = db.get_usage_traces_for_chunk(chunk_id)
    return [
        {
            "id": t.id,
            "session": t.session_id,
            "context": t.context_summary,
            "query": t.query,
            "outcome": t.outcome.value,
            "notes": t.notes,
            "timestamp": t.timestamp.isoformat()
        }
        for t in traces
    ]


def chunk_report(db: KnowledgeDB, chunk_id: str) -> dict:
    """
    Get a full report on a chunk including usage stats.
    """
    chunk = db.get_chunk(chunk_id)
    if not chunk:
        return {"error": "Chunk not found"}
    
    doc = db.get_document(chunk.document_id)
    concepts = db.get_concepts_for_chunk(chunk_id)
    usage_stats = db.get_chunk_success_rate(chunk_id)
    history = get_usage_history(db, chunk_id)
    
    return {
        "chunk": {
            "id": chunk.id,
            "content_preview": chunk.content[:200],
            "summary": chunk.summary,
            "position": chunk.position,
            "type": chunk.chunk_type.value
        },
        "document": {
            "id": doc.id,
            "title": doc.title,
            "source": doc.source
        },
        "concepts": [{"name": c.name, "weight": w} for c, w in concepts],
        "usage": usage_stats,
        "history": history
    }


# ============================================================================
# Interactive recording helpers (for Claude Code)
# ============================================================================

def prompt_for_outcome() -> UsageOutcome:
    """
    Interactive prompt for recording outcome.
    
    For use in Claude Code sessions.
    """
    print("\nHow useful was this content?")
    print("  [w]in      - Directly helpful, got what I needed")
    print("  [p]artial  - Somewhat helpful, needed more")
    print("  [m]iss     - Retrieved but not useful")
    print("  [x]        - Misleading or caused confusion")
    
    while True:
        choice = input("\nOutcome [w/p/m/x]: ").strip().lower()
        if choice in ('w', 'win'):
            return UsageOutcome.WIN
        elif choice in ('p', 'partial'):
            return UsageOutcome.PARTIAL
        elif choice in ('m', 'miss'):
            return UsageOutcome.MISS
        elif choice in ('x', 'misleading'):
            return UsageOutcome.MISLEADING
        else:
            print("Please enter w, p, m, or x")


def interactive_record(db: KnowledgeDB, chunk_ids: list[str], context: str) -> str:
    """
    Interactively record usage with prompts.
    
    For use in Claude Code sessions.
    """
    print(f"\nRecording usage for {len(chunk_ids)} chunk(s)")
    print(f"Context: {context}")
    
    outcome = prompt_for_outcome()
    notes = input("Notes (optional): ").strip() or None
    
    trace_id = record_usage(
        db=db,
        chunk_ids=chunk_ids,
        context=context,
        outcome=outcome.value,
        notes=notes
    )
    
    print(f"\nRecorded trace: {trace_id}")
    return trace_id


if __name__ == "__main__":
    # Demo
    from schema import init_db
    from ingest import quick_ingest
    
    db = init_db(":memory:")
    
    # Ingest something
    result = quick_ingest(db, "test.md", "# Test\n\nSome test content about widgets.")
    chunk_ids = result.chunk_ids
    
    # Create a session
    recorder = SessionRecorder(db, session_id="demo-session")
    
    # Log some usage
    recorder.log_retrieval(chunk_ids, query="how do widgets work?")
    
    # Record outcome
    trace_id = recorder.quick_win(
        chunk_ids=chunk_ids,
        context="Understanding widget mechanics",
        notes="Content explained the core concept well"
    )
    
    print(f"Recorded trace: {trace_id}")
    print(f"\nSession summary: {recorder.session_summary()}")
    print(f"\nChunk report: {chunk_report(db, chunk_ids[0])}")
    
    db.close()
