"""
Knowledge Harness - Database Schema and Models

SQLite-backed storage with Python dataclasses for type safety.
"""

import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
import json


# ============================================================================
# Enums
# ============================================================================

class ContentType(Enum):
    ARTICLE = "article"
    PAPER = "paper"
    NOTE = "note"
    CODE = "code"
    CONVERSATION = "conversation"
    REFERENCE = "reference"
    OTHER = "other"


class DocumentStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"


class ChunkType(Enum):
    NARRATIVE = "narrative"
    ARGUMENT = "argument"
    DATA = "data"
    CODE = "code"
    DEFINITION = "definition"
    EXAMPLE = "example"
    OTHER = "other"


class ConceptType(Enum):
    TOPIC = "topic"
    ENTITY = "entity"
    METHOD = "method"
    CLAIM = "claim"
    QUESTION = "question"


class UsageOutcome(Enum):
    WIN = "win"
    PARTIAL = "partial"
    MISS = "miss"
    MISLEADING = "misleading"


class TaskType(Enum):
    FACTUAL_LOOKUP = "factual_lookup"
    IMPLEMENTATION_HOWTO = "implementation_howto"
    CONCEPTUAL_UNDERSTANDING = "conceptual_understanding"
    OPINION_GATHERING = "opinion_gathering"
    DECISION_SUPPORT = "decision_support"
    DEBUGGING = "debugging"
    EXPLORATORY_RESEARCH = "exploratory_research"
    CREATIVE_INSPIRATION = "creative_inspiration"
    OTHER = "other"


class SourceType(Enum):
    FORUM = "forum"
    ACADEMIC = "academic"
    NEWS = "news"
    DOCS = "docs"
    SOCIAL = "social"
    CONVERSATION = "conversation"
    PERSONAL_NOTES = "personal_notes"
    OTHER = "other"


class RhetoricalMode(Enum):
    EXPLANATORY = "explanatory"
    ARGUMENTATIVE = "argumentative"
    ANECDOTAL = "anecdotal"
    SARCASTIC = "sarcastic"
    INSTRUCTIONAL = "instructional"
    NEUTRAL = "neutral"
    OTHER = "other"


class EntityType(Enum):
    DOCUMENT = "document"
    CHUNK = "chunk"
    CONCEPT = "concept"


class LinkCreator(Enum):
    AUTO = "auto"
    MANUAL = "manual"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class SourceProfile:
    """
    Captures learned understanding of a source's functional strengths and weaknesses.
    Not a reliability score—a capability profile.
    """
    id: str
    domain: str  # e.g., "reddit.com", "arxiv.org", "internal-wiki"
    source_type: SourceType
    functional_profile: Optional[str] = None  # LLM-generated prose assessment
    strengths: list[str] = field(default_factory=list)  # e.g., ["practical_howto", "community_sentiment"]
    weaknesses: list[str] = field(default_factory=list)  # e.g., ["factual_accuracy", "theoretical_depth"]
    trace_counts_by_task: dict = field(default_factory=dict)  # {"debugging": {"win": 5, "miss": 1}, ...}
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(cls, domain: str, source_type: SourceType = SourceType.OTHER, **kwargs):
        return cls(
            id=str(uuid.uuid4()),
            domain=domain.lower().strip(),
            source_type=source_type,
            **kwargs
        )


@dataclass
class Document:
    id: str
    source: str
    content_type: ContentType
    title: str
    raw_content: str
    source_profile_id: Optional[str] = None  # FK to SourceProfile
    top_summary: Optional[str] = None
    key_claims: list[str] = field(default_factory=list)
    ingested_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    status: DocumentStatus = DocumentStatus.ACTIVE

    @classmethod
    def create(cls, source: str, content_type: ContentType, title: str, raw_content: str, **kwargs):
        return cls(
            id=str(uuid.uuid4()),
            source=source,
            content_type=content_type,
            title=title,
            raw_content=raw_content,
            **kwargs
        )


@dataclass
class Chunk:
    id: str
    document_id: str
    content: str
    position: int
    summary: Optional[str] = None
    chunk_type: ChunkType = ChunkType.OTHER
    token_count: int = 0
    # LLM-generated prose assessment of what this chunk is good for
    functional_profile: Optional[str] = None
    # Flags for retrieval-time reasoning
    needs_context: bool = False  # True if chunk doesn't stand alone well
    rhetorical_mode: RhetoricalMode = RhetoricalMode.NEUTRAL

    @classmethod
    def create(cls, document_id: str, content: str, position: int, **kwargs):
        return cls(
            id=str(uuid.uuid4()),
            document_id=document_id,
            content=content,
            position=position,
            token_count=len(content.split()),  # rough estimate
            **kwargs
        )


@dataclass
class Concept:
    id: str
    name: str
    description: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    concept_type: ConceptType = ConceptType.TOPIC
    # LLM-generated assessment of our knowledge about this concept
    functional_profile: Optional[str] = None
    gap_notes: Optional[str] = None  # What's missing in our knowledge
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(cls, name: str, **kwargs):
        return cls(
            id=str(uuid.uuid4()),
            name=name.lower().strip(),
            **kwargs
        )


@dataclass
class ChunkConcept:
    """Junction table linking chunks to concepts with weight."""
    chunk_id: str
    concept_id: str
    weight: float = 1.0


@dataclass
class UsageTrace:
    id: str
    chunk_ids: list[str]
    session_id: str
    context_summary: str
    task_type: TaskType = TaskType.OTHER  # What kind of task was this
    query: Optional[str] = None
    outcome: UsageOutcome = UsageOutcome.PARTIAL
    notes: Optional[str] = None  # Why did it work or fail - the richest signal
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(cls, chunk_ids: list[str], session_id: str, context_summary: str, **kwargs):
        return cls(
            id=str(uuid.uuid4()),
            chunk_ids=chunk_ids,
            session_id=session_id,
            context_summary=context_summary,
            **kwargs
        )


@dataclass
class Link:
    id: str
    source_type: EntityType
    source_id: str
    target_type: EntityType
    target_id: str
    relation: str
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: LinkCreator = LinkCreator.AUTO

    @classmethod
    def create(cls, source_type: EntityType, source_id: str, 
               target_type: EntityType, target_id: str, relation: str, **kwargs):
        return cls(
            id=str(uuid.uuid4()),
            source_type=source_type,
            source_id=source_id,
            target_type=target_type,
            target_id=target_id,
            relation=relation,
            **kwargs
        )


# ============================================================================
# Database Manager
# ============================================================================

class KnowledgeDB:
    """SQLite database manager for the knowledge harness."""

    SCHEMA = """
    -- Source Profiles (learned understanding of source reliability by context)
    CREATE TABLE IF NOT EXISTS source_profiles (
        id TEXT PRIMARY KEY,
        domain TEXT NOT NULL UNIQUE,
        source_type TEXT DEFAULT 'other',
        functional_profile TEXT,
        strengths TEXT,  -- JSON array
        weaknesses TEXT,  -- JSON array
        trace_counts_by_task TEXT,  -- JSON object
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    -- Documents
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        source_profile_id TEXT,
        content_type TEXT NOT NULL,
        title TEXT NOT NULL,
        raw_content TEXT NOT NULL,
        top_summary TEXT,
        key_claims TEXT,  -- JSON array
        ingested_at TEXT NOT NULL,
        last_accessed TEXT NOT NULL,
        access_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'active',
        FOREIGN KEY (source_profile_id) REFERENCES source_profiles(id)
    );

    -- Chunks
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        content TEXT NOT NULL,
        position INTEGER NOT NULL,
        summary TEXT,
        chunk_type TEXT DEFAULT 'other',
        token_count INTEGER DEFAULT 0,
        functional_profile TEXT,
        needs_context INTEGER DEFAULT 0,
        rhetorical_mode TEXT DEFAULT 'neutral',
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
    );

    -- Concepts
    CREATE TABLE IF NOT EXISTS concepts (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        aliases TEXT,  -- JSON array
        concept_type TEXT DEFAULT 'topic',
        functional_profile TEXT,
        gap_notes TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    -- Chunk-Concept junction
    CREATE TABLE IF NOT EXISTS chunk_concepts (
        chunk_id TEXT NOT NULL,
        concept_id TEXT NOT NULL,
        weight REAL DEFAULT 1.0,
        PRIMARY KEY (chunk_id, concept_id),
        FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
        FOREIGN KEY (concept_id) REFERENCES concepts(id) ON DELETE CASCADE
    );

    -- Usage traces
    CREATE TABLE IF NOT EXISTS usage_traces (
        id TEXT PRIMARY KEY,
        chunk_ids TEXT NOT NULL,  -- JSON array
        session_id TEXT NOT NULL,
        task_type TEXT DEFAULT 'other',
        context_summary TEXT NOT NULL,
        query TEXT,
        outcome TEXT DEFAULT 'partial',
        notes TEXT,
        timestamp TEXT NOT NULL
    );

    -- Links (graph edges)
    CREATE TABLE IF NOT EXISTS links (
        id TEXT PRIMARY KEY,
        source_type TEXT NOT NULL,
        source_id TEXT NOT NULL,
        target_type TEXT NOT NULL,
        target_id TEXT NOT NULL,
        relation TEXT NOT NULL,
        weight REAL DEFAULT 1.0,
        created_at TEXT NOT NULL,
        created_by TEXT DEFAULT 'auto'
    );

    -- Embeddings (stored as JSON arrays for simplicity)
    CREATE TABLE IF NOT EXISTS embeddings (
        chunk_id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        embedding TEXT NOT NULL,  -- JSON array of floats
        created_at TEXT NOT NULL,
        FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
    );

    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
    CREATE INDEX IF NOT EXISTS idx_chunk_concepts_chunk ON chunk_concepts(chunk_id);
    CREATE INDEX IF NOT EXISTS idx_chunk_concepts_concept ON chunk_concepts(concept_id);
    CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_type, source_id);
    CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_type, target_id);
    CREATE INDEX IF NOT EXISTS idx_usage_traces_session ON usage_traces(session_id);
    CREATE INDEX IF NOT EXISTS idx_usage_traces_task_type ON usage_traces(task_type);
    CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(name);
    CREATE INDEX IF NOT EXISTS idx_source_profiles_domain ON source_profiles(domain);
    CREATE INDEX IF NOT EXISTS idx_documents_source_profile ON documents(source_profile_id);
    CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name);
    """

    def __init__(self, db_path: str | Path = "knowledge.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------------

    def insert_document(self, doc: Document) -> str:
        self.conn.execute("""
            INSERT INTO documents (id, source, content_type, title, raw_content, 
                                   top_summary, key_claims, ingested_at, last_accessed,
                                   access_count, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc.id, doc.source, doc.content_type.value, doc.title, doc.raw_content,
            doc.top_summary, json.dumps(doc.key_claims), 
            doc.ingested_at.isoformat(), doc.last_accessed.isoformat(),
            doc.access_count, doc.status.value
        ))
        self.conn.commit()
        return doc.id

    def get_document(self, doc_id: str) -> Optional[Document]:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_document(row)

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            source=row["source"],
            content_type=ContentType(row["content_type"]),
            title=row["title"],
            raw_content=row["raw_content"],
            top_summary=row["top_summary"],
            key_claims=json.loads(row["key_claims"]) if row["key_claims"] else [],
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            access_count=row["access_count"],
            status=DocumentStatus(row["status"])
        )

    def update_document_access(self, doc_id: str):
        """Update last_accessed and increment access_count."""
        self.conn.execute("""
            UPDATE documents 
            SET last_accessed = ?, access_count = access_count + 1
            WHERE id = ?
        """, (datetime.now(timezone.utc).isoformat(), doc_id))
        self.conn.commit()

    # ------------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------------

    def insert_chunk(self, chunk: Chunk) -> str:
        self.conn.execute("""
            INSERT INTO chunks (id, document_id, content, position, summary, 
                               chunk_type, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk.id, chunk.document_id, chunk.content, chunk.position,
            chunk.summary, chunk.chunk_type.value, chunk.token_count
        ))
        self.conn.commit()
        return chunk.id

    def get_chunks_for_document(self, doc_id: str) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY position",
            (doc_id,)
        ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        row = self.conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_chunk(row)

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            content=row["content"],
            position=row["position"],
            summary=row["summary"],
            chunk_type=ChunkType(row["chunk_type"]),
            token_count=row["token_count"],
            functional_profile=row["functional_profile"] if "functional_profile" in row.keys() else None,
            needs_context=bool(row["needs_context"]) if "needs_context" in row.keys() else False,
            rhetorical_mode=RhetoricalMode(row["rhetorical_mode"]) if "rhetorical_mode" in row.keys() and row["rhetorical_mode"] else RhetoricalMode.NEUTRAL
        )

    # ------------------------------------------------------------------------
    # Concept operations
    # ------------------------------------------------------------------------

    def insert_concept(self, concept: Concept) -> str:
        self.conn.execute("""
            INSERT INTO concepts (id, name, description, aliases, concept_type,
                                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            concept.id, concept.name, concept.description,
            json.dumps(concept.aliases), concept.concept_type.value,
            concept.created_at.isoformat(), concept.updated_at.isoformat()
        ))
        self.conn.commit()
        return concept.id

    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        row = self.conn.execute(
            "SELECT * FROM concepts WHERE name = ?", (name.lower().strip(),)
        ).fetchone()
        if not row:
            return None
        return self._row_to_concept(row)

    def get_or_create_concept(self, name: str, **kwargs) -> Concept:
        """Get existing concept or create new one."""
        existing = self.get_concept_by_name(name)
        if existing:
            return existing
        concept = Concept.create(name, **kwargs)
        self.insert_concept(concept)
        return concept

    def _row_to_concept(self, row: sqlite3.Row) -> Concept:
        return Concept(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            aliases=json.loads(row["aliases"]) if row["aliases"] else [],
            concept_type=ConceptType(row["concept_type"]),
            functional_profile=row["functional_profile"] if "functional_profile" in row.keys() else None,
            gap_notes=row["gap_notes"] if "gap_notes" in row.keys() else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )

    def link_chunk_to_concept(self, chunk_id: str, concept_id: str, weight: float = 1.0):
        self.conn.execute("""
            INSERT OR REPLACE INTO chunk_concepts (chunk_id, concept_id, weight)
            VALUES (?, ?, ?)
        """, (chunk_id, concept_id, weight))
        self.conn.commit()

    def get_concepts_for_chunk(self, chunk_id: str) -> list[tuple[Concept, float]]:
        rows = self.conn.execute("""
            SELECT c.*, cc.weight FROM concepts c
            JOIN chunk_concepts cc ON c.id = cc.concept_id
            WHERE cc.chunk_id = ?
        """, (chunk_id,)).fetchall()
        return [(self._row_to_concept(row), row["weight"]) for row in rows]

    def get_chunks_for_concept(self, concept_id: str) -> list[tuple[Chunk, float]]:
        rows = self.conn.execute("""
            SELECT ch.*, cc.weight FROM chunks ch
            JOIN chunk_concepts cc ON ch.id = cc.chunk_id
            WHERE cc.concept_id = ?
        """, (concept_id,)).fetchall()
        return [(self._row_to_chunk(row), row["weight"]) for row in rows]

    # ------------------------------------------------------------------------
    # Usage trace operations
    # ------------------------------------------------------------------------

    def insert_usage_trace(self, trace: UsageTrace) -> str:
        self.conn.execute("""
            INSERT INTO usage_traces (id, chunk_ids, session_id, task_type, context_summary,
                                      query, outcome, notes, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trace.id, json.dumps(trace.chunk_ids), trace.session_id,
            trace.task_type.value, trace.context_summary, trace.query, 
            trace.outcome.value, trace.notes, trace.timestamp.isoformat()
        ))
        self.conn.commit()
        return trace.id

    def get_usage_traces_for_chunk(self, chunk_id: str) -> list[UsageTrace]:
        """Get all usage traces that include this chunk."""
        rows = self.conn.execute(
            "SELECT * FROM usage_traces WHERE chunk_ids LIKE ?",
            (f'%{chunk_id}%',)
        ).fetchall()
        return [self._row_to_usage_trace(row) for row in rows]

    def _row_to_usage_trace(self, row: sqlite3.Row) -> UsageTrace:
        return UsageTrace(
            id=row["id"],
            chunk_ids=json.loads(row["chunk_ids"]),
            session_id=row["session_id"],
            context_summary=row["context_summary"],
            task_type=TaskType(row["task_type"]) if "task_type" in row.keys() and row["task_type"] else TaskType.OTHER,
            query=row["query"],
            outcome=UsageOutcome(row["outcome"]),
            notes=row["notes"],
            timestamp=datetime.fromisoformat(row["timestamp"])
        )

    def get_chunk_success_rate(self, chunk_id: str) -> dict:
        """Calculate success metrics for a chunk."""
        traces = self.get_usage_traces_for_chunk(chunk_id)
        if not traces:
            return {"total": 0, "success_rate": None}
        
        counts = {"win": 0, "partial": 0, "miss": 0, "misleading": 0}
        for t in traces:
            counts[t.outcome.value] += 1
        
        # Weighted success rate (misleading penalized 2x)
        total = sum(counts.values())
        success = counts["win"] + 0.5 * counts["partial"]
        penalty = counts["miss"] + 2 * counts["misleading"]
        rate = success / (success + penalty) if (success + penalty) > 0 else 0.5
        
        return {
            "total": total,
            "counts": counts,
            "success_rate": rate
        }

    def get_chunk_success_rate_by_task(self, chunk_id: str) -> dict:
        """Calculate success metrics for a chunk, broken down by task type."""
        traces = self.get_usage_traces_for_chunk(chunk_id)
        if not traces:
            return {"total": 0, "by_task": {}}
        
        by_task = {}
        for t in traces:
            task = t.task_type.value
            if task not in by_task:
                by_task[task] = {"win": 0, "partial": 0, "miss": 0, "misleading": 0}
            by_task[task][t.outcome.value] += 1
        
        # Calculate success rate per task type
        for task, counts in by_task.items():
            total = sum(counts.values())
            success = counts["win"] + 0.5 * counts["partial"]
            penalty = counts["miss"] + 2 * counts["misleading"]
            by_task[task]["success_rate"] = success / (success + penalty) if (success + penalty) > 0 else 0.5
            by_task[task]["total"] = total
        
        return {
            "total": len(traces),
            "by_task": by_task
        }

    def get_chunk_usage_history(self, chunk_id: str, limit: int = 10) -> list[dict]:
        """
        Get formatted usage history for a chunk, suitable for LLM reasoning.
        Returns most recent traces with task type and notes.
        """
        traces = self.get_usage_traces_for_chunk(chunk_id)
        traces.sort(key=lambda t: t.timestamp, reverse=True)
        
        return [
            {
                "timestamp": t.timestamp.strftime("%Y-%m-%d"),
                "task_type": t.task_type.value,
                "outcome": t.outcome.value,
                "context": t.context_summary,
                "notes": t.notes
            }
            for t in traces[:limit]
        ]

    # ------------------------------------------------------------------------
    # Source Profile operations
    # ------------------------------------------------------------------------

    def insert_source_profile(self, profile: SourceProfile) -> str:
        self.conn.execute("""
            INSERT INTO source_profiles (id, domain, source_type, functional_profile,
                                         strengths, weaknesses, trace_counts_by_task,
                                         created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            profile.id, profile.domain, profile.source_type.value,
            profile.functional_profile, json.dumps(profile.strengths),
            json.dumps(profile.weaknesses), json.dumps(profile.trace_counts_by_task),
            profile.created_at.isoformat(), profile.updated_at.isoformat()
        ))
        self.conn.commit()
        return profile.id

    def get_source_profile_by_domain(self, domain: str) -> Optional[SourceProfile]:
        row = self.conn.execute(
            "SELECT * FROM source_profiles WHERE domain = ?", (domain.lower().strip(),)
        ).fetchone()
        if not row:
            return None
        return self._row_to_source_profile(row)

    def get_or_create_source_profile(self, domain: str, source_type: SourceType = SourceType.OTHER) -> SourceProfile:
        """Get existing source profile or create new one."""
        existing = self.get_source_profile_by_domain(domain)
        if existing:
            return existing
        profile = SourceProfile.create(domain, source_type)
        self.insert_source_profile(profile)
        return profile

    def update_source_profile(self, profile: SourceProfile):
        """Update a source profile (e.g., after consolidation)."""
        self.conn.execute("""
            UPDATE source_profiles 
            SET functional_profile = ?, strengths = ?, weaknesses = ?,
                trace_counts_by_task = ?, updated_at = ?
            WHERE id = ?
        """, (
            profile.functional_profile, json.dumps(profile.strengths),
            json.dumps(profile.weaknesses), json.dumps(profile.trace_counts_by_task),
            datetime.now(timezone.utc).isoformat(), profile.id
        ))
        self.conn.commit()

    def _row_to_source_profile(self, row: sqlite3.Row) -> SourceProfile:
        return SourceProfile(
            id=row["id"],
            domain=row["domain"],
            source_type=SourceType(row["source_type"]),
            functional_profile=row["functional_profile"],
            strengths=json.loads(row["strengths"]) if row["strengths"] else [],
            weaknesses=json.loads(row["weaknesses"]) if row["weaknesses"] else [],
            trace_counts_by_task=json.loads(row["trace_counts_by_task"]) if row["trace_counts_by_task"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )

    def update_chunk_functional_profile(self, chunk_id: str, profile: str):
        """Update a chunk's functional profile after consolidation."""
        self.conn.execute(
            "UPDATE chunks SET functional_profile = ? WHERE id = ?",
            (profile, chunk_id)
        )
        self.conn.commit()

    def update_concept_functional_profile(self, concept_id: str, profile: str, gap_notes: str = None):
        """Update a concept's functional profile and gap notes."""
        self.conn.execute(
            "UPDATE concepts SET functional_profile = ?, gap_notes = ?, updated_at = ? WHERE id = ?",
            (profile, gap_notes, datetime.now(timezone.utc).isoformat(), concept_id)
        )
        self.conn.commit()

    # ------------------------------------------------------------------------
    # Link operations
    # ------------------------------------------------------------------------

    def insert_link(self, link: Link) -> str:
        self.conn.execute("""
            INSERT INTO links (id, source_type, source_id, target_type, target_id,
                              relation, weight, created_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            link.id, link.source_type.value, link.source_id,
            link.target_type.value, link.target_id, link.relation,
            link.weight, link.created_at.isoformat(), link.created_by.value
        ))
        self.conn.commit()
        return link.id

    def get_links_from(self, entity_type: EntityType, entity_id: str) -> list[Link]:
        rows = self.conn.execute("""
            SELECT * FROM links WHERE source_type = ? AND source_id = ?
        """, (entity_type.value, entity_id)).fetchall()
        return [self._row_to_link(row) for row in rows]

    def get_links_to(self, entity_type: EntityType, entity_id: str) -> list[Link]:
        rows = self.conn.execute("""
            SELECT * FROM links WHERE target_type = ? AND target_id = ?
        """, (entity_type.value, entity_id)).fetchall()
        return [self._row_to_link(row) for row in rows]

    def _row_to_link(self, row: sqlite3.Row) -> Link:
        return Link(
            id=row["id"],
            source_type=EntityType(row["source_type"]),
            source_id=row["source_id"],
            target_type=EntityType(row["target_type"]),
            target_id=row["target_id"],
            relation=row["relation"],
            weight=row["weight"],
            created_at=datetime.fromisoformat(row["created_at"]),
            created_by=LinkCreator(row["created_by"])
        )

    # ------------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------------

    def search_documents(self, query: str, limit: int = 10) -> list[Document]:
        """Simple text search across documents."""
        rows = self.conn.execute("""
            SELECT * FROM documents 
            WHERE title LIKE ? OR raw_content LIKE ? OR top_summary LIKE ?
            ORDER BY last_accessed DESC
            LIMIT ?
        """, (f'%{query}%', f'%{query}%', f'%{query}%', limit)).fetchall()
        return [self._row_to_document(row) for row in rows]

    def search_chunks(self, query: str, limit: int = 20) -> list[Chunk]:
        """Simple text search across chunks."""
        rows = self.conn.execute("""
            SELECT * FROM chunks 
            WHERE content LIKE ? OR summary LIKE ?
            LIMIT ?
        """, (f'%{query}%', f'%{query}%', limit)).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_all_concepts(self) -> list[Concept]:
        rows = self.conn.execute("SELECT * FROM concepts ORDER BY name").fetchall()
        return [self._row_to_concept(row) for row in rows]

    def stats(self) -> dict:
        """Get basic statistics about the knowledge base."""
        return {
            "documents": self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            "chunks": self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "concepts": self.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0],
            "source_profiles": self.conn.execute("SELECT COUNT(*) FROM source_profiles").fetchone()[0],
            "usage_traces": self.conn.execute("SELECT COUNT(*) FROM usage_traces").fetchone()[0],
            "links": self.conn.execute("SELECT COUNT(*) FROM links").fetchone()[0],
            "embeddings": self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0],
        }

    # ------------------------------------------------------------------------
    # Embedding operations
    # ------------------------------------------------------------------------

    def store_embedding(self, chunk_id: str, embedding: list[float], model_name: str):
        """Store an embedding for a chunk."""
        self.conn.execute("""
            INSERT OR REPLACE INTO embeddings (chunk_id, model_name, embedding, created_at)
            VALUES (?, ?, ?, ?)
        """, (chunk_id, model_name, json.dumps(embedding), datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def get_embedding(self, chunk_id: str) -> Optional[list[float]]:
        """Get the embedding for a chunk."""
        row = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if not row:
            return None
        return json.loads(row["embedding"])

    def get_all_embeddings(self, model_name: str = None) -> list[tuple[str, list[float]]]:
        """Get all embeddings, optionally filtered by model."""
        if model_name:
            rows = self.conn.execute(
                "SELECT chunk_id, embedding FROM embeddings WHERE model_name = ?",
                (model_name,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT chunk_id, embedding FROM embeddings"
            ).fetchall()
        return [(row["chunk_id"], json.loads(row["embedding"])) for row in rows]

    def get_chunks_without_embeddings(self, model_name: str) -> list[Chunk]:
        """Find chunks that don't have embeddings for the given model."""
        rows = self.conn.execute("""
            SELECT c.* FROM chunks c
            LEFT JOIN embeddings e ON c.id = e.chunk_id AND e.model_name = ?
            WHERE e.chunk_id IS NULL
        """, (model_name,)).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def delete_embeddings_for_model(self, model_name: str) -> int:
        """Delete all embeddings for a specific model (useful when switching models)."""
        cursor = self.conn.execute(
            "DELETE FROM embeddings WHERE model_name = ?", (model_name,)
        )
        self.conn.commit()
        return cursor.rowcount


# ============================================================================
# Convenience function
# ============================================================================

def init_db(path: str | Path = "knowledge.db") -> KnowledgeDB:
    """Initialize and return a database connection."""
    return KnowledgeDB(path)


if __name__ == "__main__":
    # Quick sanity check
    db = init_db(":memory:")
    print("Schema initialized successfully")
    print(f"Stats: {db.stats()}")
    db.close()
