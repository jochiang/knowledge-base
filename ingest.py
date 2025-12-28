"""
Knowledge Harness - Ingestion Pipeline

Handles: content detection, chunking, summarization, concept extraction.
This module is designed to be called by Claude Code, where Claude itself
provides the LLM-powered operations (summarization, concept extraction).
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable

from schema import (
    KnowledgeDB, Document, Chunk, Concept, Link,
    ContentType, ChunkType, EntityType, ConceptType
)


# ============================================================================
# Content Type Detection
# ============================================================================

def detect_content_type(source: str, content: str) -> ContentType:
    """Infer content type from source path/URL and content analysis."""
    source_lower = source.lower()
    
    # Check file extension
    if source_lower.endswith(('.py', '.js', '.ts', '.go', '.rs', '.java', '.cpp', '.c', '.h')):
        return ContentType.CODE
    if source_lower.endswith(('.md', '.txt', '.rst')):
        # Could be article or note - check content
        if len(content) < 2000:
            return ContentType.NOTE
        return ContentType.ARTICLE
    if source_lower.endswith('.pdf'):
        # PDFs are often papers
        if 'abstract' in content.lower()[:2000]:
            return ContentType.PAPER
        return ContentType.ARTICLE
    
    # Check URL patterns
    if 'arxiv.org' in source_lower or 'doi.org' in source_lower:
        return ContentType.PAPER
    if 'wikipedia.org' in source_lower:
        return ContentType.REFERENCE
    
    # Content heuristics
    content_lower = content.lower()
    if content_lower.startswith('```') or 'def ' in content[:500] or 'function ' in content[:500]:
        return ContentType.CODE
    if 'abstract' in content_lower[:1000] and 'references' in content_lower[-5000:]:
        return ContentType.PAPER
    
    # Default based on length
    if len(content) < 1500:
        return ContentType.NOTE
    return ContentType.ARTICLE


# ============================================================================
# Chunking Strategies
# ============================================================================

@dataclass
class ChunkResult:
    content: str
    chunk_type: ChunkType
    metadata: dict = None


def chunk_by_paragraphs(content: str, min_chunk_size: int = 200, max_chunk_size: int = 1500) -> list[ChunkResult]:
    """Split content by paragraphs, merging small ones."""
    paragraphs = re.split(r'\n\s*\n', content)
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        if len(current_chunk) + len(para) < max_chunk_size:
            current_chunk += ("\n\n" if current_chunk else "") + para
        else:
            if current_chunk:
                chunks.append(ChunkResult(
                    content=current_chunk,
                    chunk_type=ChunkType.NARRATIVE
                ))
            current_chunk = para
    
    if current_chunk:
        chunks.append(ChunkResult(
            content=current_chunk,
            chunk_type=ChunkType.NARRATIVE
        ))
    
    return chunks


def chunk_by_sections(content: str) -> list[ChunkResult]:
    """Split content by markdown/header sections."""
    # Match markdown headers
    section_pattern = r'^(#{1,6})\s+(.+)$'
    lines = content.split('\n')
    
    chunks = []
    current_section = ""
    current_header = None
    
    for line in lines:
        header_match = re.match(section_pattern, line)
        if header_match:
            # Save previous section
            if current_section.strip():
                chunks.append(ChunkResult(
                    content=current_section.strip(),
                    chunk_type=ChunkType.NARRATIVE,
                    metadata={"header": current_header}
                ))
            current_section = line + "\n"
            current_header = header_match.group(2)
        else:
            current_section += line + "\n"
    
    # Don't forget last section
    if current_section.strip():
        chunks.append(ChunkResult(
            content=current_section.strip(),
            chunk_type=ChunkType.NARRATIVE,
            metadata={"header": current_header}
        ))
    
    return chunks if chunks else chunk_by_paragraphs(content)


def chunk_code(content: str) -> list[ChunkResult]:
    """Split code by function/class definitions."""
    # This is a simplified version - in practice you'd want language-specific parsing
    
    # Try to split on common function/class patterns
    patterns = [
        r'(?=^(?:async\s+)?(?:def|class)\s+\w+)',  # Python
        r'(?=^(?:export\s+)?(?:async\s+)?(?:function|class|const|let|var)\s+\w+)',  # JS/TS
        r'(?=^(?:pub\s+)?(?:fn|struct|impl|enum)\s+)',  # Rust
        r'(?=^(?:func|type|struct)\s+)',  # Go
    ]
    
    for pattern in patterns:
        chunks = re.split(pattern, content, flags=re.MULTILINE)
        chunks = [c.strip() for c in chunks if c.strip()]
        if len(chunks) > 1:
            return [ChunkResult(content=c, chunk_type=ChunkType.CODE) for c in chunks]
    
    # Fallback: split by blank lines but keep larger chunks
    return chunk_by_paragraphs(content, min_chunk_size=100, max_chunk_size=2000)


def chunk_content(content: str, content_type: ContentType) -> list[ChunkResult]:
    """Route to appropriate chunking strategy."""
    if content_type == ContentType.CODE:
        return chunk_code(content)
    elif content_type in (ContentType.PAPER, ContentType.ARTICLE):
        # Try section-based first, fall back to paragraphs
        chunks = chunk_by_sections(content)
        if len(chunks) <= 1:
            chunks = chunk_by_paragraphs(content)
        return chunks
    elif content_type == ContentType.NOTE:
        # Notes are often kept whole or lightly chunked
        if len(content) < 1500:
            return [ChunkResult(content=content, chunk_type=ChunkType.NARRATIVE)]
        return chunk_by_paragraphs(content, min_chunk_size=300, max_chunk_size=2000)
    else:
        return chunk_by_paragraphs(content)


# ============================================================================
# Ingestion Pipeline
# ============================================================================

@dataclass
class IngestResult:
    document_id: str
    chunk_ids: list[str]
    concept_ids: list[str]
    link_ids: list[str]


class IngestPipeline:
    """
    Orchestrates the ingestion process.

    LLM-powered steps (summarization, concept extraction) are provided as
    callbacks so Claude Code can plug in its own implementations.
    """

    def __init__(
        self,
        db: KnowledgeDB,
        summarize_chunk: Callable[[str], str] = None,
        summarize_document: Callable[[str, list[str]], str] = None,
        extract_concepts: Callable[[str], list[dict]] = None,
        extract_claims: Callable[[str], list[str]] = None,
        embedder: "LocalEmbedder" = None,  # Optional embedder for eager embedding
    ):
        self.db = db
        self._summarize_chunk = summarize_chunk or (lambda x: None)
        self._summarize_document = summarize_document or (lambda x, y: None)
        self._extract_concepts = extract_concepts or (lambda x: [])
        self._extract_claims = extract_claims or (lambda x: [])
        self._embedder = embedder
    
    def ingest(
        self,
        source: str,
        content: str,
        title: Optional[str] = None,
        content_type: Optional[ContentType] = None,
    ) -> IngestResult:
        """
        Full ingestion pipeline.
        
        1. Detect content type (if not provided)
        2. Create document record
        3. Chunk content
        4. Summarize chunks
        5. Summarize document
        6. Extract concepts
        7. Create links
        """
        # Step 1: Content type
        if content_type is None:
            content_type = detect_content_type(source, content)
        
        # Step 2: Create document (without summary yet)
        if title is None:
            title = self._infer_title(source, content)
        
        doc = Document.create(
            source=source,
            content_type=content_type,
            title=title,
            raw_content=content
        )
        self.db.insert_document(doc)
        
        # Step 3: Chunk
        chunk_results = chunk_content(content, content_type)
        
        # Step 4: Create chunks and summarize
        chunk_ids = []
        chunk_summaries = []
        for i, cr in enumerate(chunk_results):
            summary = self._summarize_chunk(cr.content)
            
            chunk = Chunk.create(
                document_id=doc.id,
                content=cr.content,
                position=i,
                summary=summary,
                chunk_type=cr.chunk_type
            )
            self.db.insert_chunk(chunk)
            chunk_ids.append(chunk.id)
            if summary:
                chunk_summaries.append(summary)
        
        # Step 5: Document summary
        doc_summary = self._summarize_document(content, chunk_summaries)
        if doc_summary:
            self.db.conn.execute(
                "UPDATE documents SET top_summary = ? WHERE id = ?",
                (doc_summary, doc.id)
            )
            self.db.conn.commit()
        
        # Step 6: Extract key claims
        claims = self._extract_claims(content)
        if claims:
            import json
            self.db.conn.execute(
                "UPDATE documents SET key_claims = ? WHERE id = ?",
                (json.dumps(claims), doc.id)
            )
            self.db.conn.commit()
        
        # Step 7: Extract and link concepts
        concept_ids = []
        link_ids = []
        
        # Extract concepts from full document
        raw_concepts = self._extract_concepts(content)
        for rc in raw_concepts:
            name = rc.get("name", "").strip().lower()
            if not name:
                continue
            
            concept = self.db.get_or_create_concept(
                name,
                description=rc.get("description"),
                concept_type=ConceptType(rc.get("type", "topic"))
            )
            concept_ids.append(concept.id)
            
            # Link concept to document
            link = Link.create(
                source_type=EntityType.DOCUMENT,
                source_id=doc.id,
                target_type=EntityType.CONCEPT,
                target_id=concept.id,
                relation="about"
            )
            self.db.insert_link(link)
            link_ids.append(link.id)
        
        # Link concepts to relevant chunks (simplified: link to all for now)
        # A more sophisticated version would do per-chunk extraction
        for chunk_id in chunk_ids:
            for concept_id in concept_ids:
                self.db.link_chunk_to_concept(chunk_id, concept_id, weight=0.5)

        # Step 8: Embed chunks (if embedder provided)
        if self._embedder is not None:
            chunks_to_embed = [self.db.get_chunk(cid) for cid in chunk_ids]
            self._embedder.embed_chunks(chunks_to_embed, skip_existing=False)

        return IngestResult(
            document_id=doc.id,
            chunk_ids=chunk_ids,
            concept_ids=concept_ids,
            link_ids=link_ids
        )
    
    def _infer_title(self, source: str, content: str) -> str:
        """Extract or generate a title."""
        # Try to get from source path
        if '/' in source or '\\' in source:
            return Path(source).stem
        
        # Try first line if it looks like a title
        first_line = content.strip().split('\n')[0].strip()
        if first_line.startswith('#'):
            return first_line.lstrip('#').strip()
        if len(first_line) < 100 and not first_line.endswith('.'):
            return first_line
        
        # Fallback
        return source[:50] if len(source) <= 50 else source[:47] + "..."


# ============================================================================
# Convenience functions for Claude Code
# ============================================================================

def quick_ingest(db: KnowledgeDB, source: str, content: str, title: str = None) -> IngestResult:
    """
    Ingest without LLM-powered features (no summarization/concept extraction).
    Useful for bulk imports or when you want to add LLM processing later.
    """
    pipeline = IngestPipeline(db)
    return pipeline.ingest(source, content, title)


def ingest_file(db: KnowledgeDB, filepath: str | Path) -> IngestResult:
    """Convenience function to ingest a file from disk."""
    path = Path(filepath)
    content = path.read_text(encoding='utf-8')
    pipeline = IngestPipeline(db)
    return pipeline.ingest(str(path), content, path.stem)


# ============================================================================
# Example LLM callbacks for Claude Code
# ============================================================================

# These are templates - in actual use, Claude Code would implement these
# by calling itself or another model.

CHUNK_SUMMARY_PROMPT = """Summarize this text chunk in 1-2 sentences. Focus on the key information or argument.

CHUNK:
{chunk}

SUMMARY:"""

DOCUMENT_SUMMARY_PROMPT = """Based on the full content and these chunk summaries, write a 2-3 sentence summary of the entire document.

CHUNK SUMMARIES:
{summaries}

FULL CONTENT (for reference):
{content}

DOCUMENT SUMMARY:"""

CONCEPT_EXTRACTION_PROMPT = """Extract the key concepts from this text. For each concept, provide:
- name: a short canonical name (lowercase)
- type: one of (topic, entity, method, claim, question)
- description: a brief description if not obvious

Return as JSON array.

TEXT:
{content}

CONCEPTS:"""

CLAIMS_EXTRACTION_PROMPT = """Extract the main claims or assertions made in this text. 
List each as a single clear sentence.

TEXT:
{content}

CLAIMS:"""


if __name__ == "__main__":
    # Demo: ingest some sample content
    db = KnowledgeDB(":memory:")
    
    sample_content = """
# Understanding Vector Databases

Vector databases are specialized systems designed to store and query high-dimensional vectors efficiently.

## Why Vectors?

Traditional databases excel at exact matches and range queries. But when you need to find "similar" items—
documents with related meanings, images with similar content, or products a user might like—you need 
a different approach.

Embedding models convert complex data (text, images, audio) into dense vectors where geometric 
proximity corresponds to semantic similarity. A vector database makes querying these embeddings fast.

## Key Concepts

**Approximate Nearest Neighbor (ANN)**: Finding the exact closest vectors is expensive at scale. 
ANN algorithms like HNSW or IVF trade a small amount of accuracy for massive speed improvements.

**Indexing**: Vector databases build specialized index structures that partition the vector space,
allowing queries to skip most of the data.

## When to Use

Vector databases shine when you need:
- Semantic search (finding documents by meaning, not keywords)
- Recommendation systems
- Deduplication or clustering
- Any application where "similarity" matters more than exact matching
"""

    result = quick_ingest(db, "vector_databases.md", sample_content)
    
    print(f"Ingested document: {result.document_id}")
    print(f"Created {len(result.chunk_ids)} chunks")
    print(f"\nChunks:")
    for chunk in db.get_chunks_for_document(result.document_id):
        print(f"  [{chunk.position}] {chunk.content[:80]}...")
    
    db.close()
