"""
Knowledge Harness MCP Server

Exposes the knowledge harness as MCP tools for Claude Code integration.

Usage:
    uv run python mcp_server.py

Configure in Claude Code settings:
    "mcpServers": {
        "knowledge": {
            "command": "uv",
            "args": ["run", "--directory", "/path/to/knowledge_base", "python", "mcp_server.py"],
        }
    }
"""

import json
import os
from pathlib import Path
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import harness components
from harness import KnowledgeHarness
from schema import TaskType, UsageOutcome
from context import ContextAssembler, ChunkDossier
from reflect import run_reflection, quick_insights


# ============================================================================
# Server Setup
# ============================================================================

# Database path - can be overridden with KB_DATABASE_PATH env var
DB_PATH = os.environ.get("KB_DATABASE_PATH", "./knowledge.db")

SERVER_INSTRUCTIONS = """
# Knowledge Base Usage Guidelines

You have access to a persistent knowledge base that learns from usage patterns.

## When to SEARCH (kb_search, kb_semantic_search)
- Before answering technical questions, search for relevant prior knowledge
- When the user asks about topics you've discussed before
- When you need context from previous conversations or ingested documents

## When to INGEST (kb_ingest)
- After fetching useful web content (articles, documentation, Stack Overflow answers)
- When the user shares valuable information worth preserving
- After synthesizing research or analysis that could be reused
- When you create documentation, guides, or explanations worth keeping

## When to RECORD (kb_record)
- After using retrieved knowledge to answer a question, record the outcome:
  - "win" = knowledge directly solved the problem
  - "partial" = helped but needed more
  - "miss" = retrieved but not useful
  - "misleading" = led to wrong conclusions
- Include notes on WHY it worked or failed - this is the most valuable signal
- Include the task_type to help the system learn what content works for what tasks

## Task Types (for search and record)
- factual_lookup: Checking facts, definitions, specifications
- implementation_howto: How to build/code something
- conceptual_understanding: Understanding concepts/theory
- debugging: Fixing errors or issues
- decision_support: Choosing between options
- exploratory_research: Open-ended exploration

## When to REFLECT (kb_reflect, kb_quick_insights)
- Use kb_quick_insights for a fast health check at the start of a session
- Use kb_reflect periodically to understand:
  - Which sources are reliable for which task types
  - Where knowledge gaps exist
  - What content is problematic and should be removed
  - What patterns lead to success
- After a series of misses or misleading results, reflect to diagnose issues

## Best Practices
- Be generous with ingestion - storage is cheap, forgetting is expensive
- Be honest with recording - misleading signals improve future retrieval
- Include context when recording - "helped debug async issue" is better than "helped"
- Search before you fetch - the answer might already be in the knowledge base
- Reflect periodically to identify what's working and what needs improvement
"""

server = Server("knowledge-harness", instructions=SERVER_INSTRUCTIONS)

# Lazy-loaded harness instance
_harness: Optional[KnowledgeHarness] = None


def get_harness() -> KnowledgeHarness:
    """Get or create the harness instance."""
    global _harness
    if _harness is None:
        _harness = KnowledgeHarness(DB_PATH)
    return _harness


# ============================================================================
# Tool Definitions
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available knowledge harness tools."""
    return [
        Tool(
            name="kb_search",
            description="""Search the knowledge base using multi-strategy retrieval (keyword, semantic, concept, usage history).

Returns ranked chunks with rich context for LLM reasoning:
- Individual usage traces with dates, task types, outcomes, and notes
- Task-type breakdown (success rates per task type)
- Functional profiles (consolidated learning about what content is good for)
- Source profiles (what this source is generally good for)

The usage history lets you reason about whether a chunk is likely to help with your current task.
Optionally provide task_type to boost chunks that have worked well for similar tasks.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query - can be a question or keywords"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 5)",
                        "default": 5
                    },
                    "task_type": {
                        "type": "string",
                        "description": "Type of task for context-aware retrieval",
                        "enum": ["factual_lookup", "implementation_howto", "conceptual_understanding",
                                "debugging", "decision_support", "exploratory_research", "other"]
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="kb_semantic_search",
            description="""Direct semantic similarity search (embedding-based only).

Use this when you want pure meaning-based matching without keyword or usage weighting.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="kb_ingest",
            description="""Add content to the knowledge base.

Content is chunked, summarized, and embedded automatically.
Use this to store useful information for future retrieval.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to ingest (markdown, text, code, etc.)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the document"
                    },
                    "source": {
                        "type": "string",
                        "description": "Source URL or identifier (default: 'manual')",
                        "default": "manual"
                    }
                },
                "required": ["content", "title"]
            }
        ),
        Tool(
            name="kb_record",
            description="""Record usage outcome for retrieved knowledge.

Call this after using knowledge to help the system learn what works.
This is how the knowledge base learns which content is good for which tasks.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of chunks that were used"
                    },
                    "outcome": {
                        "type": "string",
                        "description": "How useful was the knowledge",
                        "enum": ["win", "partial", "miss", "misleading"]
                    },
                    "context": {
                        "type": "string",
                        "description": "What were you trying to accomplish"
                    },
                    "task_type": {
                        "type": "string",
                        "description": "Type of task",
                        "enum": ["factual_lookup", "implementation_howto", "conceptual_understanding",
                                "debugging", "decision_support", "exploratory_research", "other"],
                        "default": "other"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Why did it work or fail? (optional but valuable)"
                    }
                },
                "required": ["chunk_ids", "outcome", "context"]
            }
        ),
        Tool(
            name="kb_stats",
            description="Get statistics about the knowledge base (document count, chunks, embeddings, etc.)",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="kb_get_chunk",
            description="Get full details about a specific chunk by ID, including usage history.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk ID"
                    }
                },
                "required": ["chunk_id"]
            }
        ),
        Tool(
            name="kb_list_documents",
            description="List all documents in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum documents to return (default: 20)",
                        "default": 20
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="kb_reflect",
            description="""Run a comprehensive reflection on the knowledge base.

Analyzes:
- Performance by task type (what's working, what isn't)
- Source quality assessments (which sources are reliable for what)
- Knowledge gaps (where we need more/better content)
- Problematic content (chunks that consistently mislead)
- Successful patterns (source+task combinations that work well)

Use this periodically to understand the health of the knowledge base and identify improvements.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="kb_quick_insights",
            description="""Get quick insights about the knowledge base without full reflection.

Returns:
- Size stats (documents, chunks, concepts)
- Task performance summary (win rates by task type)
- Flagged issues (low success rates, unused content)

Use this for a fast health check.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
    ]


# ============================================================================
# Tool Implementations
# ============================================================================

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    harness = get_harness()

    try:
        if name == "kb_search":
            return await handle_search(harness, arguments)
        elif name == "kb_semantic_search":
            return await handle_semantic_search(harness, arguments)
        elif name == "kb_ingest":
            return await handle_ingest(harness, arguments)
        elif name == "kb_record":
            return await handle_record(harness, arguments)
        elif name == "kb_stats":
            return await handle_stats(harness)
        elif name == "kb_get_chunk":
            return await handle_get_chunk(harness, arguments)
        elif name == "kb_list_documents":
            return await handle_list_documents(harness, arguments)
        elif name == "kb_reflect":
            return await handle_reflect(harness)
        elif name == "kb_quick_insights":
            return await handle_quick_insights(harness)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_search(harness: KnowledgeHarness, args: dict) -> list[TextContent]:
    """Handle kb_search tool."""
    import asyncio

    query = args["query"]
    limit = args.get("limit", 5)
    task_type_str = args.get("task_type")

    # Convert task_type string to enum if provided
    task_type = TaskType(task_type_str) if task_type_str else None

    # Run blocking search in thread pool to avoid blocking event loop
    results = await asyncio.to_thread(harness.search, query, limit=limit, task_type=task_type_str)

    if not results:
        return [TextContent(type="text", text="No results found.")]

    # Use ContextAssembler to build rich dossiers
    assembler = ContextAssembler(harness.db)

    output_lines = [f"Found {len(results)} results for '{query}':\n"]

    for i, r in enumerate(results, 1):
        # Build full dossier for this chunk
        dossier = assembler.assemble_dossier(r.chunk)

        output_lines.append(f"{'='*60}")
        output_lines.append(f"RESULT {i} (score: {r.score:.3f})")
        output_lines.append(f"{'='*60}")
        output_lines.append(f"Chunk ID: {r.chunk.id}")
        output_lines.append(f"Document: {r.document.title}")
        output_lines.append(f"Source: {r.document.source}")
        output_lines.append(f"Strategies: {', '.join(r.strategies)}")

        # Functional profile (consolidated learning)
        if dossier.functional_profile:
            output_lines.append(f"\nFUNCTIONAL PROFILE: {dossier.functional_profile}")

        # Source-level profile if available
        if dossier.source_functional_profile:
            output_lines.append(f"SOURCE PROFILE: {dossier.source_functional_profile}")

        # Individual usage traces - the key signal for LLM reasoning
        if dossier.usage_history:
            output_lines.append(f"\nUSAGE HISTORY ({len(dossier.usage_history)} traces):")
            for trace in dossier.usage_history[:5]:  # Show up to 5 recent traces
                outcome_symbol = {"win": "✓", "partial": "~", "miss": "✗", "misleading": "⚠"}
                symbol = outcome_symbol.get(trace["outcome"], "?")
                task = trace.get("task_type", "other")
                date = trace.get("timestamp", "")[:10]  # Just the date part
                notes = trace.get("notes") or trace.get("context", "")
                output_lines.append(f"  {date} [{task}] {symbol} {trace['outcome'].upper()}: {notes}")

        # Usage stats by task type
        if dossier.usage_stats.get("total", 0) > 0:
            output_lines.append(f"\nUSAGE STATS:")
            counts = dossier.usage_stats.get("counts", {})
            output_lines.append(f"  Overall: {counts.get('win', 0)} wins, {counts.get('partial', 0)} partial, "
                              f"{counts.get('miss', 0)} misses, {counts.get('misleading', 0)} misleading")

            # Breakdown by task type
            by_task = dossier.usage_stats_by_task.get("by_task", {})
            if by_task:
                output_lines.append("  By task type:")
                for task_name, stats in by_task.items():
                    if stats["total"] > 0:
                        output_lines.append(f"    {task_name}: {stats['success_rate']:.0%} success ({stats['total']} uses)")

        # Related concepts
        if dossier.concepts:
            output_lines.append(f"\nCONCEPTS: {', '.join(dossier.concepts)}")

        # The actual content
        output_lines.append(f"\nCONTENT:\n{r.chunk.content}")
        output_lines.append("")

    return [TextContent(type="text", text="\n".join(output_lines))]


async def handle_semantic_search(harness: KnowledgeHarness, args: dict) -> list[TextContent]:
    """Handle kb_semantic_search tool."""
    import asyncio

    query = args["query"]
    limit = args.get("limit", 5)

    if not harness.embeddings_enabled:
        return [TextContent(type="text", text="Semantic search not available (sentence-transformers not installed)")]

    # Run blocking search in thread pool to avoid blocking event loop
    results = await asyncio.to_thread(harness.semantic_search, query, limit=limit)

    if not results:
        return [TextContent(type="text", text="No results found.")]

    output_lines = [f"Semantic search results for '{query}':\n"]

    for chunk, score in results:
        doc = harness.get_document(chunk.document_id)
        output_lines.append(f"--- Score: {score:.3f} ---")
        output_lines.append(f"Chunk ID: {chunk.id}")
        output_lines.append(f"Document: {doc.title if doc else 'Unknown'}")
        output_lines.append(f"\nContent:\n{chunk.content}\n")

    return [TextContent(type="text", text="\n".join(output_lines))]


async def handle_ingest(harness: KnowledgeHarness, args: dict) -> list[TextContent]:
    """Handle kb_ingest tool."""
    import asyncio

    content = args["content"]
    title = args["title"]
    source = args.get("source", "manual")

    # Run blocking ingest in thread pool to avoid blocking event loop
    result = await asyncio.to_thread(harness.ingest_text, content, title=title, source=source)

    output = {
        "status": "success",
        "document_id": result.document_id,
        "chunks_created": len(result.chunk_ids),
        "chunk_ids": result.chunk_ids,
        "concepts_extracted": len(result.concept_ids),
    }

    return [TextContent(type="text", text=f"Ingested successfully:\n{json.dumps(output, indent=2)}")]


async def handle_record(harness: KnowledgeHarness, args: dict) -> list[TextContent]:
    """Handle kb_record tool."""
    from record import record_usage

    chunk_ids = args["chunk_ids"]
    outcome = args["outcome"]
    context = args["context"]
    task_type = args.get("task_type", "other")
    notes = args.get("notes")

    trace_id = record_usage(
        harness.db,
        chunk_ids=chunk_ids,
        context=context,
        outcome=outcome,
        task_type=task_type,
        notes=notes
    )

    return [TextContent(type="text", text=f"Recorded usage trace: {trace_id}")]


async def handle_stats(harness: KnowledgeHarness) -> list[TextContent]:
    """Handle kb_stats tool."""
    stats = harness.stats()
    stats["embeddings_enabled"] = harness.embeddings_enabled
    return [TextContent(type="text", text=json.dumps(stats, indent=2))]


async def handle_get_chunk(harness: KnowledgeHarness, args: dict) -> list[TextContent]:
    """Handle kb_get_chunk tool."""
    chunk_id = args["chunk_id"]
    info = harness.chunk_info(chunk_id)

    if "error" in info:
        return [TextContent(type="text", text=f"Error: {info['error']}")]

    return [TextContent(type="text", text=json.dumps(info, indent=2, default=str))]


async def handle_list_documents(harness: KnowledgeHarness, args: dict) -> list[TextContent]:
    """Handle kb_list_documents tool."""
    limit = args.get("limit", 20)
    docs = harness.list_documents(limit=limit)

    output_lines = [f"Documents ({len(docs)}):\n"]
    for doc in docs:
        chunks = harness.get_chunks(doc.id)
        output_lines.append(f"- {doc.title}")
        output_lines.append(f"  ID: {doc.id}")
        output_lines.append(f"  Source: {doc.source}")
        output_lines.append(f"  Chunks: {len(chunks)}")
        output_lines.append(f"  Status: {doc.status.value}")
        output_lines.append("")

    return [TextContent(type="text", text="\n".join(output_lines))]


async def handle_reflect(harness: KnowledgeHarness) -> list[TextContent]:
    """Handle kb_reflect tool."""
    report = run_reflection(harness.db)
    return [TextContent(type="text", text=report.format_report())]


async def handle_quick_insights(harness: KnowledgeHarness) -> list[TextContent]:
    """Handle kb_quick_insights tool."""
    insights = quick_insights(harness.db)
    return [TextContent(type="text", text=json.dumps(insights, indent=2))]


# ============================================================================
# Main
# ============================================================================

def _preload():
    """Pre-load harness and embedding model before accepting requests."""
    import sys
    print("Pre-loading harness...", file=sys.stderr)
    harness = get_harness()

    # Trigger model loading by doing a dummy embed
    if harness.embeddings_enabled and harness._embedder:
        print("Pre-loading embedding model...", file=sys.stderr)
        harness._embedder.embed("warmup")
        print("Model ready.", file=sys.stderr)


async def main():
    """Run the MCP server."""
    # Pre-load before starting to avoid blocking during requests
    _preload()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
