# Usage Augmented Retrieval 

> **Note**: This is a proof of concept, vibe-coded with Claude. It works, but expect rough edges.

A usage-augmented retrieval (UAR) system for Claude Code. Unlike traditional RAG which is stateless, this system learns from usage outcomes to improve retrieval over time.

## Core Idea

### The Problem with Traditional RAG

Standard RAG is stateless. It retrieves content based on semantic similarity, but it doesn't learn from outcomes. A chunk that has consistently led you astray gets retrieved just as readily as one that has saved you dozens of times—as long as the embeddings are close enough.

The result: sarcastic Reddit comments presented as factual claims, outdated Stack Overflow answers weighted equally with current documentation, and no sense of "this source is great for practical tips but terrible for authoritative statements."

### What UAR Does Differently

RAG asks: *"What content is semantically similar to this query?"*

UAR asks: *"What content is semantically similar AND has actually helped with tasks like this before?"*

Every time you retrieve and use a piece of knowledge, the system records:
- What task were you trying to accomplish? (debugging, factual lookup, conceptual understanding, etc.)
- Did it help, partially help, miss entirely, or actively mislead?
- Why?

This creates a usage history for each chunk. Over time, the system learns what each piece of knowledge is actually good for—not through a reliability score, but through **role-casting**: the same content might be excellent for debugging but useless for theoretical understanding.

Rather than pre-computing a score that bakes in these signals, the raw usage history is surfaced directly to the LLM. It sees "this chunk won 3x for debugging, missed 2x for factual lookup" and reasons about whether to trust it for the current task. The judgment stays with the reasoning system.

## Features

- **Multi-strategy retrieval**: keyword, semantic (MiniLM embeddings), concept, usage history, recency
- **Usage tracking**: Record outcomes (win/partial/miss/misleading) per task type
- **Role-casting**: Same content can be good for debugging but bad for factual lookup—tracked separately
- **LLM-in-the-loop**: Usage history is surfaced directly, letting the LLM reason about trust
- **Local embeddings**: Uses `all-MiniLM-L6-v2` via sentence-transformers, no external API needed
- **SQLite storage**: Single-file database with WAL mode for concurrent access

## Installation

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).

```bash
cd knowledge_base
uv sync
```

## Usage with Claude Code

Add to your `~/.claude.json` for global access:

```json
{
  "mcpServers": {
    "knowledge": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/knowledge_base", "python", "mcp_server.py"]
    }
  }
}
```

Restart Claude Code. The knowledge tools will be available.

## MCP Tools

### Core Tools

| Tool | Description |
|------|-------------|
| `kb_search` | Multi-strategy search with rich usage history |
| `kb_semantic_search` | Pure embedding similarity search |
| `kb_ingest` | Add content (auto-chunked and embedded) |
| `kb_record` | Record usage outcome for retrieved chunks |
| `kb_reflect` | Comprehensive analysis of what's working |
| `kb_quick_insights` | Fast health check |

### Utility Tools (not advertised to LLM)

These are available but not included in the server instructions—useful for debugging and admin:

| Tool | Description |
|------|-------------|
| `kb_stats` | Database statistics |
| `kb_get_chunk` | Get chunk details with full usage history |
| `kb_list_documents` | List all documents |
| `kb_delete_document` | Remove a document and all its chunks |
| `kb_consolidate` | Convert usage traces into functional profiles |

## Task Types

When recording usage, specify the task type:

- `factual_lookup` - Checking facts, definitions, specifications
- `implementation_howto` - How to build/code something
- `conceptual_understanding` - Understanding concepts/theory
- `debugging` - Fixing errors or issues
- `decision_support` - Choosing between options
- `exploratory_research` - Open-ended exploration

## Usage Outcomes

- `win` - Directly solved the problem
- `partial` - Helped but needed more
- `miss` - Retrieved but not useful
- `misleading` - Led to wrong conclusions

## Example Workflow

1. **Search** for relevant knowledge before answering a question
2. **Use** the retrieved content to help with the task
3. **Record** the outcome with task type and notes
4. **Over time**, retrieval improves based on what actually worked

## Recommended Usage: Research Sub-Agent

For best results, use a sub-agent pattern for research tasks:

```
┌─────────────────────────────────────────────────────────────┐
│                      MAIN CLAUDE                            │
│  - Focuses on user conversation                             │
│  - Spawns research agent when needed                        │
│  - Provides feedback on results                             │
└─────────────────────────────────────────────────────────────┘
        │                                   ▲
        │ query + context                   │ results + feedback request
        ▼                                   │
┌─────────────────────────────────────────────────────────────┐
│                    RESEARCH SUB-AGENT                       │
│  - Searches KB first                                        │
│  - Searches web if KB insufficient                          │
│  - Ingests valuable findings                                │
│  - Records outcome based on main Claude's feedback          │
└─────────────────────────────────────────────────────────────┘
```

This pattern:
- Keeps the main conversation focused
- Ensures the feedback loop is always closed
- Lets the sub-agent handle KB complexity

The MCP server instructions include a template for spawning research sub-agents.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     THE UAR LOOP                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   RETRIEVE ──────► USE ──────► RECORD                   │
│       ▲                           │                     │
│       │                           │                     │
│       │         CONSOLIDATE ◄─────┘                     │
│       │              │                                  │
│       └──────────────┘                                  │
│                                                         │
│   Retrieval is augmented by usage history.              │
│   Usage history consolidates into functional profiles.  │
│   Profiles inform future retrieval.                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Files

- `schema.py` - Data models and SQLite database
- `harness.py` - Unified API
- `ingest.py` - Content ingestion pipeline
- `retrieve.py` - Multi-strategy retrieval
- `record.py` - Usage tracking
- `embeddings.py` - Local MiniLM embeddings
- `context.py` - Chunk dossier assembly for LLM reasoning
- `consolidate.py` - Episodic to semantic consolidation
- `reflect.py` - Knowledge base analysis
- `mcp_server.py` - MCP server for Claude Code integration

## License

MIT
