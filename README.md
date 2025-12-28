# Knowledge Harness

A usage-augmented retrieval (UAR) system for Claude Code. Unlike traditional RAG which is stateless, this system learns from usage outcomes to improve retrieval over time.

## Core Idea

RAG asks: *"What content is semantically similar to this query?"*

UAR asks: *"What content is semantically similar AND has actually helped with tasks like this before?"*

The system tracks what worked, what didn't, and why—then surfaces that signal for LLM reasoning.

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

| Tool | Description |
|------|-------------|
| `kb_search` | Multi-strategy search with rich usage history |
| `kb_semantic_search` | Pure embedding similarity search |
| `kb_ingest` | Add content (auto-chunked and embedded) |
| `kb_record` | Record usage outcome for retrieved chunks |
| `kb_stats` | Database statistics |
| `kb_get_chunk` | Get chunk details with full usage history |
| `kb_list_documents` | List all documents |
| `kb_reflect` | Comprehensive analysis of what's working |
| `kb_quick_insights` | Fast health check |

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
