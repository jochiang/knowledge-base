# Knowledge Management Harness

## Design Principles

1. **Content-agnostic**: Works with articles, papers, notes, code, conversations, whatever
2. **Hierarchical summarization**: Multiple levels of abstraction, not just chunks
3. **Usage-aware**: Tracks how knowledge gets applied, not just what it contains
4. **Graph-structured**: Relationships are first-class citizens, not afterthoughts
5. **Evolvable**: Schema and heuristics can change as we learn what works
6. **Role-casting over scoring**: Knowledge isn't "reliable" or "unreliable"—it has functional niches where it excels or fails. A sarcastic Reddit comment is perfect for understanding community sentiment, terrible for factual claims.
7. **LLM-in-the-loop reasoning**: Rather than pre-computing reliability scores, we surface usage history as context and let the LLM reason about what to trust and why.
8. **Episodic-to-semantic consolidation**: Individual usage traces are episodic memory. Over time, they consolidate into prose assessments—semantic memory about what knowledge is good for.

---

## Core Entities

### Document
The atomic unit of ingestion. Could be an article, a PDF, a markdown file, a conversation transcript.

```
document:
  id: uuid
  source: string (filepath, URL, manual entry)
  source_profile_id: fk (optional, links to source profile)
  content_type: string (article, paper, note, code, conversation, reference)
  title: string
  raw_content: text
  top_summary: text (1-3 sentences, the "what is this")
  key_claims: text[] (extractable assertions, optional)
  ingested_at: timestamp
  last_accessed: timestamp
  access_count: int
  status: enum (active, archived, superseded)
```

### Source Profile
Captures learned understanding of a source's functional strengths and weaknesses. Not a reliability score—a capability profile.

```
source_profile:
  id: uuid
  domain: string (reddit.com, arxiv.org, internal-wiki, etc.)
  source_type: enum (forum, academic, news, docs, social, conversation, personal_notes)
  
  # LLM-generated prose assessment, updated during consolidation
  # Example: "Strong for practical implementation gotchas and community sentiment. 
  #           Weak for authoritative factual claims. Watch for sarcasm and hyperbole."
  functional_profile: text
  
  # Structured summary derived from traces (for quick filtering)
  strengths: text[] (e.g., ["practical_howto", "community_sentiment", "current_events"])
  weaknesses: text[] (e.g., ["factual_accuracy", "theoretical_depth"])
  
  # Raw counts for analysis (is the functional profile actually predictive?)
  trace_counts_by_task: json  # {"implementation": {"win": 5, "miss": 1}, "factual": {"win": 1, "misleading": 3}}
  
  created_at: timestamp
  updated_at: timestamp
```

### Chunk
A semantically coherent segment of a document. Granularity varies by content type.

```
chunk:
  id: uuid
  document_id: fk
  content: text
  position: int (order within document)
  summary: text (1 sentence)
  chunk_type: string (narrative, argument, data, code, definition, example)
  token_count: int
  
  # LLM-generated prose assessment of what this chunk is good for
  # Starts null, populated during consolidation once enough traces exist
  # Example: "Useful for understanding the practical gotchas of async Python.
  #           Too implementation-specific for conceptual understanding of concurrency."
  functional_profile: text
  
  # Flags for retrieval-time reasoning
  needs_context: bool (true if chunk doesn't stand alone well)
  rhetorical_mode: string (explanatory, argumentative, anecdotal, sarcastic, instructional)
```

### Concept
An extracted idea, topic, entity, or theme. Lives independently of documents—multiple docs can reference the same concept.

```
concept:
  id: uuid
  name: string (canonical form)
  aliases: string[] (alternate phrasings)
  description: text (working definition, can evolve)
  concept_type: enum (topic, entity, method, claim, question)
  
  # Concepts can also have functional profiles
  # Example for "async/await": "Well-covered for Python and JavaScript implementation.
  #           Sparse on theoretical foundations. Conflicting advice on error handling patterns."
  functional_profile: text
  gap_notes: text (what's missing in our knowledge about this concept)
  
  created_at: timestamp
  updated_at: timestamp
```

### Usage Trace
The episodic memory of how knowledge was applied. This is the raw signal.

```
usage_trace:
  id: uuid
  chunk_ids: uuid[] (which chunks were used)
  session_id: string (groups traces from same work session)
  
  # What were we trying to do? This is crucial for learning functional niches.
  task_type: enum (factual_lookup, implementation_howto, conceptual_understanding, 
                   opinion_gathering, decision_support, debugging, exploratory_research,
                   creative_inspiration, other)
  context_summary: text (freeform description of the task)
  query: text (the actual question/task, if applicable)
  
  outcome: enum (win, partial, miss, misleading)
  
  # Why did it work or fail? This is the richest signal.
  notes: text (freeform—capture reasoning, not just verdict)
  
  timestamp: timestamp
```

**Outcome definitions:**
- **win**: Chunk directly contributed to a good result
- **partial**: Chunk was relevant but not sufficient
- **miss**: Chunk was retrieved but turned out irrelevant
- **misleading**: Chunk's phrasing/content led us astray (important negative signal)

**Task type matters because:**
A chunk might be a "win" for implementation_howto and "misleading" for factual_lookup. The same content has different utility depending on what you're trying to do. This is the core of role-casting.

### Link
Explicit relationships between any entities. Typed and weighted.

```
link:
  id: uuid
  source_type: enum (document, chunk, concept)
  source_id: uuid
  target_type: enum (document, chunk, concept)
  target_id: uuid
  relation: string (see relation taxonomy below)
  weight: float (0-1, strength/confidence)
  created_at: timestamp
  created_by: enum (auto, manual, consolidated)
```

**Relation taxonomy (starter set, extensible):**
- `related_to`: generic association
- `supports`: evidence for
- `contradicts`: tension or conflict
- `supersedes`: newer version of same idea
- `elaborates`: deeper dive on
- `applies`: theoretical → practical connection
- `derived_from`: intellectual lineage
- `same_as`: concept aliasing
- `complements`: useful together (learned from co-retrieval patterns)
- `better_for`: this chunk is better than that chunk for X task type

---

## Operations

### 1. Ingest
Takes raw content, produces structured knowledge.

```
ingest(source, content_type=auto) -> document_id

Steps:
1. Detect/confirm content type
2. Find or create source profile for the domain/source
3. Chunk content (strategy varies by type)
4. Generate chunk summaries
5. Detect rhetorical mode for each chunk (explanatory, sarcastic, etc.)
6. Generate document summary
7. Extract concepts (new or link to existing)
8. Auto-generate links (within doc, to existing concepts)
9. Store everything
```

**Chunking strategies:**
- **article/paper**: By section/paragraph, respecting semantic boundaries
- **code**: By function/class/logical block
- **conversation**: By turn or topic shift
- **note**: Often kept whole or split by headers

### 2. Retrieve
Multi-strategy retrieval, returns ranked results with context for LLM reasoning.

```
retrieve(query, task_type=inferred, limit=10) -> enriched_results

Strategies (combined):
- semantic: Embedding similarity against chunks
- concept: Find chunks sharing concepts with query's extracted concepts
- usage: Find chunks that were useful in similar past task contexts
- graph: Walk from seed nodes, surface connected content
- temporal: Recency weighting

Results include:
- chunk content and summary
- chunk's functional_profile (if populated)
- parent document context
- source profile summary
- why this was retrieved (which strategy, what score)
- FULL usage history for this chunk (for LLM reasoning)
```

**The key insight:** Rather than pre-computing a reliability score, we surface the usage history directly:

```
Retrieved: chunk_abc123 from reddit.com/r/python

Usage history:
- 2024-03-15 [implementation_howto] → WIN: "Explained the exact async gotcha I was hitting"  
- 2024-04-02 [conceptual_understanding] → MISS: "Too implementation-specific, needed theory"
- 2024-04-18 [debugging] → WIN: "Stack trace example matched my issue"
- 2024-05-01 [factual_lookup] → MISLEADING: "Upvoted answer was actually wrong, corrected in replies"

Functional profile: "Reliable for practical debugging and implementation patterns. 
                     Not suitable for factual claims or theoretical understanding."

Source profile (reddit.com/r/python): "Strong community knowledge for practical Python. 
                                        Watch for outdated answers and sarcasm."
```

The LLM sees this and reasons about whether to trust the chunk for the current task.

### 3. Record
Log usage after a work session.

```
record(session_id, chunk_ids, task_type, context_summary, outcome, notes=None)

Called after using retrieved knowledge to capture what worked.
Can be explicit ("that was helpful") or inferred from conversation.
```

**Important:** Capture not just the outcome, but *why*. The notes field is where the real learning happens.

### 4. Consolidate
Periodic maintenance pass. This is where episodic memory becomes semantic memory.

```
consolidate() -> report

Steps:
1. For chunks with enough traces (e.g., 5+), generate/update functional_profile
   - LLM reads all traces and writes prose: "This chunk is good for X, bad for Y, because Z"
   - Individual traces can then be archived (keep counts, drop details)
   
2. For source profiles with enough data, update functional_profile
   - Aggregate across all chunks from this source
   - "Reddit/r/python is excellent for implementation help, unreliable for factual claims"

3. For concepts with enough coverage, update functional_profile and gap_notes
   - "We have strong practical knowledge of async/await, weak on theory"
   - "No knowledge about error handling in async contexts" (gap)

4. Discover co-retrieval patterns → create 'complements' links
   - Chunks that frequently succeed together get linked
   
5. Discover task-type preferences → create 'better_for' links
   - "For debugging tasks, chunk A outperforms chunk B on similar queries"

6. Surface contradictions for review
   - Two chunks with 'supports' relations to opposite claims

7. Surface lonely content (ingested but never retrieved or used)

8. Surface stale content (old, not accessed, possibly outdated)
```

### 5. Reflect
Higher-order synthesis (run occasionally, not every session).

```
reflect(scope=all) -> insights

Questions to answer:
- What task types do we serve well vs poorly?
- Which sources are pulling their weight? Which are noise?
- What concepts keep coming up that we have gaps in?
- Are there chunks we keep retrieving that consistently fail? (retire them)
- What unexpected connections have emerged from co-retrieval patterns?
- How has our understanding of [concept] evolved over time?
```

---

## Retrieval: Scoring vs Reasoning

**Old model (scoring):**
```
score = w1*semantic + w2*concept + w3*usage_success_rate + w4*recency
```

This is still useful for initial candidate selection (get top 50 chunks), but final ranking should involve LLM reasoning over usage history.

**New model (reasoning):**
1. Score-based retrieval gets candidate chunks
2. For each candidate, we assemble a "chunk dossier":
   - Content and summary
   - Functional profile (if exists)
   - Source profile
   - Recent usage traces relevant to current task type
3. LLM reasons about which chunks to actually use and how much to trust them
4. This reasoning becomes part of the session record

**Context budget optimization:**

Early on: Show full usage history (sparse, fits in context)

Later (rich history): Show only:
- Functional profile (prose summary)
- Most recent 2-3 traces matching current task type
- Aggregate stats ("7 wins, 2 misses for implementation tasks")

Even later: Just the functional profile. The traces have been "compiled" into understanding.

---

## The Learning Loop

```
┌─────────────────────────────────────────────────────────────┐
│                         INGEST                               │
│   Raw content → Chunks → Summaries → Concepts → Links        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        RETRIEVE                              │
│   Query + Task Type → Candidates → Enriched with history     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                          USE                                 │
│   LLM reasons about what to trust → Applies knowledge        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        RECORD                                │
│   Outcome + Task Type + Notes → Usage Trace (episodic)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      CONSOLIDATE                             │
│   Traces → Functional Profiles (semantic)                    │
│   Patterns → Links                                           │
│   Gaps → Explicit gap notes                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              └──────────── back to RETRIEVE ──┘
```

The system learns:
- What each chunk is good for (functional profile)
- What each source is good for (source profile)  
- What we know well vs have gaps in (concept profiles)
- What works together (co-retrieval links)

---

## File Structure

```
knowledge_harness/
├── knowledge.db          # SQLite database
├── embeddings/           # Cached embeddings (optional, for semantic search)
├── config.yaml           # Tunable parameters
├── src/
│   ├── __init__.py
│   ├── schema.py         # Database schema and models
│   ├── ingest.py         # Ingestion pipeline
│   ├── retrieve.py       # Retrieval strategies
│   ├── record.py         # Usage trace logging
│   ├── consolidate.py    # Episodic → semantic consolidation
│   ├── reflect.py        # Higher-order analysis
│   ├── context.py        # Assembles chunk dossiers for LLM reasoning
│   └── prompts.py        # Prompt templates for LLM operations
├── harness.py            # Unified interface
└── ARCHITECTURE.md       # This file
```

---

## Open Questions

1. **Consolidation triggers**: After N traces? On a schedule? Manual?
2. **Profile versioning**: Should we track how functional profiles evolve?
3. **Cross-source learning**: Can we learn "forums in general" patterns vs just "reddit.com"?
4. **Negative space**: How explicitly do we track "searched for X, found nothing useful"?
5. **Trace compression**: When is it safe to drop individual traces and keep only profiles?
6. **Task type inference**: Can we reliably infer task type from query, or must it be explicit?

---

## Example: Role-Casting in Action

**Query:** "Why is my async Python code deadlocking?"
**Inferred task type:** debugging

**Retrieved chunk from Reddit:**
```
Content: "Ran into this exact issue. The problem is you're calling await inside 
          a sync callback. You need to use asyncio.create_task() or run it in 
          an executor. Took me 3 hours to figure this out lol"

Functional profile: "Practical debugging gold for async Python. Author clearly 
                     hit this in production. Don't cite as authoritative—it's 
                     one person's experience."

Usage history (debugging tasks): 3 wins, 0 misses
Usage history (factual_lookup): 0 wins, 1 misleading

Source profile: "r/python is strong for 'I hit this problem' pattern matching. 
                 Verify solutions before adopting—upvotes don't guarantee correctness."
```

**LLM reasoning:** "This chunk has a strong track record for debugging tasks, which matches my current task. The rhetorical mode is anecdotal but the pattern matches the user's issue. I'll use this as a strong lead but verify the solution."

This is role-casting: the chunk isn't generically "reliable" or "unreliable"—it's being cast into a role (debugging helper) that it's suited for.
