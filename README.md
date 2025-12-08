# semantic-memory

Local semantic memory with PGlite + pgvector. Budget Qdrant that runs anywhere Bun runs.

## Why

You want semantic search for your AI agents but don't want to run a vector database server. This gives you:

- **Zero infrastructure** - PGlite is Postgres compiled to WASM, runs in-process
- **Real vector search** - pgvector with HNSW indexes, not some janky cosine similarity loop
- **Collection-based organization** - Different collections for different contexts (codebase, research, notes)
- **Configurable tool descriptions** - The Qdrant MCP pattern: same tool, different behaviors via env vars
- **Effect-TS** - Proper error handling, resource management, composable services

## Install

```bash
# npm/bun/pnpm
npm install semantic-memory

# Need Ollama for embeddings
brew install ollama
ollama pull mxbai-embed-large
```

## CLI

```bash
# Via npx
npx semantic-memory store "The auth flow uses JWT tokens stored in httpOnly cookies"
npx semantic-memory find "how does authentication work"

# Or install globally
npm install -g semantic-memory
semantic-memory store "React component patterns" --collection code
semantic-memory find "components" --collection code

# Full-text search (no embeddings)
semantic-memory find "JWT" --fts

# Add metadata
semantic-memory store "API rate limits are 100 req/min" --metadata '{"source":"docs","priority":"high"}'

# List, get, delete
semantic-memory list
semantic-memory get <id>
semantic-memory delete <id>

# Stats
semantic-memory stats
```

## Collections for Context

Collections let you organize memories by purpose. The collection name carries semantic meaning:

```bash
# Codebase analysis - store patterns, architecture notes, API quirks
semantic-memory store "Auth uses httpOnly JWT cookies with 7-day refresh" --collection codebase
semantic-memory store "The useOptimistic hook requires a reducer pattern" --collection codebase
semantic-memory find "authentication" --collection codebase

# Research/learning - concepts, connections, questions
semantic-memory store "Effect-TS uses generators for async, not Promises" --collection research
semantic-memory find "effect async patterns" --collection research

# Project onboarding - gotchas, tribal knowledge, "why is it like this"
semantic-memory store "Don't use React.memo on components with children - causes stale closures" --collection gotchas
semantic-memory find "performance issues" --collection gotchas

# Personal knowledge - decisions, preferences, breakthroughs
semantic-memory store "Prefer composition over inheritance for React components" --collection decisions
semantic-memory find "react patterns" --collection decisions
```

Search across all collections or within one:

```bash
# Search everything
semantic-memory find "authentication"

# Search specific collection
semantic-memory find "authentication" --collection codebase
```

## The Qdrant Pattern

The killer feature: **tool descriptions are configurable**.

Same semantic memory, different agent behaviors:

```bash
# Codebase assistant - searches before generating, stores patterns found
TOOL_STORE_DESCRIPTION="Store code patterns, architecture decisions, and API quirks discovered while analyzing the codebase. Include file paths and context." \
TOOL_FIND_DESCRIPTION="Search codebase knowledge. Query BEFORE making changes to understand existing patterns." \
semantic-memory find "auth patterns"

# Research assistant - accumulates and connects ideas
TOOL_STORE_DESCRIPTION="Store concepts, insights, and connections between ideas. Include source references." \
TOOL_FIND_DESCRIPTION="Search research notes. Use to find related concepts and prior findings." \
semantic-memory find "async patterns"

# Onboarding assistant - captures tribal knowledge
TOOL_STORE_DESCRIPTION="Store gotchas, workarounds, and 'why is it like this' explanations. Future devs will thank you." \
TOOL_FIND_DESCRIPTION="Search for known issues and gotchas. Check BEFORE debugging to avoid known pitfalls." \
semantic-memory find "common mistakes"
```

The description tells the LLM _when_ and _how_ to use the tool. Change the description, change the behavior. No code changes.

## OpenCode Integration

Drop this in `~/.config/opencode/tool/semantic-memory.ts`:

```typescript
import { tool } from "@opencode-ai/plugin";
import { $ } from "bun";

// Rich descriptions that shape agent behavior
// Override via env vars for different contexts
const STORE_DESC =
  process.env.TOOL_STORE_DESCRIPTION ||
  "Persist important discoveries, decisions, and learnings for future sessions. Use for: architectural decisions, debugging breakthroughs, user preferences, project-specific patterns. Include context about WHY something matters.";
const FIND_DESC =
  process.env.TOOL_FIND_DESCRIPTION ||
  "Search your persistent memory for relevant context. Query BEFORE making architectural decisions, when hitting familiar-feeling bugs, or when you need project history. Returns semantically similar memories ranked by relevance.";

async function run(args: string[]): Promise<string> {
  const result = await $`npx semantic-memory ${args}`.text();
  return result.trim();
}

export const store = tool({
  description: STORE_DESC,
  args: {
    information: tool.schema.string().describe("The information to store"),
    collection: tool.schema
      .string()
      .optional()
      .describe("Collection name (e.g., 'codebase', 'research', 'gotchas')"),
    metadata: tool.schema
      .string()
      .optional()
      .describe("Optional JSON metadata"),
  },
  async execute({ information, collection, metadata }) {
    const args = ["store", information];
    if (collection) args.push("--collection", collection);
    if (metadata) args.push("--metadata", metadata);
    return run(args);
  },
});

export const find = tool({
  description: FIND_DESC,
  args: {
    query: tool.schema.string().describe("Natural language search query"),
    collection: tool.schema
      .string()
      .optional()
      .describe("Collection to search (omit for all)"),
    limit: tool.schema
      .number()
      .optional()
      .describe("Max results (default: 10)"),
  },
  async execute({ query, collection, limit }) {
    const args = ["find", query];
    if (collection) args.push("--collection", collection);
    if (limit) args.push("--limit", String(limit));
    return run(args);
  },
});
```

### Per-Project Configuration

For project-specific behavior, create a wrapper script or use direnv:

```bash
# .envrc (with direnv)
export TOOL_STORE_DESCRIPTION="Store patterns found in this Next.js codebase. Include file paths."
export TOOL_FIND_DESCRIPTION="Search codebase patterns. Check before implementing new features."
```

Or create project-specific OpenCode tools that hardcode the collection:

```typescript
// .opencode/tool/codebase-memory.ts
export const remember = tool({
  description: "Store a pattern or insight about this codebase",
  args: { info: tool.schema.string() },
  async execute({ info }) {
    return $`npx semantic-memory store ${info} --collection ${process.cwd()}`.text();
  },
});
```

## Configuration

All via environment variables:

| Variable                 | Default                  | Description                    |
| ------------------------ | ------------------------ | ------------------------------ |
| `SEMANTIC_MEMORY_PATH`   | `~/.semantic-memory`     | Where to store the database    |
| `OLLAMA_HOST`            | `http://localhost:11434` | Ollama API endpoint            |
| `OLLAMA_MODEL`           | `mxbai-embed-large`      | Embedding model (1024 dims)    |
| `COLLECTION_NAME`        | `default`                | Default collection             |
| `TOOL_STORE_DESCRIPTION` | (see code)               | MCP tool description for store |
| `TOOL_FIND_DESCRIPTION`  | (see code)               | MCP tool description for find  |

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Ollama    │────▶│   PGlite    │────▶│  pgvector   │
│ (embeddings)│     │ (WASM PG)   │     │ (HNSW idx)  │
└─────────────┘     └─────────────┘     └─────────────┘
      │                    │                   │
      │              memories table      memory_embeddings
      │              - id                - memory_id (FK)
      │              - content           - embedding vector(1024)
      │              - metadata (JSONB)
      │              - collection
      └──────────────────────────────────────────┘
                    cosine similarity search
```

- **Ollama** generates embeddings locally with `mxbai-embed-large` (1024 dimensions)
- **PGlite** is Postgres compiled to WASM - no server, runs in your process
- **pgvector** provides real vector operations with HNSW indexes for fast approximate nearest neighbor search
- **Effect-TS** handles errors, retries, and resource cleanup properly

## Use Cases

### Codebase Analysis

Store patterns, architecture decisions, and API quirks as you explore a new codebase. Query before making changes.

### Session Memory

Remember facts across AI sessions. No more re-explaining context every conversation.

### Documentation Cache

Pre-load docs into a collection, search before hallucinating answers.

### Research Assistant

Accumulate findings, connect ideas across sources, build up domain knowledge.

### Onboarding Knowledge Base

Capture the "why" behind decisions, known gotchas, and tribal knowledge for future team members.

## License

MIT
