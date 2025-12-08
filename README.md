# 🧠 semantic-memory

Local semantic memory with PGlite + pgvector. Budget Qdrant that runs anywhere Bun runs.

## Why

You want semantic search for your AI agents but don't want to run a vector database server. This gives you:

- **Zero infrastructure** - PGlite is Postgres compiled to WASM, runs in-process
- **Real vector search** - pgvector with HNSW indexes, not some janky cosine similarity loop
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

## The Qdrant Pattern

The killer feature: **tool descriptions are configurable**.

Same semantic memory, different agent behaviors:

```bash
# Code assistant - searches before generating, stores after
TOOL_STORE_DESCRIPTION="Store code patterns after the user accepts generated code" \
TOOL_FIND_DESCRIPTION="Search for similar code patterns. Use BEFORE writing any new code." \
semantic-memory find "auth patterns"

# Meeting notes assistant
TOOL_STORE_DESCRIPTION="Remember important points from meetings" \
TOOL_FIND_DESCRIPTION="Recall what was discussed in previous meetings" \
semantic-memory find "Q4 planning"

# Documentation helper
TOOL_STORE_DESCRIPTION="Store documentation snippets for reference" \
TOOL_FIND_DESCRIPTION="Search documentation. Always check before answering questions." \
semantic-memory find "rate limits"
```

The description tells the LLM _when_ and _how_ to use the tool. Change the description, change the behavior. No code changes.

## OpenCode Integration

Drop this in `~/.config/opencode/tool/semantic-memory.ts`:

```typescript
import { tool } from "@opencode-ai/plugin";
import { $ } from "bun";

const STORE_DESC =
  process.env.TOOL_STORE_DESCRIPTION ||
  "Store information for later semantic retrieval";
const FIND_DESC =
  process.env.TOOL_FIND_DESCRIPTION ||
  "Search for relevant information using semantic similarity";

async function run(args: string[]): Promise<string> {
  const result = await $`npx semantic-memory ${args}`.text();
  return result.trim();
}

export const store = tool({
  description: STORE_DESC,
  args: {
    information: tool.schema.string(),
    collection: tool.schema.string().optional(),
  },
  async execute({ information, collection }) {
    const args = ["store", information];
    if (collection) args.push("--collection", collection);
    return run(args);
  },
});

export const find = tool({
  description: FIND_DESC,
  args: {
    query: tool.schema.string(),
    limit: tool.schema.number().optional(),
  },
  async execute({ query, limit }) {
    const args = ["find", query];
    if (limit) args.push("--limit", String(limit));
    return run(args);
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

- **Code pattern memory** - Store patterns you generate, search before writing new code
- **Session memory** - Remember facts across AI sessions
- **Documentation cache** - Pre-load docs, search before hallucinating
- **Meeting notes** - Store and recall discussion points
- **Research assistant** - Accumulate and query findings

## License

MIT
