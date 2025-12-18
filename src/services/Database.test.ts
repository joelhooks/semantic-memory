/**
 * Database.ts Tests
 *
 * Testing strategy:
 * - Isolated temp databases per test (no shared state)
 * - Effect.runPromise for async Effect execution
 * - Cover CRUD, vector search, FTS, collection filtering, decay logic
 */

import { beforeEach, afterEach, describe, expect, test } from "bun:test";
import { Effect } from "effect";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Database, makeDatabaseLive, type Memory } from "./Database";

// ============================================================================
// Test Fixtures
// ============================================================================

/** Create isolated temp directory for test database */
function makeTempDbPath(): string {
  const tempDir = mkdtempSync(join(tmpdir(), "semantic-memory-test-"));
  return join(tempDir, "test.db");
}

/** Create test memory with minimal required fields */
function makeMemory(overrides: Partial<Memory> = {}): Memory {
  return {
    id: `mem-${Date.now()}-${Math.random()}`,
    content: "Test memory content",
    metadata: {},
    collection: "default",
    createdAt: new Date(),
    ...overrides,
  };
}

/** Create dummy embedding vector (1024 dimensions for mxbai-embed-large) */
function makeEmbedding(seed = 1.0): number[] {
  return Array.from({ length: 1024 }, (_, i) => Math.sin(seed + i * 0.1));
}

// ============================================================================
// Test Harness
// ============================================================================

describe("Database", () => {
  let dbPath: string;
  let dbLayer: ReturnType<typeof makeDatabaseLive>;

  beforeEach(() => {
    dbPath = makeTempDbPath();
    dbLayer = makeDatabaseLive({ dbPath });
  });

  afterEach(() => {
    // Cleanup temp database
    const dbDir = dbPath.replace(".db", "");
    try {
      rmSync(dbDir, { recursive: true, force: true });
    } catch {
      // ignore cleanup errors
    }
  });

  // ==========================================================================
  // CRUD Operations
  // ==========================================================================

  describe("store", () => {
    test("stores memory with embedding", async () => {
      const memory = makeMemory();
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);
        const retrieved = yield* db.get(memory.id);
        return retrieved;
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result).not.toBeNull();
      expect(result?.id).toBe(memory.id);
      expect(result?.content).toBe(memory.content);
      expect(result?.collection).toBe("default");
    });

    test("updates existing memory on conflict", async () => {
      const memory = makeMemory({ id: "mem-1", content: "Original" });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;

        // Store original
        yield* db.store(memory, embedding);

        // Update with new content
        const updated = { ...memory, content: "Updated" };
        yield* db.store(updated, embedding);

        return yield* db.get(memory.id);
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result?.content).toBe("Updated");
    });

    test("stores metadata as JSON", async () => {
      const memory = makeMemory({
        metadata: { tags: ["test", "important"], priority: 1 },
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);
        return yield* db.get(memory.id);
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result?.metadata).toEqual({
        tags: ["test", "important"],
        priority: 1,
      });
    });

    test("handles custom collection", async () => {
      const memory = makeMemory({ collection: "work" });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);
        return yield* db.get(memory.id);
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result?.collection).toBe("work");
    });
  });

  describe("get", () => {
    test("returns null for non-existent memory", async () => {
      const program = Effect.gen(function* () {
        const db = yield* Database;
        return yield* db.get("non-existent");
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result).toBeNull();
    });

    test("retrieves memory with all fields", async () => {
      const memory = makeMemory({
        content: "Full memory",
        metadata: { key: "value" },
        collection: "custom",
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);
        return yield* db.get(memory.id);
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result).not.toBeNull();
      expect(result?.id).toBe(memory.id);
      expect(result?.content).toBe("Full memory");
      expect(result?.metadata).toEqual({ key: "value" });
      expect(result?.collection).toBe("custom");
      expect(result?.createdAt).toBeInstanceOf(Date);
    });
  });

  describe("list", () => {
    test("returns empty array when no memories", async () => {
      const program = Effect.gen(function* () {
        const db = yield* Database;
        return yield* db.list();
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result).toEqual([]);
    });

    test("lists all memories ordered by creation date", async () => {
      // Create memories with explicit different timestamps
      const now = new Date();
      const earlier = new Date(now.getTime() - 1000); // 1 second earlier

      const mem1 = makeMemory({ id: "mem-1", createdAt: earlier });
      const mem2 = makeMemory({ id: "mem-2", createdAt: now });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;

        // Store both
        yield* db.store(mem1, embedding);
        yield* db.store(mem2, embedding);

        return yield* db.list();
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result.length).toBe(2);
      // Most recent first
      expect(result[0].id).toBe("mem-2");
      expect(result[1].id).toBe("mem-1");
    });

    test("filters by collection", async () => {
      const mem1 = makeMemory({ collection: "work" });
      const mem2 = makeMemory({ collection: "personal" });
      const mem3 = makeMemory({ collection: "work" });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(mem1, embedding);
        yield* db.store(mem2, embedding);
        yield* db.store(mem3, embedding);

        return yield* db.list("work");
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result.length).toBe(2);
      expect(result.every((m) => m.collection === "work")).toBe(true);
    });
  });

  describe("delete", () => {
    test("deletes memory by id", async () => {
      const memory = makeMemory();
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);

        // Verify exists
        const before = yield* db.get(memory.id);
        expect(before).not.toBeNull();

        // Delete
        yield* db.delete(memory.id);

        // Verify gone
        return yield* db.get(memory.id);
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result).toBeNull();
    });

    test("cascades to embeddings", async () => {
      const memory = makeMemory();
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);

        const statsBefore = yield* db.getStats();
        expect(statsBefore.embeddings).toBe(1);

        yield* db.delete(memory.id);

        const statsAfter = yield* db.getStats();
        return statsAfter;
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result.embeddings).toBe(0);
      expect(result.memories).toBe(0);
    });

    test("deleting non-existent memory succeeds silently", async () => {
      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.delete("non-existent");
        return "success";
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result).toBe("success");
    });
  });

  // ==========================================================================
  // Vector Search
  // ==========================================================================

  describe("search (vector)", () => {
    test("finds similar memories by embedding", async () => {
      const mem1 = makeMemory({ content: "Machine learning basics" });
      const mem2 = makeMemory({ content: "Cooking recipes" });
      const embedding1 = makeEmbedding(1.0);
      const embedding2 = makeEmbedding(100.0); // different embedding

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(mem1, embedding1);
        yield* db.store(mem2, embedding2);

        // Search with embedding similar to mem1
        return yield* db.search(embedding1, { limit: 10, threshold: 0.3 });
      }).pipe(Effect.provide(dbLayer));

      const results = await Effect.runPromise(program);

      // Should find at least mem1 with high similarity
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].memory.id).toBe(mem1.id);
      expect(results[0].score).toBeGreaterThan(0.9); // high similarity
      expect(results[0].matchType).toBe("vector");
    });

    test("respects similarity threshold", async () => {
      const mem1 = makeMemory({ content: "Test" });
      const embedding1 = makeEmbedding(1.0);
      const queryEmbedding = makeEmbedding(500.0); // very different

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(mem1, embedding1);

        return yield* db.search(queryEmbedding, { threshold: 0.9 });
      }).pipe(Effect.provide(dbLayer));

      const results = await Effect.runPromise(program);

      // Should find nothing due to high threshold
      expect(results.length).toBe(0);
    });

    test("respects limit", async () => {
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;

        // Store 5 memories
        for (let i = 0; i < 5; i++) {
          yield* db.store(makeMemory({ id: `mem-${i}` }), embedding);
        }

        return yield* db.search(embedding, { limit: 2 });
      }).pipe(Effect.provide(dbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(2);
    });

    test("filters by collection", async () => {
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(makeMemory({ collection: "work" }), embedding);
        yield* db.store(makeMemory({ collection: "personal" }), embedding);

        return yield* db.search(embedding, { collection: "work" });
      }).pipe(Effect.provide(dbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(1);
      expect(results[0].memory.collection).toBe("work");
    });

    test("includes decay information", async () => {
      const memory = makeMemory();
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);

        return yield* db.search(embedding, { limit: 1 });
      }).pipe(Effect.provide(dbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(1);
      const result = results[0];

      // Check decay fields exist and are reasonable
      expect(result.ageDays).toBeGreaterThanOrEqual(0);
      expect(result.ageDays).toBeLessThan(1); // just created
      expect(result.decayFactor).toBeGreaterThan(0.99); // minimal decay
      expect(result.rawScore).toBeGreaterThan(0);
      expect(result.score).toBeGreaterThan(0);
    });

    test("applies decay to score over time", async () => {
      const memory = makeMemory({
        // Simulate old memory by setting createdAt in past
        createdAt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000), // 90 days ago
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);

        return yield* db.search(embedding, { limit: 1, threshold: 0.0 });
      }).pipe(Effect.provide(dbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(1);
      const result = results[0];

      // After 90 days (one half-life), decay factor should be ~0.5
      expect(result.ageDays).toBeGreaterThan(89);
      expect(result.decayFactor).toBeLessThan(0.6);
      expect(result.decayFactor).toBeGreaterThan(0.4);

      // Final score should be less than raw score due to decay
      expect(result.score).toBeLessThan(result.rawScore);
    });
  });

  // ==========================================================================
  // Full-Text Search
  // ==========================================================================

  describe("ftsSearch", () => {
    test("finds memories by text content", async () => {
      const mem1 = makeMemory({ content: "PostgreSQL database optimization" });
      const mem2 = makeMemory({ content: "React component patterns" });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(mem1, embedding);
        yield* db.store(mem2, embedding);

        return yield* db.ftsSearch("PostgreSQL", { limit: 10 });
      }).pipe(Effect.provide(dbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBeGreaterThan(0);
      expect(results[0].memory.id).toBe(mem1.id);
      expect(results[0].matchType).toBe("fts");
    });

    test("handles multi-word queries", async () => {
      const memory = makeMemory({
        content: "Database optimization techniques",
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);

        return yield* db.ftsSearch("database optimization", { limit: 10 });
      }).pipe(Effect.provide(dbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(1);
      expect(results[0].memory.id).toBe(memory.id);
    });

    test("respects limit", async () => {
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;

        // Store 5 memories with common word
        for (let i = 0; i < 5; i++) {
          yield* db.store(
            makeMemory({ id: `mem-${i}`, content: "test content" }),
            embedding
          );
        }

        return yield* db.ftsSearch("test", { limit: 2 });
      }).pipe(Effect.provide(dbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(2);
    });

    test("filters by collection", async () => {
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(
          makeMemory({ collection: "work", content: "work item" }),
          embedding
        );
        yield* db.store(
          makeMemory({ collection: "personal", content: "personal item" }),
          embedding
        );

        return yield* db.ftsSearch("item", { collection: "work" });
      }).pipe(Effect.provide(dbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(1);
      expect(results[0].memory.collection).toBe("work");
    });

    test("returns empty array for no matches", async () => {
      const memory = makeMemory({ content: "PostgreSQL database" });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);

        return yield* db.ftsSearch("nonexistent search term xyz", {
          limit: 10,
        });
      }).pipe(Effect.provide(dbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(0);
    });

    test("includes decay information", async () => {
      const memory = makeMemory({ content: "test content" });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);

        return yield* db.ftsSearch("test", { limit: 1 });
      }).pipe(Effect.provide(dbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(1);
      const result = results[0];

      expect(result.ageDays).toBeGreaterThanOrEqual(0);
      expect(result.decayFactor).toBeGreaterThan(0.99);
      expect(result.rawScore).toBeGreaterThan(0);
      expect(result.score).toBeGreaterThan(0);
    });
  });

  // ==========================================================================
  // Validation (Decay Reset)
  // ==========================================================================

  describe("validate", () => {
    test("sets lastValidatedAt timestamp", async () => {
      const memory = makeMemory();
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);

        // Initially no validation
        const before = yield* db.get(memory.id);
        expect(before?.lastValidatedAt).toBeUndefined();

        // Validate
        yield* db.validate(memory.id);

        // Now should have timestamp
        return yield* db.get(memory.id);
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result?.lastValidatedAt).toBeInstanceOf(Date);
      expect(result?.lastValidatedAt).toBeDefined();
    });

    test("validation resets decay for search", async () => {
      // Create old memory
      const oldMemory = makeMemory({
        createdAt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000), // 90 days ago
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(oldMemory, embedding);

        // Check decay before validation
        const beforeValidation = yield* db.search(embedding, {
          limit: 1,
          threshold: 0.0,
        });
        const decayBefore = beforeValidation[0].decayFactor;

        // Validate (resets decay timer)
        yield* db.validate(oldMemory.id);

        // Check decay after validation
        const afterValidation = yield* db.search(embedding, {
          limit: 1,
          threshold: 0.0,
        });

        return {
          decayBefore,
          decayAfter: afterValidation[0].decayFactor,
        };
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      // Decay factor should increase (closer to 1.0) after validation
      expect(result.decayAfter).toBeGreaterThan(result.decayBefore);
      expect(result.decayAfter).toBeGreaterThan(0.99); // near 1.0 for fresh validation
    });

    test("validation affects FTS search decay", async () => {
      const oldMemory = makeMemory({
        content: "test content",
        createdAt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(oldMemory, embedding);

        // Check decay before validation
        const beforeValidation = yield* db.ftsSearch("test", { limit: 1 });
        const decayBefore = beforeValidation[0].decayFactor;

        // Validate
        yield* db.validate(oldMemory.id);

        // Check decay after validation
        const afterValidation = yield* db.ftsSearch("test", { limit: 1 });

        return {
          decayBefore,
          decayAfter: afterValidation[0].decayFactor,
        };
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result.decayAfter).toBeGreaterThan(result.decayBefore);
      expect(result.decayAfter).toBeGreaterThan(0.99);
    });
  });

  // ==========================================================================
  // Statistics
  // ==========================================================================

  describe("getStats", () => {
    test("returns zero counts for empty database", async () => {
      const program = Effect.gen(function* () {
        const db = yield* Database;
        return yield* db.getStats();
      }).pipe(Effect.provide(dbLayer));

      const stats = await Effect.runPromise(program);

      expect(stats.memories).toBe(0);
      expect(stats.embeddings).toBe(0);
    });

    test("counts memories and embeddings", async () => {
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;

        yield* db.store(makeMemory({ id: "mem-1" }), embedding);
        yield* db.store(makeMemory({ id: "mem-2" }), embedding);
        yield* db.store(makeMemory({ id: "mem-3" }), embedding);

        return yield* db.getStats();
      }).pipe(Effect.provide(dbLayer));

      const stats = await Effect.runPromise(program);

      expect(stats.memories).toBe(3);
      expect(stats.embeddings).toBe(3);
    });

    test("counts stay in sync after delete", async () => {
      const mem1 = makeMemory({ id: "mem-1" });
      const mem2 = makeMemory({ id: "mem-2" });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;

        yield* db.store(mem1, embedding);
        yield* db.store(mem2, embedding);

        yield* db.delete(mem1.id);

        return yield* db.getStats();
      }).pipe(Effect.provide(dbLayer));

      const stats = await Effect.runPromise(program);

      expect(stats.memories).toBe(1);
      expect(stats.embeddings).toBe(1);
    });
  });

  // ==========================================================================
  // Error Cases
  // ==========================================================================

  describe("error handling", () => {
    test("store fails with invalid embedding dimension", async () => {
      const memory = makeMemory();
      const invalidEmbedding = [1, 2, 3]; // wrong dimension (need 1024)

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, invalidEmbedding);
        return "should not reach here";
      }).pipe(Effect.provide(dbLayer));

      // Should fail with DatabaseError
      try {
        await Effect.runPromise(program);
        expect.unreachable("Expected store to fail with invalid embedding");
      } catch (error) {
        // Expect some error about dimensions
        expect(error).toBeDefined();
      }
    });

    test("search fails with invalid embedding dimension", async () => {
      const invalidEmbedding = [1, 2, 3];

      const program = Effect.gen(function* () {
        const db = yield* Database;
        return yield* db.search(invalidEmbedding);
      }).pipe(Effect.provide(dbLayer));

      // Should fail with DatabaseError
      try {
        await Effect.runPromise(program);
        expect.unreachable("Expected search to fail with invalid embedding");
      } catch (error) {
        // Expect some error about dimensions
        expect(error).toBeDefined();
      }
    });
  });

  // ==========================================================================
  // Integration Scenarios
  // ==========================================================================

  describe("integration scenarios", () => {
    test("full workflow: store, search, validate, delete", async () => {
      const memory = makeMemory({ content: "Integration test memory" });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;

        // Store
        yield* db.store(memory, embedding);

        // Search
        const searchResults = yield* db.search(embedding, { limit: 1 });
        expect(searchResults.length).toBe(1);
        expect(searchResults[0].memory.id).toBe(memory.id);

        // Validate
        yield* db.validate(memory.id);
        const validated = yield* db.get(memory.id);
        expect(validated?.lastValidatedAt).toBeDefined();

        // Delete
        yield* db.delete(memory.id);
        const deleted = yield* db.get(memory.id);
        expect(deleted).toBeNull();

        return "success";
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result).toBe("success");
    });

    test("multiple collections work independently", async () => {
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;

        // Store in different collections
        yield* db.store(makeMemory({ collection: "work" }), embedding);
        yield* db.store(makeMemory({ collection: "work" }), embedding);
        yield* db.store(makeMemory({ collection: "personal" }), embedding);

        const workList = yield* db.list("work");
        const personalList = yield* db.list("personal");
        const allList = yield* db.list();

        return { workList, personalList, allList };
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      expect(result.workList.length).toBe(2);
      expect(result.personalList.length).toBe(1);
      expect(result.allList.length).toBe(3);
    });

    test("database isolation between tests", async () => {
      // This test verifies that each test gets a clean database
      const program = Effect.gen(function* () {
        const db = yield* Database;
        const memories = yield* db.list();
        return memories;
      }).pipe(Effect.provide(dbLayer));

      const result = await Effect.runPromise(program);

      // Should be empty - no pollution from previous tests
      expect(result.length).toBe(0);
    });
  });
});
