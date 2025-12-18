/**
 * Ollama Service Tests
 *
 * Tests for embedding generation and health checks.
 * Uses mocked fetch to avoid requiring Ollama to be running.
 */

import { describe, test, expect, beforeEach, mock } from "bun:test";
import { Effect } from "effect";
import { Ollama, makeOllamaLive } from "./Ollama.js";
import { OllamaError, MemoryConfig } from "../types.js";

// ============================================================================
// Test Fixtures
// ============================================================================

const testConfig = new MemoryConfig({
  dataPath: "/tmp/test",
  ollamaModel: "mxbai-embed-large",
  ollamaHost: "http://localhost:11434",
  toolStoreDescription: "test",
  toolFindDescription: "test",
  defaultCollection: "default",
});

const mockEmbedding = Array(1024).fill(0.1); // mxbai-embed-large is 1024 dims

const mockEmbeddingResponse = {
  embedding: mockEmbedding,
};

const mockTagsResponse = {
  models: [{ name: "mxbai-embed-large:latest" }, { name: "llama2:latest" }],
};

// ============================================================================
// Mock Setup
// ============================================================================

let fetchMock: unknown;

beforeEach(() => {
  // Reset fetch mock before each test
  (fetchMock as any)?.mockRestore?.();
});

// ============================================================================
// Tests: embed()
// ============================================================================

describe("Ollama.embed", () => {
  test("generates embeddings from text", async () => {
    // ARRANGE: Mock successful embedding response
    fetchMock = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockEmbeddingResponse),
      } as Response)
    );
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.embed("hello world");
    });

    // ACT
    const result = await Effect.runPromise(program.pipe(Effect.provide(layer)));

    // ASSERT
    expect(result).toEqual(mockEmbedding);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:11434/api/embeddings",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "mxbai-embed-large",
          prompt: "hello world",
        }),
      })
    );
  });

  test("handles empty text input", async () => {
    // ARRANGE
    fetchMock = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockEmbeddingResponse),
      } as Response)
    );
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.embed("");
    });

    // ACT
    const result = await Effect.runPromise(program.pipe(Effect.provide(layer)));

    // ASSERT
    expect(result).toEqual(mockEmbedding);
  });

  test("fails with OllamaError on connection failure", async () => {
    // ARRANGE: Mock network error
    fetchMock = mock(() =>
      Promise.reject(new Error("ECONNREFUSED: Connection refused"))
    );
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.embed("test");
    });

    // ACT & ASSERT
    const result = await Effect.runPromise(
      program.pipe(
        Effect.provide(layer),
        Effect.flip // Convert failure to success so we can inspect the error
      )
    );

    expect(result).toBeInstanceOf(OllamaError);
    expect((result as OllamaError).reason).toContain("Connection failed");
  });

  test("fails with OllamaError on HTTP error response", async () => {
    // ARRANGE: Mock 500 error
    fetchMock = mock(() =>
      Promise.resolve({
        ok: false,
        status: 500,
        text: () => Promise.resolve("Internal server error"),
      } as Response)
    );
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.embed("test");
    });

    // ACT & ASSERT
    const result = await Effect.runPromise(
      program.pipe(Effect.provide(layer), Effect.flip)
    );

    expect(result).toBeInstanceOf(OllamaError);
    expect((result as OllamaError).reason).toContain("Internal server error");
  });

  test("fails with OllamaError on invalid JSON response", async () => {
    // ARRANGE: Mock malformed JSON
    fetchMock = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.reject(new Error("Unexpected token")),
      } as Response)
    );
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.embed("test");
    });

    // ACT & ASSERT
    const result = await Effect.runPromise(
      program.pipe(Effect.provide(layer), Effect.flip)
    );

    expect(result).toBeInstanceOf(OllamaError);
    expect((result as OllamaError).reason).toContain("Invalid JSON response");
  });

  test("retries on transient failures", async () => {
    // ARRANGE: Mock failing twice, then succeeding
    let callCount = 0;
    fetchMock = mock(() => {
      callCount++;
      if (callCount < 3) {
        return Promise.reject(new Error("Temporary failure"));
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockEmbeddingResponse),
      } as Response);
    });
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.embed("test");
    });

    // ACT
    const result = await Effect.runPromise(program.pipe(Effect.provide(layer)));

    // ASSERT
    expect(result).toEqual(mockEmbedding);
    expect(callCount).toBe(3); // Initial + 2 retries
  });
});

// ============================================================================
// Tests: embedBatch()
// ============================================================================

describe("Ollama.embedBatch", () => {
  test("generates embeddings for multiple texts", async () => {
    // ARRANGE
    fetchMock = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockEmbeddingResponse),
      } as Response)
    );
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.embedBatch(["text1", "text2", "text3"]);
    });

    // ACT
    const result = await Effect.runPromise(program.pipe(Effect.provide(layer)));

    // ASSERT
    expect(result).toHaveLength(3);
    expect(result[0]).toEqual(mockEmbedding);
    expect(result[1]).toEqual(mockEmbedding);
    expect(result[2]).toEqual(mockEmbedding);
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  test("handles empty array input", async () => {
    // ARRANGE
    fetchMock = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockEmbeddingResponse),
      } as Response)
    );
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.embedBatch([]);
    });

    // ACT
    const result = await Effect.runPromise(program.pipe(Effect.provide(layer)));

    // ASSERT
    expect(result).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  test("respects custom concurrency limit", async () => {
    // ARRANGE: Track concurrent calls
    let maxConcurrent = 0;
    let currentConcurrent = 0;

    fetchMock = mock(async () => {
      currentConcurrent++;
      maxConcurrent = Math.max(maxConcurrent, currentConcurrent);

      // Simulate some work
      await new Promise((resolve) => setTimeout(resolve, 10));

      currentConcurrent--;
      return {
        ok: true,
        json: () => Promise.resolve(mockEmbeddingResponse),
      } as Response;
    });
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      // Request 10 embeddings with concurrency of 2
      return yield* ollama.embedBatch(
        Array(10).fill("test"),
        2 // concurrency
      );
    });

    // ACT
    await Effect.runPromise(program.pipe(Effect.provide(layer)));

    // ASSERT
    expect(maxConcurrent).toBeLessThanOrEqual(2);
  });

  test("fails if any batch item fails", async () => {
    // ARRANGE: Always fail on text2 (even after retries)
    fetchMock = mock((url: string, options?: RequestInit) => {
      const body = JSON.parse(options?.body as string);
      if (body.prompt === "text2") {
        // Always fail for text2, even on retries
        return Promise.reject(new Error("Failed on text2"));
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockEmbeddingResponse),
      } as Response);
    });
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.embedBatch(["text1", "text2", "text3"]);
    });

    // ACT & ASSERT
    const result = await Effect.runPromise(
      program.pipe(Effect.provide(layer), Effect.flip)
    );

    expect(result).toBeInstanceOf(OllamaError);
  });
});

// ============================================================================
// Tests: checkHealth()
// ============================================================================

describe("Ollama.checkHealth", () => {
  test("succeeds when Ollama is running with required model", async () => {
    // ARRANGE
    fetchMock = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockTagsResponse),
      } as Response)
    );
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.checkHealth();
    });

    // ACT & ASSERT
    await Effect.runPromise(program.pipe(Effect.provide(layer)));

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:11434/api/tags");
  });

  test("succeeds when model has version tag", async () => {
    // ARRANGE: Model name includes version suffix
    const tagsResponse = {
      models: [{ name: "mxbai-embed-large:v1" }],
    };

    fetchMock = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(tagsResponse),
      } as Response)
    );
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.checkHealth();
    });

    // ACT & ASSERT - Should not throw
    await Effect.runPromise(program.pipe(Effect.provide(layer)));
  });

  test("fails when Ollama is not reachable", async () => {
    // ARRANGE
    fetchMock = mock(() =>
      Promise.reject(new Error("ECONNREFUSED: Connection refused"))
    );
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.checkHealth();
    });

    // ACT & ASSERT
    const result = await Effect.runPromise(
      program.pipe(Effect.provide(layer), Effect.flip)
    );

    expect(result).toBeInstanceOf(OllamaError);
    expect((result as OllamaError).reason).toContain(
      "Cannot connect to Ollama"
    );
  });

  test("fails when Ollama returns HTTP error", async () => {
    // ARRANGE
    fetchMock = mock(() =>
      Promise.resolve({
        ok: false,
        status: 503,
      } as Response)
    );
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.checkHealth();
    });

    // ACT & ASSERT
    const result = await Effect.runPromise(
      program.pipe(Effect.provide(layer), Effect.flip)
    );

    expect(result).toBeInstanceOf(OllamaError);
    expect((result as OllamaError).reason).toContain("not responding");
  });

  test("fails when required model is not available", async () => {
    // ARRANGE: Different model available
    const tagsResponse = {
      models: [{ name: "llama2:latest" }, { name: "codellama:latest" }],
    };

    fetchMock = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(tagsResponse),
      } as Response)
    );
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.checkHealth();
    });

    // ACT & ASSERT
    const result = await Effect.runPromise(
      program.pipe(Effect.provide(layer), Effect.flip)
    );

    expect(result).toBeInstanceOf(OllamaError);
    expect((result as OllamaError).reason).toContain("Model");
    expect((result as OllamaError).reason).toContain("not found");
    expect((result as OllamaError).reason).toContain("ollama pull");
  });

  test("fails when JSON parsing fails", async () => {
    // ARRANGE
    fetchMock = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.reject(new Error("Malformed JSON")),
      } as Response)
    );
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(testConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.checkHealth();
    });

    // ACT & ASSERT
    const result = await Effect.runPromise(
      program.pipe(Effect.provide(layer), Effect.flip)
    );

    expect(result).toBeInstanceOf(OllamaError);
    expect((result as OllamaError).reason).toContain("Invalid response");
  });
});

// ============================================================================
// Tests: Configuration
// ============================================================================

describe("Ollama configuration", () => {
  test("uses custom ollama host from config", async () => {
    // ARRANGE
    const customConfig = new MemoryConfig({
      ...testConfig,
      ollamaHost: "http://custom-host:9999",
    });

    fetchMock = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockEmbeddingResponse),
      } as Response)
    );
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(customConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.embed("test");
    });

    // ACT
    await Effect.runPromise(program.pipe(Effect.provide(layer)));

    // ASSERT
    expect(fetchMock).toHaveBeenCalledWith(
      "http://custom-host:9999/api/embeddings",
      expect.any(Object)
    );
  });

  test("uses custom model from config", async () => {
    // ARRANGE
    const customConfig = new MemoryConfig({
      ...testConfig,
      ollamaModel: "custom-embed-model",
    });

    fetchMock = mock(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockEmbeddingResponse),
      } as Response)
    );
    globalThis.fetch = fetchMock as any;

    const layer = makeOllamaLive(customConfig);
    const program = Effect.gen(function* () {
      const ollama = yield* Ollama;
      return yield* ollama.embed("test");
    });

    // ACT
    await Effect.runPromise(program.pipe(Effect.provide(layer)));

    // ASSERT
    const calls = (fetchMock as any).mock.calls;
    const body = JSON.parse(calls[0][1]?.body as string);
    expect(body.model).toBe("custom-embed-model");
  });
});
