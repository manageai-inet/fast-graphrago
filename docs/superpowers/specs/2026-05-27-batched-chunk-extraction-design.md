---
name: batched-chunk-extraction
description: Design spec for speeding up indexing by batching multiple chunks into a single LLM call during extraction
metadata:
  type: project
---

# Batched Chunk Extraction — Design Spec

**Date:** 2026-05-27  
**Branch:** feat/optimize  
**Goal:** Reduce LLM calls during the extraction phase of indexing by processing multiple chunks per call instead of one.

---

## Problem

The current extraction pipeline calls `ExtractGraphFromChunk` once per chunk, each call being a separate LLM API request. With `DefaultMaxConcurrent = 1`, all chunks are processed sequentially. Even raising `MaxConcurrent` does not meaningfully improve throughput because all requests queue at the same model endpoint — the model itself is the bottleneck.

**Root causes:**
1. 1 chunk → 1 LLM call: N chunks = N API roundtrips
2. `DefaultMaxConcurrent = 1`: no parallelism even within a phase
3. `GetGraphExtractionTool()` / `GetRelationClusterTool()` called per chunk, re-running JSON schema reflection every time

---

## Solution: Batch Extraction

Process K chunks per LLM call. Each batch is a single API request that returns all entities and relations across the K chunks, with each entity/relation carrying a `chunk_index` field to preserve the source attribution (`ChunkIds`) needed by the retrieval phase.

```
Before:  [c0]→LLM  [c1]→LLM  [c2]→LLM  [c3]→LLM  [c4]→LLM  ...  (N calls)
After:   [c0,c1,c2,c3,c4]→LLM  [c5,c6,c7,c8,c9]→LLM  ...         (N/K calls)
```

---

## Changes

### 1. `utils/const.go`

Add `DefaultBatchSize` and increase `DefaultMaxConcurrent`:

```go
DefaultMaxConcurrent = 3   // was 1; allows overlapping batch requests
DefaultBatchSize     = 5   // chunks per LLM call
```

`DefaultMaxConcurrent = 3` provides meaningful overlap without overwhelming a single model endpoint. Callers can tune both values via options.

---

### 2. `models/graph.go` — Schema

Add `ChunkIndex int` to `GraphEntity` and `GraphRelation` so the LLM can attribute each extracted item back to the correct source chunk:

```go
type GraphEntity struct {
    Name       string `json:"name"        jsonschema:"The name of the entity"`
    Type       string `json:"type"        jsonschema:"The type of the entity"`
    Desc       string `json:"desc"        jsonschema:"The description for the entity context"`
    ChunkIndex int    `json:"chunk_index" jsonschema:"Zero-based index of the source chunk in the input batch"`
}

type GraphRelation struct {
    Source     RelativeEntity `json:"source"      jsonschema:"The source of the relation"`
    Target     RelativeEntity `json:"target"      jsonschema:"The target of the relation"`
    Desc       string         `json:"desc"        jsonschema:"The description for the relation context"`
    ChunkIndex int            `json:"chunk_index" jsonschema:"Zero-based index of the source chunk in the input batch"`
}
```

Tool schemas are computed once in `NewFastGraphExtractor()` and stored as fields on the struct, eliminating repeated JSON schema reflection on every call. `GetGraphExtractionTool` and `GetRelationClusterTool` are called only at construction time.

---

### 3. `prompts/` — Extraction Prompt

`EntityRelationshipExtractionPrompt` is updated to accept multiple chunks. Format:

```
[chunk_0]
<content of chunk 0>

[chunk_1]
<content of chunk 1>
...
```

The system prompt gains one instruction: **set `chunk_index` to the zero-based index of the chunk the entity or relation was found in.**

Single-chunk behaviour is preserved: a batch of 1 is indistinguishable from the old format.

---

### 4. `graph/extractor.go` — Extractor

**New fields on `FastGraphExtractor`:**
- `BatchSize int` — chunks per LLM call (default `utils.DefaultBatchSize`)
- `extractionTool openai.ChatCompletionToolParam` — cached at `NewFastGraphExtractor()`
- `deduplicationTool openai.ChatCompletionToolParam` — cached at `NewFastGraphExtractor()`

**New method `ExtractGraphFromChunks(ctx, chunks []ContextualAsset, options) (GraphAssets, error)`:**
- Builds a single multi-chunk prompt
- Calls LLM once
- Maps `chunk_index` on each entity/relation back to the correct `chunk.AssetId`
- Returns `GraphAssets` with correct `ChunkIds` populated

**`ExtractGraph` updated:**
- Splits input chunks into batches of `BatchSize`
- Dispatches each batch as a goroutine (capped by `MaxConcurrent` semaphore)
- Merges results (existing dedup logic unchanged)

**`ExtractGraphFromChunk` (single-chunk) is kept** as a thin wrapper calling `ExtractGraphFromChunks` with a slice of one, so any external callers and tests continue to work.

---

### 5. `rag/options.go` — Options

Add `WithBatchSize(n int) Option` so callers can override the default:

```go
indexer, _ := fastgraphrag.New(llm, vecStore, assetStore,
    fastgraphrag.WithBatchSize(10),
)
```

---

## Error Handling

- If a batch LLM call fails, the entire batch is retried (up to `MaxRetryAttempt`) before propagating the error — same behaviour as the current per-chunk retry loop.
- If `chunk_index` in the LLM response is out of range for the batch, the call is treated as a failed attempt and retried.
- Existing error propagation via `errChan` is unchanged.

---

## What Does Not Change

- Relation deduplication phase (`DeduplicateRelations`) — same logic, unaffected by batching
- Downstream `ChunkIds` usage in retrieval — preserved through `chunk_index` mapping
- Public interface of `GraphRAGServiceImpl` (`Index`, `Retrieve`)
- `ExtractEntitiesFromQuery` — not part of indexing hot path

---

## Expected Impact

| Scenario | Before | After (BatchSize=5, MaxConcurrent=3) |
|---|---|---|
| 50 chunks | 50 sequential LLM calls | 10 calls, up to 3 in parallel |
| 100 chunks | 100 sequential LLM calls | 20 calls, up to 3 in parallel |

Effective speedup for the extraction phase: **~10-15x** for typical documents, assuming model latency scales sub-linearly with input size.
