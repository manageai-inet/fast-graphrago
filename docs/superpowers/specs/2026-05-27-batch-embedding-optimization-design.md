# Batch Embedding Optimization Design

## Goal

Replace the per-entity embedding loop (N individual API calls) with chunked `EmbedBatch` calls, reducing round trips from N to N/BatchSize without changing graph structure or requiring external library modifications.

## Context

During indexing, entity embedding is the final slow step:

| Step | Current State |
|------|--------------|
| LLM extraction | ✅ Batched (5 chunks/call) |
| Relation dedup | ✅ Parallelized |
| **Entity embedding** | ❌ 1 API call/entity via `EmbedAsset` |
| ES insert | ✅ Bulk API |

With 500 entities and `MaxConcurrent=3`, the current approach makes ~167 sequential rounds of 3 parallel calls each. Batching reduces this to ceil(500/50) = 10 API calls.

Embedding happens **after** graph construction and does not affect entity/relation structure.

## Architecture

### Files Modified

- `rag/options.go` — add `EmbedBatchSize int` to compile-time options, add `WithEmbedder` and `WithEmbedBatchSize` option functions
- `rag/graph_rag.go` — add `Embedder am.Embedder` field, replace per-entity goroutine loop with chunked `EmbedBatch` goroutines

### New Fields

**`GraphRAGServiceCompileTimeOptions`** (`rag/options.go`):
```go
EmbedBatchSize int `json:"embed_batch_size"`
```
Default: `50`

**`GraphRAGServiceImpl`** (`rag/graph_rag.go`):
```go
Embedder am.Embedder
```

### New Options

```go
func WithEmbedder(embedder am.Embedder) Option
func WithEmbedBatchSize(size int) Option
```

### Embedding Flow (new)

```
entities [N]
  → split into batches of EmbedBatchSize (default 50)
  → goroutine per batch, controlled by MaxConcurrent semaphore
      → collect batch entity contents (name | type | description)
      → embedder.EmbedBatch(contents)  // 1 API call per batch
      → construct []VectorAsset from returned vectors
  → merge all VectorAssets
  → InsertBatchVectors (unchanged)
```

### Fallback

If `g.Embedder == nil`, the code falls back to the existing per-entity `EmbedAsset` goroutine loop. This keeps backward compatibility for callers that do not set `WithEmbedder`.

### Entity Content Construction

`EmbedAsset` is called with `nil` contentConstructorFn, so it uses `asset.Content` directly. The entity's `Content` field is already pre-formatted as `"name | type | description"` when the entity asset is built. The batch path simply reads the same field:

```go
content := entity.Content  // already "name | type | description"
```

## Usage (example/config.go)

```go
embedder := mai.NewEmbedderFromEnv()
vectorStore := v8.NewElasticsearchV8VectorRepo(..., embedder, "cosine")

opts = append(opts, rag.WithEmbedder(embedder))      // same instance, no extra config
opts = append(opts, rag.WithEmbedBatchSize(50))       // optional, 50 is the default
```

## Error Handling

- If `EmbedBatch` returns an error for a batch, that batch's goroutine sends the error to `errsCh` and returns — same pattern as the current per-entity error channel.
- Partial batch (last batch may be smaller than `EmbedBatchSize`) is handled naturally by slicing.

## Testing

- Unit test: mock embedder that records batch calls, verify N entities → ceil(N/BatchSize) calls
- Unit test: verify fallback (nil Embedder) still uses `EmbedAsset` path
- Unit test: verify VectorAsset content matches what `EmbedAsset` would produce
