# Batch Embedding Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-entity `EmbedAsset` loop (N API calls) with chunked `EmbedBatch` calls (N/BatchSize API calls) during indexing, without changing the graph structure or requiring external library modifications.

**Architecture:** Add `Embedder am.Embedder` field and `EmbedBatchSize int` option to `GraphRAGServiceImpl`. In `Index()`, when `Embedder` is set, use a new `batchEmbed()` helper that groups entities into batches and calls `embedder.EmbedBatch()` once per batch in parallel goroutines. If `Embedder` is nil, fall back to the existing `EmbedAsset` loop.

**Tech Stack:** Go 1.25, `github.com/manageai-inet/agentic-assets` (`am.Embedder`, `am.VectorAsset`, `am.ContextualAsset`)

---

### Task 1: Add constant, options fields, and struct field

**Files:**
- Modify: `utils/const.go`
- Modify: `rag/options.go`
- Modify: `rag/graph_rag.go`

No behavioral change in this task — purely wiring the new configuration through the existing option system.

- [ ] **Step 1: Add `DefaultEmbedBatchSize` to `utils/const.go`**

Add after `DefaultBatchSize`:

```go
const DefaultEmbedBatchSize = 50
```

- [ ] **Step 2: Add `EmbedBatchSize` field to `GraphRAGServiceCompileTimeOptions` in `rag/options.go`**

In the `GraphRAGServiceCompileTimeOptions` struct, add after `BatchSize`:

```go
// (Engine Config) Batch size for embedding (entities per EmbedBatch API call)
EmbedBatchSize int `json:"embed_batch_size"`
```

- [ ] **Step 3: Add `Embedder` field to `GraphRAGServiceCompileTimeOptions` in `rag/options.go`**

Add after `EmbedBatchSize` (requires import `asset_manager "github.com/manageai-inet/agentic-assets"`):

```go
// (Engine Config) Embedder for batch embedding; if nil, falls back to per-entity EmbedAsset
Embedder asset_manager.Embedder
```

Add the import to `rag/options.go` if not already present:
```go
asset_manager "github.com/manageai-inet/agentic-assets"
```

- [ ] **Step 4: Set default in `NewGraphRAGServiceOptions()`**

In `NewGraphRAGServiceOptions()`, add `EmbedBatchSize: utils.DefaultEmbedBatchSize` to the `GraphRAGServiceCompileTimeOptions` literal:

```go
GraphRAGServiceCompileTimeOptions: GraphRAGServiceCompileTimeOptions{
    MaxConcurrent:  utils.DefaultMaxConcurrent,
    BatchSize:      utils.DefaultBatchSize,
    EmbedBatchSize: utils.DefaultEmbedBatchSize,
},
```

- [ ] **Step 5: Add `WithEmbedder` option function to `rag/options.go`**

```go
func WithEmbedder(embedder asset_manager.Embedder) Option {
    return func(opts *GraphRAGServiceOptions) {
        opts.Embedder = embedder
    }
}
```

- [ ] **Step 6: Add `WithEmbedBatchSize` option function to `rag/options.go`**

```go
func WithEmbedBatchSize(size int) Option {
    return func(opts *GraphRAGServiceOptions) {
        opts.EmbedBatchSize = size
    }
}
```

- [ ] **Step 7: Add `Embedder` field to `GraphRAGServiceImpl` in `rag/graph_rag.go`**

`GraphRAGServiceImpl` embeds `GraphRAGServiceCompileTimeOptions`, which now contains `Embedder` and `EmbedBatchSize`. No separate field declaration needed — they are already promoted. Verify the struct definition still compiles:

```go
type GraphRAGServiceImpl struct {
    serviceName string
    VectorStore asset_manager.VectorStorage
    AssetStore  asset_manager.AssetStorage
    GraphRAGServiceCompileTimeOptions  // includes Embedder, EmbedBatchSize
    GraphRAGServiceRuntimeOptions
    asset_manager.LoggingCapacity
}
```

- [ ] **Step 8: Verify it compiles**

```bash
cd /Users/bestty/Documents/fast-graphrago && go build ./...
```

Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add utils/const.go rag/options.go rag/graph_rag.go
git commit -m "feat: add EmbedBatchSize and Embedder options to GraphRAGService"
```

---

### Task 2: Extract `batchEmbed` helper and write tests

**Files:**
- Create: `rag/graph_rag_test.go`
- Modify: `rag/graph_rag.go`

The helper is a package-level function in `rag/`. Tests live in the same package (`package rag`).

- [ ] **Step 1: Write the failing test file `rag/graph_rag_test.go`**

```go
package rag

import (
    "context"
    "fmt"
    "sync"
    "testing"

    asset_manager "github.com/manageai-inet/agentic-assets"
)

// mockEmbedder records EmbedBatch calls and returns fixed-dimension zero vectors.
type mockEmbedder struct {
    mu    sync.Mutex
    calls [][]string
    model string
    dim   int
}

func (m *mockEmbedder) GetEmbeddingModel() string { return m.model }
func (m *mockEmbedder) GetEmbeddingDim() int      { return m.dim }
func (m *mockEmbedder) Embed(ctx context.Context, content string) ([]float32, error) {
    vecs, err := m.EmbedBatch(ctx, []string{content})
    if err != nil {
        return nil, err
    }
    return vecs[0], nil
}
func (m *mockEmbedder) EmbedBatch(ctx context.Context, contents []string) ([][]float32, error) {
    m.mu.Lock()
    m.calls = append(m.calls, append([]string{}, contents...))
    m.mu.Unlock()
    result := make([][]float32, len(contents))
    for i := range result {
        result[i] = make([]float32, m.dim)
    }
    return result, nil
}

func makeTestEntities(n int) []asset_manager.ContextualAsset {
    entities := make([]asset_manager.ContextualAsset, n)
    for i := range entities {
        entities[i] = asset_manager.ContextualAsset{
            KbId:      "kb-test",
            AssetId:   fmt.Sprintf("entity-%d", i),
            AssetType: asset_manager.AssetTypeEntity,
            Version:   1,
            Content:   fmt.Sprintf("entity%d | type | desc%d", i, i),
        }
    }
    return entities
}

// TestBatchEmbed_BatchCount verifies N entities → ceil(N/batchSize) EmbedBatch calls.
func TestBatchEmbed_BatchCount(t *testing.T) {
    entities := makeTestEntities(5)
    emb := &mockEmbedder{model: "test-model", dim: 3}

    vecs, err := batchEmbed(context.Background(), entities, emb, 2, 4)
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
    if len(vecs) != 5 {
        t.Errorf("expected 5 VectorAssets, got %d", len(vecs))
    }
    if len(emb.calls) != 3 { // ceil(5/2) = 3
        t.Errorf("expected 3 EmbedBatch calls, got %d", len(emb.calls))
    }
}

// TestBatchEmbed_VectorAssetFields verifies VectorAsset fields match the source entity.
func TestBatchEmbed_VectorAssetFields(t *testing.T) {
    entities := makeTestEntities(1)
    entities[0].Content = "ENTITY | type | description"
    emb := &mockEmbedder{model: "my-model", dim: 3}

    vecs, err := batchEmbed(context.Background(), entities, emb, 50, 1)
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
    v := vecs[0]
    if v.KbId != "kb-test" {
        t.Errorf("KbId: got %q, want %q", v.KbId, "kb-test")
    }
    if v.AssetId != "entity-0" {
        t.Errorf("AssetId: got %q, want %q", v.AssetId, "entity-0")
    }
    if v.Content != "ENTITY | type | description" {
        t.Errorf("Content: got %q, want %q", v.Content, "ENTITY | type | description")
    }
    if v.EmbeddingModel == nil || *v.EmbeddingModel != "my-model" {
        t.Errorf("EmbeddingModel: got %v, want %q", v.EmbeddingModel, "my-model")
    }
    if len(v.EmbededVector) != 3 {
        t.Errorf("EmbededVector length: got %d, want 3", len(v.EmbededVector))
    }
    if v.Refs == nil || len(*v.Refs) != 1 {
        t.Errorf("expected 1 parent ref, got %v", v.Refs)
    } else {
        ref := (*v.Refs)[0]
        if ref.RefType != asset_manager.AssetRefTypeParent {
            t.Errorf("ref.RefType: got %q, want %q", ref.RefType, asset_manager.AssetRefTypeParent)
        }
        if ref.AssetId != "entity-0" {
            t.Errorf("ref.AssetId: got %q, want %q", ref.AssetId, "entity-0")
        }
    }
}

// TestBatchEmbed_PartialLastBatch verifies the last batch is handled correctly when N % batchSize != 0.
func TestBatchEmbed_PartialLastBatch(t *testing.T) {
    entities := makeTestEntities(7)
    emb := &mockEmbedder{model: "test-model", dim: 2}

    vecs, err := batchEmbed(context.Background(), entities, emb, 3, 4)
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
    if len(vecs) != 7 {
        t.Errorf("expected 7 VectorAssets, got %d", len(vecs))
    }
    if len(emb.calls) != 3 { // ceil(7/3) = 3: batches of [3, 3, 1]
        t.Errorf("expected 3 EmbedBatch calls, got %d", len(emb.calls))
    }
    if len(emb.calls[2]) != 1 { // last batch has 1 entity
        t.Errorf("last batch: expected 1 content, got %d", len(emb.calls[2]))
    }
}
```


- [ ] **Step 2: Run the test to verify it fails (function not defined)**

```bash
cd /Users/bestty/Documents/fast-graphrago && go test ./rag/... -run TestBatchEmbed -v
```

Expected: compile error `undefined: batchEmbed`

- [ ] **Step 3: Implement `batchEmbed` in `rag/graph_rag.go`**

Add this function at the package level (before or after `NewGraphRAGService`):

```go
// batchEmbed embeds entities in batches using EmbedBatch, returning one VectorAsset per entity.
// Batches run concurrently up to maxConcurrent. Results are ordered to match entities.
func batchEmbed(ctx context.Context, entities []asset_manager.ContextualAsset, embedder asset_manager.Embedder, batchSize, maxConcurrent int) ([]asset_manager.VectorAsset, error) {
    if batchSize <= 0 {
        batchSize = 50
    }
    n := len(entities)
    numBatches := (n + batchSize - 1) / batchSize
    embModel := embedder.GetEmbeddingModel()

    vectorAssets := make([]asset_manager.VectorAsset, n)
    errCh := make(chan error, 1)
    sem := make(chan struct{}, maxConcurrent)
    var wg sync.WaitGroup

    for b := 0; b < numBatches; b++ {
        start := b * batchSize
        end := min(start+batchSize, n)
        wg.Add(1)
        sem <- struct{}{}
        go func(batch []asset_manager.ContextualAsset, startIdx int) {
            defer wg.Done()
            defer func() { <-sem }()

            contents := make([]string, len(batch))
            for i, e := range batch {
                contents[i] = e.Content
            }
            vectors, err := embedder.EmbedBatch(ctx, contents)
            if err != nil {
                select {
                case errCh <- err:
                default:
                }
                return
            }
            model := embModel
            for i, e := range batch {
                parentRef := asset_manager.AssetRef{
                    KbId:      e.KbId,
                    AssetType: e.AssetType,
                    AssetId:   e.AssetId,
                    RefType:   asset_manager.AssetRefTypeParent,
                }
                refs := []asset_manager.AssetRef{parentRef}
                vectorAssets[startIdx+i] = asset_manager.VectorAsset{
                    KbId:           e.KbId,
                    AssetId:        e.AssetId,
                    Version:        e.Version,
                    Content:        e.Content,
                    Refs:           &refs,
                    Labels:         e.Labels,
                    Metadata:       e.Metadata,
                    EmbeddingModel: &model,
                    EmbededVector:  vectors[i],
                }
            }
        }(entities[start:end], start)
    }

    wg.Wait()
    select {
    case err := <-errCh:
        return nil, err
    default:
    }
    return vectorAssets, nil
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /Users/bestty/Documents/fast-graphrago && go test ./rag/... -run TestBatchEmbed -v -race
```

Expected: all 3 tests PASS, no race conditions.

- [ ] **Step 5: Commit**

```bash
git add rag/graph_rag.go rag/graph_rag_test.go
git commit -m "feat: add batchEmbed helper with tests"
```

---

### Task 3: Use `batchEmbed` in `Index()` with fallback

**Files:**
- Modify: `rag/graph_rag.go` (embedding section only, lines ~330–368)

- [ ] **Step 1: Replace the embedding section in `Index()`**

Find the block in `graph_rag.go` starting with the comment `// 4. Generate Embeddings for Nodes (Entities)` (around line 330) and replace the entire block through `logger.DebugContext(ctx, "generated embeddings for entities successfully"...)` with:

```go
// 4. Generate Embeddings for Nodes (Entities)
logger.DebugContext(ctx, "generating embeddings for entities", slog.String("kbId", kbId), slog.Int("entities", len(entityAssets)))
var vectorAssets []asset_manager.VectorAsset
if g.Embedder != nil {
    var embedErr error
    vectorAssets, embedErr = batchEmbed(ctx, entityAssets, g.Embedder, g.EmbedBatchSize, g.MaxConcurrent)
    if embedErr != nil {
        logger.ErrorContext(ctx, embedErr.Error(), slog.String("kbId", kbId))
        return []asset_manager.ContextualAsset{}, []asset_manager.VectorAsset{}, embedErr
    }
} else {
    vectorAssets = make([]asset_manager.VectorAsset, 0, len(entityAssets))
    var wg sync.WaitGroup
    var mu sync.Mutex
    sem := make(chan struct{}, g.MaxConcurrent)
    errsCh := make(chan error, len(entityAssets))
    for _, entity := range entityAssets {
        wg.Add(1)
        sem <- struct{}{}
        go func(vecArray *[]asset_manager.VectorAsset, entity *asset_manager.ContextualAsset) {
            defer wg.Done()
            defer func() { <-sem }()
            v, err := g.VectorStore.EmbedAsset(ctx, entity, nil)
            if err != nil {
                errsCh <- err
                return
            }
            entity.EmbeddingModel = v.EmbeddingModel
            mu.Lock()
            *vecArray = append(*vecArray, v)
            mu.Unlock()
        }(&vectorAssets, &entity)
    }
    done := make(chan struct{})
    go func() {
        wg.Wait()
        close(done)
    }()
    select {
    case err := <-errsCh:
        logger.ErrorContext(ctx, err.Error(), slog.String("kbId", kbId))
        return []asset_manager.ContextualAsset{}, []asset_manager.VectorAsset{}, err
    case <-done:
    }
}
logger.DebugContext(ctx, "generated embeddings for entities successfully", slog.String("kbId", kbId), slog.Int("entities", len(vectorAssets)))
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /Users/bestty/Documents/fast-graphrago && go build ./...
```

Expected: no errors.

- [ ] **Step 3: Run all tests**

```bash
cd /Users/bestty/Documents/fast-graphrago && go test ./... -race
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add rag/graph_rag.go
git commit -m "feat: use batchEmbed in Index() when Embedder is set, fallback to EmbedAsset"
```

---

### Task 4: Wire up options in `example/config.go`

**Files:**
- Modify: `example/config.go`

- [ ] **Step 1: Add `EmbedBatchSize` env var to `AppConfig`**

In the `AppConfig` struct in `example/config.go`, add after `MaxConcurrent`:

```go
// Batch size for embedding (entities per EmbedBatch API call)
EmbedBatchSize int `envconfig:"embed_batch_size" default:"50"`
```

- [ ] **Step 2: Pass `WithEmbedder` and `WithEmbedBatchSize` to options in `InitializeFastGraphIndexerFromConfig`**

The `embedder` variable is already created at line ~143. Add two lines after `opts = append(opts, rag.WithGraphExtractor(graphExt))`:

```go
opts = append(opts, rag.WithEmbedder(embedder))
opts = append(opts, rag.WithEmbedBatchSize(cfg.EmbedBatchSize))
```

- [ ] **Step 3: Build the example to verify**

```bash
cd /Users/bestty/Documents/fast-graphrago/example && go build ./...
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
cd /Users/bestty/Documents/fast-graphrago
git add example/config.go
git commit -m "feat: wire WithEmbedder and WithEmbedBatchSize in example config"
```
