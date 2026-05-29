# Batched Chunk Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up the indexing extraction phase by batching K chunks into a single LLM call instead of one call per chunk, reducing total LLM calls from N to N/K.

**Architecture:** `FastGraphExtractor.ExtractGraph` splits chunks into batches of `BatchSize` and dispatches each batch to `ExtractGraphFromChunks`, which sends all chunks in one LLM call. The LLM returns `chunk_index` on every entity/relation so `ChunkIds` attribution is preserved. Tool schemas are pre-computed at constructor time instead of on every call.

**Tech Stack:** Go 1.25, `github.com/openai/openai-go`, `github.com/manageai-inet/agentic-assets`

---

## File Map

| File | Change |
|---|---|
| `utils/const.go` | `DefaultMaxConcurrent` 1→3, add `DefaultBatchSize = 5` |
| `models/graph.go` | Add `ChunkIndex int` to `GraphEntity` and `GraphRelation` |
| `prompts/prompts.go` | Replace single-chunk user prompt with batch format; add chunk_index instruction to system prompt |
| `graph/interface.go` | Add `SetBatchSize(int)` to `GraphExtractor` interface |
| `graph/extractor.go` | Add `BatchSize`, `extractionTool`, `deduplicationTool`, `queryTool` fields; implement `ExtractGraphFromChunks`; update `ExtractGraph` to batch; update `ExtractGraphFromChunk` as thin wrapper; use cached tools in `DeduplicateRelations` and `ExtractEntitiesFromQuery` |
| `graph/extractor_test.go` | New — unit tests for `ExtractGraphFromChunks` and batched `ExtractGraph` |
| `rag/options.go` | Add `BatchSize int` to `GraphRAGServiceCompileTimeOptions`; add `WithBatchSize` option |

---

## Task 1: Update constants

**Files:**
- Modify: `utils/const.go`

- [ ] **Step 1: Update `utils/const.go`**

Replace:
```go
const DefaultMaxConcurrent = 1
```
With:
```go
const DefaultMaxConcurrent = 3
const DefaultBatchSize     = 5
```

The full file after change:
```go
package utils

import "strings"

const DefaultDomain = "enterprise"

var DefaultEntityTypes = []string{
	"person",
	"organization",
	"location",
	"event",
	"date",
	"product",
	"technology",
	"concept",
	"document",
	"project",
	"role",
	"process",
	"regulation",
	"communication",
	"goal",
	"metric",
	"challenge",
	"solution",
	"resource",
	"outcome",
	"table",
	"image",
	"form",
}

const DefaultChunkSize = 500
const DefaultChunkOverlap = 50

const DefaultMaxRetryAttempt = 1
const DefaultMaxConcurrent   = 3
const DefaultBatchSize        = 5
const DefaultMaxWalks         = 100
const DefaultDampingFactor    = 0.5
const DefaultTopK             = 50
const DefaultThreshold        = float32(0.3)

var DefaultQueryWeights = [3]float32{0.35, 0.15, 0.5}

func NormalizeName(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}
```

- [ ] **Step 2: Verify it compiles**

```bash
go build ./utils/...
```
Expected: no output (clean build).

- [ ] **Step 3: Commit**

```bash
git add utils/const.go
git commit -m "feat: add DefaultBatchSize=5, raise DefaultMaxConcurrent to 3"
```

---

## Task 2: Add `ChunkIndex` to extraction models

**Files:**
- Modify: `models/graph.go`

- [ ] **Step 1: Add `ChunkIndex int` to `GraphEntity` and `GraphRelation`**

In `models/graph.go`, replace the two struct definitions:

```go
// before
type GraphEntity struct {
	Name string `json:"name" jsonschema:"The name of the entity"`
	Type string `json:"type" jsonschema:"The type of the entity"`
	Desc string `json:"desc" jsonschema:"The description for the entity context"`
}

type GraphRelation struct {
	Source RelativeEntity `json:"source" jsonschema:"The source of the relation"`
	Target RelativeEntity `json:"target" jsonschema:"The target of the relation"`
	Desc   string `json:"desc" jsonschema:"The description for the relation context"`
}
```

With:

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

- [ ] **Step 2: Verify it compiles**

```bash
go build ./models/...
```
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add models/graph.go
git commit -m "feat: add chunk_index field to GraphEntity and GraphRelation for batch attribution"
```

---

## Task 3: Update extraction prompt for batch input

**Files:**
- Modify: `prompts/prompts.go`

The system prompt needs one added instruction (step 5) about `chunk_index`. The user prompt changes from a single `%s` document to a multi-chunk format.

- [ ] **Step 1: Update `EntityRelationshipExtractionSystem` — add step 5**

In `prompts/prompts.go`, find the `INSTRUCTIONS` block inside `EntityRelationshipExtractionSystem` and add step 5 after step 4:

```
5. **CHUNK ATTRIBUTION**: For each extracted entity and relation, set the ` + "`chunk_index`" + ` field to the zero-based index of the chunk it was extracted from (matching the ` + "`[chunk_N]`" + ` labels in the input).
```

The full updated `EntityRelationshipExtractionSystem` constant:

```go
const EntityRelationshipExtractionSystem = `# DOMAIN
%s

# GOAL
Your goal is to highlight information that is relevant to the domain and the queries that may be asked on it. Given an input document, identify all relevant entities and all relationships among them.

# INSTRUCTIONS
1. **ENTITY IDENTIFICATION**: Identify and meticulously extract all entities mentioned in the document that belong to the provided ENTITY TYPES. For each entity, provide a concise description capturing its key features within the document's context. Use singular entity names and split compound concepts when necessary for clarity.
2. **RELATIONSHIP DISCOVERY**: Identify and describe ALL relationships between the extracted entities. Resolve pronouns to entity names for clarity. Ensure relationship descriptions clearly explain the connection between entities.
3. **ENTITY COVERAGE CHECK**: Verify that every identified entity is part of at least one relationship. If any entity is isolated, infer and add a relationship to connect it to the graph, even if the relationship is implicit.
4. **OUTPUT FORMAT: STRICTLY VALID JSON**: Output MUST be in strictly valid JSON format.  Adhere to ALL standard JSON rules. The JSON MUST contain two top-level lists: "entities", "relationships". Each list item must be a JSON object with the REQUIRED fields ("name", "type", "desc" for entities; "source", "target", "desc" for relationships), all as JSON strings enclosed in DOUBLE QUOTES ONLY.  **ABSOLUTELY NO SINGLE QUOTES, BRACKETS FOR STRINGS, TRAILING COMMAS, OR MARKDOWN FORMATTING (like triple backticks) ARE PERMITTED.**  Invalid JSON output is unacceptable.
5. **CHUNK ATTRIBUTION**: For each extracted entity and relation, set the ` + "`chunk_index`" + ` field to the zero-based index of the chunk it was extracted from (matching the ` + "`[chunk_N]`" + ` labels in the input). For a single-chunk input the value is always 0.

# EXAMPLE INPUT DATA
Domain: 
Allowed Entity Types: [location, organization, person, communication]
Document: "Radio City: Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs."

# EXAMPLE OUTPUT DATA (VALID JSON - DO NOT DEVIATE)
{
  "entities": [
    {"name": "RADIO CITY", "type": "organization", "desc": "India's first private FM radio station", "chunk_index": 0},
    {"name": "INDIA", "type": "location", "desc": "A country in South Asia", "chunk_index": 0},
    {"name": "FM RADIO STATION", "type": "communication", "desc": "A radio broadcasting service using frequency modulation", "chunk_index": 0},
    {"name": "ENGLISH", "type": "communication", "desc": "A language of global communication", "chunk_index": 0},
    {"name": "HINDI", "type": "communication", "desc": "An Indo-Aryan language of India", "chunk_index": 0}
  ],
  "relationships": [
    {"source": {"name": "RADIO CITY", "type": "organization"}, "target": {"name": "INDIA", "type": "location"}, "desc": "Radio City is geographically situated in India", "chunk_index": 0},
    {"source": {"name": "RADIO CITY", "type": "organization"}, "target": {"name": "FM RADIO STATION", "type": "communication"}, "desc": "Radio City operates as a private FM radio station, launched on July 3, 2001", "chunk_index": 0},
    {"source": {"name": "RADIO CITY", "type": "organization"}, "target": {"name": "ENGLISH", "type": "communication"}, "desc": "Radio City's broadcasts include songs in the English language", "chunk_index": 0},
    {"source": {"name": "RADIO CITY", "type": "organization"}, "target": {"name": "HINDI", "type": "communication"}, "desc": "Radio City's broadcasts also feature songs in the Hindi language", "chunk_index": 0}
  ]
}`
```

- [ ] **Step 2: Replace `EntityRelationshipExtractionPrompt` with batch version**

Replace:
```go
const EntityRelationshipExtractionPrompt = `
# INPUT DATA
<<ENTITY_TYPES_START>>
**Entity Types**:
%s
<<ENTITY_TYPES_END>>

<<DOCUMENT_START>>
**Document**:
%s
<<DOCUMENT_END>>
`
```

With:
```go
const EntityRelationshipExtractionPrompt = `
# INPUT DATA
<<ENTITY_TYPES_START>>
**Entity Types**:
%s
<<ENTITY_TYPES_END>>

<<DOCUMENTS_START>>
**Documents** (%d chunks):

%s
<<DOCUMENTS_END>>
`
```

The format arguments are now `(entityTypes string, numChunks int, chunksContent string)` where `chunksContent` is pre-built as:
```
[chunk_0]
<content>

[chunk_1]
<content>
```

- [ ] **Step 3: Verify it compiles**

```bash
go build ./prompts/...
```
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add prompts/prompts.go
git commit -m "feat: update extraction prompt to support batch chunks with chunk_index attribution"
```

---

## Task 4: Add `SetBatchSize` to interface and pre-compute tools in constructor

**Files:**
- Modify: `graph/interface.go`
- Modify: `graph/extractor.go`

- [ ] **Step 1: Add `SetBatchSize(int)` to `GraphExtractor` interface**

In `graph/interface.go`, add `SetBatchSize(int)` next to `SetLLM`:

```go
package graph

import (
	"context"
	"github.com/manageai-inet/fast-graphrago/llm"
	"github.com/manageai-inet/fast-graphrago/models"
	"github.com/manageai-inet/fast-graphrago/utils"

	asset_manager "github.com/manageai-inet/agentic-assets"
)

type GraphExtractionOptions struct {
	Domain      string
	EntityTypes []string
}

func NewGraphExtractionOptions() GraphExtractionOptions {
	return GraphExtractionOptions{
		Domain:      utils.DefaultDomain,
		EntityTypes: utils.DefaultEntityTypes,
	}
}

type GraphExtractor interface {
	SetLLM(llm llm.LLM)
	SetBatchSize(batchSize int)
	ExtractGraph(ctx context.Context, chunks []asset_manager.ContextualAsset, options GraphExtractionOptions) (models.GraphAssets, error)
	ExtractGraphFromChunk(ctx context.Context, chunk asset_manager.ContextualAsset, options GraphExtractionOptions) (models.GraphAssets, error)
	ExtractEntitiesFromQuery(ctx context.Context, query string) (models.ExtractedQuery, error)
	asset_manager.Loggable
}
```

- [ ] **Step 2: Add fields and update `NewFastGraphExtractor` in `graph/extractor.go`**

Replace the `FastGraphExtractor` struct and constructor:

```go
type FastGraphExtractor struct {
	LLM              llm.LLM
	MaxRetryAttempt  int
	MaxConcurrent    int
	BatchSize        int
	extractionTool   openai.ChatCompletionToolParam
	deduplicationTool openai.ChatCompletionToolParam
	queryTool        openai.ChatCompletionToolParam
	asset_manager.LoggingCapacity
}

func NewFastGraphExtractor(llm llm.LLM) *FastGraphExtractor {
	extractionTool, err := models.GetGraphExtractionTool()
	if err != nil {
		panic("failed to build extraction tool schema: " + err.Error())
	}
	deduplicationTool, err := models.GetRelationClusterTool()
	if err != nil {
		panic("failed to build deduplication tool schema: " + err.Error())
	}
	queryTool, err := models.GetQueryExtractionTool()
	if err != nil {
		panic("failed to build query tool schema: " + err.Error())
	}
	return &FastGraphExtractor{
		LLM:               llm,
		MaxRetryAttempt:   utils.DefaultMaxRetryAttempt,
		MaxConcurrent:     utils.DefaultMaxConcurrent,
		BatchSize:         utils.DefaultBatchSize,
		extractionTool:    extractionTool,
		deduplicationTool: deduplicationTool,
		queryTool:         queryTool,
		LoggingCapacity:   *asset_manager.GetDefaultLoggingCapacity(),
	}
}
```

- [ ] **Step 3: Add `SetBatchSize` method**

After the existing `SetLLM` method, add:

```go
func (f *FastGraphExtractor) SetBatchSize(batchSize int) {
	f.BatchSize = batchSize
}
```

- [ ] **Step 4: Verify it compiles**

```bash
go build ./graph/...
```
Expected: no output. (The next tasks will fix remaining compilation errors from the prompt signature change.)

- [ ] **Step 5: Commit**

```bash
git add graph/interface.go graph/extractor.go
git commit -m "feat: add BatchSize field, pre-compute tool schemas in FastGraphExtractor constructor"
```

---

## Task 5: Implement `ExtractGraphFromChunks` (TDD)

**Files:**
- Create: `graph/extractor_test.go`
- Modify: `graph/extractor.go`

- [ ] **Step 1: Create `graph/extractor_test.go` with mock LLM and failing test**

```go
package graph

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/manageai-inet/fast-graphrago/models"
	"github.com/manageai-inet/fast-graphrago/utils"

	"github.com/openai/openai-go"

	asset_manager "github.com/manageai-inet/agentic-assets"
)

// mockLLM replays pre-set responses in order.
type mockLLM struct {
	responses []openai.ChatCompletionMessage
	callCount int
}

func (m *mockLLM) Generate(
	_ context.Context,
	_ []openai.ChatCompletionMessage,
	_ []openai.ChatCompletionToolParam,
	_ *openai.ChatCompletionToolChoiceOptionUnionParam,
) (*openai.ChatCompletionMessage, error) {
	resp := m.responses[m.callCount]
	m.callCount++
	return &resp, nil
}

func toolCallMsg(v any) openai.ChatCompletionMessage {
	b, _ := json.Marshal(v)
	return openai.ChatCompletionMessage{
		ToolCalls: []openai.ChatCompletionMessageToolCall{
			{Function: openai.ChatCompletionMessageToolCallFunction{Arguments: string(b)}},
		},
	}
}

func newTestExtractor(responses []openai.ChatCompletionMessage) *FastGraphExtractor {
	e := NewFastGraphExtractor(&mockLLM{responses: responses})
	e.BatchSize = utils.DefaultBatchSize
	return e
}

// TestExtractGraphFromChunks_MapChunkIndex verifies that chunk_index in the
// LLM response is mapped back to the correct ChunkIds on each entity/relation.
func TestExtractGraphFromChunks_MapChunkIndex(t *testing.T) {
	chunks := []asset_manager.ContextualAsset{
		{AssetId: "kb:file.txt:chunk-0", Content: "Alice works at Acme."},
		{AssetId: "kb:file.txt:chunk-1", Content: "Bob founded Globex."},
	}

	llmResp := toolCallMsg(models.ExtractedGraph{
		Entities: []models.GraphEntity{
			{Name: "Alice", Type: "person", Desc: "an employee", ChunkIndex: 0},
			{Name: "Acme",  Type: "organization", Desc: "a company",  ChunkIndex: 0},
			{Name: "Bob",   Type: "person", Desc: "a founder",  ChunkIndex: 1},
			{Name: "Globex", Type: "organization", Desc: "another company", ChunkIndex: 1},
		},
		Relations: []models.GraphRelation{
			{
				Source:     models.RelativeEntity{Name: "Alice", Type: "person"},
				Target:     models.RelativeEntity{Name: "Acme", Type: "organization"},
				Desc:       "works at",
				ChunkIndex: 0,
			},
			{
				Source:     models.RelativeEntity{Name: "Bob", Type: "person"},
				Target:     models.RelativeEntity{Name: "Globex", Type: "organization"},
				Desc:       "founded",
				ChunkIndex: 1,
			},
		},
	})

	e := newTestExtractor([]openai.ChatCompletionMessage{llmResp})
	opts := GraphExtractionOptions{Domain: "test", EntityTypes: []string{"person", "organization"}}

	result, err := e.ExtractGraphFromChunks(context.Background(), chunks, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// LLM must have been called exactly once for the whole batch
	if e.LLM.(*mockLLM).callCount != 1 {
		t.Errorf("expected 1 LLM call, got %d", e.LLM.(*mockLLM).callCount)
	}

	// Check entity ChunkIds
	entityChunkIds := map[string]string{}
	for _, ea := range result.EntityAssets {
		if len(ea.ChunkIds) != 1 {
			t.Errorf("entity %s: expected 1 ChunkId, got %d", ea.Name, len(ea.ChunkIds))
		} else {
			entityChunkIds[ea.Name] = ea.ChunkIds[0]
		}
	}

	if entityChunkIds["alice"] != "kb:file.txt:chunk-0" {
		t.Errorf("alice ChunkId = %q, want kb:file.txt:chunk-0", entityChunkIds["alice"])
	}
	if entityChunkIds["bob"] != "kb:file.txt:chunk-1" {
		t.Errorf("bob ChunkId = %q, want kb:file.txt:chunk-1", entityChunkIds["bob"])
	}

	// Check relation ChunkIds
	for _, ra := range result.RelationAssets {
		if len(ra.ChunkIds) != 1 {
			t.Errorf("relation %s→%s: expected 1 ChunkId, got %d", ra.Source, ra.Target, len(ra.ChunkIds))
		}
	}
}

// TestExtractGraphFromChunks_InvalidChunkIndex verifies an out-of-range
// chunk_index causes an error so the caller can retry.
func TestExtractGraphFromChunks_InvalidChunkIndex(t *testing.T) {
	chunks := []asset_manager.ContextualAsset{
		{AssetId: "kb:file.txt:chunk-0", Content: "Alice works at Acme."},
	}
	llmResp := toolCallMsg(models.ExtractedGraph{
		Entities: []models.GraphEntity{
			{Name: "Alice", Type: "person", Desc: "an employee", ChunkIndex: 5}, // out of range
		},
		Relations: []models.GraphRelation{},
	})

	e := newTestExtractor([]openai.ChatCompletionMessage{llmResp})
	opts := GraphExtractionOptions{Domain: "test", EntityTypes: []string{"person"}}

	_, err := e.ExtractGraphFromChunks(context.Background(), chunks, opts)
	if err == nil {
		t.Fatal("expected error for out-of-range chunk_index, got nil")
	}
}
```

- [ ] **Step 2: Run the test — confirm it fails to compile (method not yet implemented)**

```bash
go test ./graph/... -run TestExtractGraphFromChunks -v 2>&1 | head -20
```
Expected: compilation error — `ExtractGraphFromChunks undefined`.

- [ ] **Step 3: Implement `ExtractGraphFromChunks` in `graph/extractor.go`**

Add after `ExtractGraph` (before `DeduplicateRelations`):

```go
func (f *FastGraphExtractor) ExtractGraphFromChunks(ctx context.Context, chunks []asset_manager.ContextualAsset, options GraphExtractionOptions) (models.GraphAssets, error) {
	var sb strings.Builder
	for i, ch := range chunks {
		fmt.Fprintf(&sb, "[chunk_%d]\n%s\n\n", i, ch.Content)
	}
	chunksContent := strings.TrimRight(sb.String(), "\n")

	systemPrompt := fmt.Sprintf(prompts.EntityRelationshipExtractionSystem, options.Domain)
	entityTypes := strings.Join(options.EntityTypes, ", ")
	extractionPrompt := fmt.Sprintf(prompts.EntityRelationshipExtractionPrompt, entityTypes, len(chunks), chunksContent)

	messages := []openai.ChatCompletionMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: extractionPrompt},
	}
	toolChoice := openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("required")}

	generation, err := f.LLM.Generate(ctx, messages, []openai.ChatCompletionToolParam{f.extractionTool}, &toolChoice)
	if err != nil {
		return models.GraphAssets{}, err
	}
	if len(generation.ToolCalls) == 0 {
		return models.GraphAssets{}, fmt.Errorf("no tool calls found")
	}

	graph, err := utils.HandleToolCall[models.ExtractedGraph](generation.ToolCalls[0].Function.Arguments)
	if err != nil {
		return models.GraphAssets{}, err
	}

	entityKeyToId := make(map[string]string)
	entityAssets := make([]models.EntityAsset, 0, len(graph.Entities))
	relationAssets := make([]models.RelationAsset, 0, len(graph.Relations))

	for _, entity := range graph.Entities {
		entity.Name = utils.NormalizeName(entity.Name)
		entity.Type = utils.NormalizeName(entity.Type)
		if entity.ChunkIndex < 0 || entity.ChunkIndex >= len(chunks) {
			return models.GraphAssets{}, fmt.Errorf("entity %q has invalid chunk_index %d (batch size %d)", entity.Name, entity.ChunkIndex, len(chunks))
		}
		entityAssets = append(entityAssets, models.EntityAsset{
			Name:        entity.Name,
			Type:        entity.Type,
			Description: entity.Desc,
			ChunkIds:    []string{chunks[entity.ChunkIndex].AssetId},
		})
		entityKeyToId[entity.Name+"|"+entity.Type] = entity.Name
	}

	for _, relation := range graph.Relations {
		relation.Source.Name = utils.NormalizeName(relation.Source.Name)
		relation.Source.Type = utils.NormalizeName(relation.Source.Type)
		relation.Target.Name = utils.NormalizeName(relation.Target.Name)
		relation.Target.Type = utils.NormalizeName(relation.Target.Type)
		if relation.ChunkIndex < 0 || relation.ChunkIndex >= len(chunks) {
			return models.GraphAssets{}, fmt.Errorf("relation %q→%q has invalid chunk_index %d (batch size %d)", relation.Source.Name, relation.Target.Name, relation.ChunkIndex, len(chunks))
		}
		if _, ok := entityKeyToId[relation.Source.Name+"|"+relation.Source.Type]; !ok {
			return models.GraphAssets{}, fmt.Errorf("relation source entity %q not found", relation.Source.Name)
		}
		if _, ok := entityKeyToId[relation.Target.Name+"|"+relation.Target.Type]; !ok {
			return models.GraphAssets{}, fmt.Errorf("relation target entity %q not found", relation.Target.Name)
		}
		relationAssets = append(relationAssets, models.RelationAsset{
			Source:      relation.Source.Name,
			SourceType:  relation.Source.Type,
			Target:      relation.Target.Name,
			TargetType:  relation.Target.Type,
			Description: relation.Desc,
			ChunkIds:    []string{chunks[relation.ChunkIndex].AssetId},
		})
	}

	return models.GraphAssets{EntityAssets: entityAssets, RelationAssets: relationAssets}, nil
}
```

- [ ] **Step 4: Update `ExtractGraphFromChunk` to be a thin wrapper**

Replace the entire `ExtractGraphFromChunk` method body:

```go
func (f *FastGraphExtractor) ExtractGraphFromChunk(ctx context.Context, chunk asset_manager.ContextualAsset, options GraphExtractionOptions) (models.GraphAssets, error) {
	return f.ExtractGraphFromChunks(ctx, []asset_manager.ContextualAsset{chunk}, options)
}
```

- [ ] **Step 5: Run the tests — confirm they pass**

```bash
go test ./graph/... -run TestExtractGraphFromChunks -v
```
Expected:
```
--- PASS: TestExtractGraphFromChunks_MapChunkIndex (0.00s)
--- PASS: TestExtractGraphFromChunks_InvalidChunkIndex (0.00s)
PASS
```

- [ ] **Step 6: Commit**

```bash
git add graph/extractor.go graph/extractor_test.go
git commit -m "feat: implement ExtractGraphFromChunks with chunk_index attribution"
```

---

## Task 6: Update `ExtractGraph` to dispatch batches + use cached tools

**Files:**
- Modify: `graph/extractor.go`
- Modify: `graph/extractor_test.go` (add batch test)

- [ ] **Step 1: Add the batch-dispatch test**

Append to `graph/extractor_test.go`:

```go
// TestExtractGraph_BatchesChunks verifies that ExtractGraph splits N chunks into
// ceil(N/BatchSize) LLM calls and merges all entities/relations correctly.
func TestExtractGraph_BatchesChunks(t *testing.T) {
	// BatchSize=2, 5 chunks → 3 LLM calls (batches of 2, 2, 1)
	chunks := []asset_manager.ContextualAsset{
		{AssetId: "c0", Content: "chunk 0"},
		{AssetId: "c1", Content: "chunk 1"},
		{AssetId: "c2", Content: "chunk 2"},
		{AssetId: "c3", Content: "chunk 3"},
		{AssetId: "c4", Content: "chunk 4"},
	}

	// batch 0: entities Alice(c0), Bob(c1) + relation
	batch0 := toolCallMsg(models.ExtractedGraph{
		Entities: []models.GraphEntity{
			{Name: "Alice", Type: "person", Desc: "p", ChunkIndex: 0},
			{Name: "Bob",   Type: "person", Desc: "p", ChunkIndex: 1},
		},
		Relations: []models.GraphRelation{
			{Source: models.RelativeEntity{Name: "Alice", Type: "person"}, Target: models.RelativeEntity{Name: "Bob", Type: "person"}, Desc: "knows", ChunkIndex: 0},
		},
	})
	// batch 1: entity Carol(c2), Dave(c3)
	batch1 := toolCallMsg(models.ExtractedGraph{
		Entities: []models.GraphEntity{
			{Name: "Carol", Type: "person", Desc: "p", ChunkIndex: 0},
			{Name: "Dave",  Type: "person", Desc: "p", ChunkIndex: 1},
		},
		Relations: []models.GraphRelation{
			{Source: models.RelativeEntity{Name: "Carol", Type: "person"}, Target: models.RelativeEntity{Name: "Dave", Type: "person"}, Desc: "knows", ChunkIndex: 0},
		},
	})
	// batch 2: entity Eve(c4)
	batch2 := toolCallMsg(models.ExtractedGraph{
		Entities: []models.GraphEntity{
			{Name: "Eve", Type: "person", Desc: "p", ChunkIndex: 0},
		},
		Relations: []models.GraphRelation{},
	})

	e := newTestExtractor([]openai.ChatCompletionMessage{batch0, batch1, batch2})
	e.BatchSize = 2
	e.MaxConcurrent = 1 // serial for deterministic call count in tests

	opts := GraphExtractionOptions{Domain: "test", EntityTypes: []string{"person"}}
	result, err := e.ExtractGraph(context.Background(), chunks, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if e.LLM.(*mockLLM).callCount != 3 {
		t.Errorf("expected 3 LLM calls (3 batches), got %d", e.LLM.(*mockLLM).callCount)
	}

	if len(result.EntityAssets) != 5 {
		t.Errorf("expected 5 entities, got %d", len(result.EntityAssets))
	}

	// Verify ChunkIds map back to the original chunk AssetIds
	entityChunkMap := map[string]string{}
	for _, ea := range result.EntityAssets {
		entityChunkMap[ea.Name] = ea.ChunkIds[0]
	}
	if entityChunkMap["alice"] != "c0" {
		t.Errorf("alice ChunkId = %q, want c0", entityChunkMap["alice"])
	}
	if entityChunkMap["bob"] != "c1" {
		t.Errorf("bob ChunkId = %q, want c1", entityChunkMap["bob"])
	}
	if entityChunkMap["carol"] != "c2" {
		t.Errorf("carol ChunkId = %q, want c2", entityChunkMap["carol"])
	}
	if entityChunkMap["dave"] != "c3" {
		t.Errorf("dave ChunkId = %q, want c3", entityChunkMap["dave"])
	}
	if entityChunkMap["eve"] != "c4" {
		t.Errorf("eve ChunkId = %q, want c4", entityChunkMap["eve"])
	}
}
```

- [ ] **Step 2: Run to confirm test fails**

```bash
go test ./graph/... -run TestExtractGraph_BatchesChunks -v 2>&1 | head -20
```
Expected: FAIL — `ExtractGraph` still calls per-chunk, not per-batch.

- [ ] **Step 3: Rewrite `ExtractGraph` in `graph/extractor.go` to dispatch batches**

Replace the entire `ExtractGraph` method with:

```go
func (f *FastGraphExtractor) ExtractGraph(ctx context.Context, chunks []asset_manager.ContextualAsset, options GraphExtractionOptions) (models.GraphAssets, error) {
	var mu sync.Mutex
	var wg sync.WaitGroup

	logger := asset_manager.GetLogger(f)
	logger.InfoContext(ctx, "starting to extract graph", slog.Int("chunks", len(chunks)), slog.Any("options", options))

	entityAssetsGroup := make(map[string]models.EntityAsset)
	relationAssetsGroup := make(map[string][]models.RelationAsset)

	batchSize := f.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}
	batches := make([][]asset_manager.ContextualAsset, 0, (len(chunks)+batchSize-1)/batchSize)
	for i := 0; i < len(chunks); i += batchSize {
		end := min(i+batchSize, len(chunks))
		batches = append(batches, chunks[i:end])
	}

	errChan := make(chan error, len(batches))
	sem := make(chan struct{}, f.MaxConcurrent)
	for _, batch := range batches {
		wg.Add(1)
		sem <- struct{}{}
		go func(batch []asset_manager.ContextualAsset) {
			defer wg.Done()
			defer func() { <-sem }()

			maxAttempt := max(f.MaxRetryAttempt, 1)
			var graph models.GraphAssets
			var err error
			for i := 0; i < maxAttempt; i++ {
				logger.DebugContext(ctx, "extracting graph from batch", slog.Int("batchSize", len(batch)), slog.Int("attempt", i+1))
				graph, err = f.ExtractGraphFromChunks(ctx, batch, options)
				if err == nil {
					break
				}
				logger.Warn("extracting graph from batch failed: "+err.Error(), slog.Int("batchSize", len(batch)), slog.Int("attempt", i+1))
			}
			if err != nil {
				errChan <- err
				return
			}
			mu.Lock()
			for _, e := range graph.EntityAssets {
				entityKey := e.Name + "|" + e.Type
				if _, ok := entityAssetsGroup[entityKey]; !ok {
					entityAssetsGroup[entityKey] = e
				} else {
					logger.Warn("found duplicate entity, merging", slog.String("entity", e.Name), slog.String("type", e.Type))
					existing := entityAssetsGroup[entityKey]
					existing.Description = existing.Description + "\n" + e.Description
					existing.ChunkIds = append(existing.ChunkIds, e.ChunkIds...)
					entityAssetsGroup[entityKey] = existing
				}
			}
			for _, r := range graph.RelationAssets {
				keys := []string{r.Source + "|" + r.SourceType, r.Target + "|" + r.TargetType}
				slices.Sort(keys)
				relationKey := strings.Join(keys, "|")
				relationAssetsGroup[relationKey] = append(relationAssetsGroup[relationKey], r)
			}
			mu.Unlock()
		}(batch)
	}
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case err := <-errChan:
		logger.ErrorContext(ctx, "extracting graph from batch failed: "+err.Error())
		return models.GraphAssets{}, err
	case <-done:
	}

	entityKeyToId := make(map[string]string)
	entityAssets := []models.EntityAsset{}
	for _, entity := range entityAssetsGroup {
		entityKeyToId[entity.Name+"|"+entity.Type] = entity.Name
		entityAssets = append(entityAssets, entity)
	}

	logger.DebugContext(ctx, "deduping relations", slog.Int("relations", len(relationAssetsGroup)))
	relationAssets := []models.RelationAsset{}
	dedupSem := make(chan struct{}, f.MaxConcurrent)
	dedupErrChan := make(chan error, len(relationAssetsGroup))
	for rk, relationWithSameKey := range relationAssetsGroup {
		wg.Add(1)
		dedupSem <- struct{}{}
		go func(relGroup []models.RelationAsset) {
			defer wg.Done()
			defer func() { <-dedupSem }()

			if len(relGroup) > 1 {
				var newRelations []models.RelationAsset
				var err error
				for i := 0; i < f.MaxRetryAttempt; i++ {
					logger.Debug("deduping relations in a group", slog.String("relationGroupKey", rk), slog.Int("relations", len(relGroup)), slog.Int("attempt", i+1))
					newRelations, err = f.DeduplicateRelations(ctx, relGroup)
					if err == nil {
						isFailed := false
						for _, relation := range newRelations {
							sourceKey := relation.Source + "|" + relation.SourceType
							targetKey := relation.Target + "|" + relation.TargetType
							if _, ok := entityKeyToId[sourceKey]; !ok {
								if i == f.MaxRetryAttempt-1 {
									err = fmt.Errorf("relation bind to source entity %s which is not found", sourceKey)
								}
								isFailed = true
							}
							if _, ok := entityKeyToId[targetKey]; !ok {
								if i == f.MaxRetryAttempt-1 {
									err = fmt.Errorf("relation bind to target entity %s which is not found", targetKey)
								}
								isFailed = true
							}
						}
						if !isFailed {
							break
						} else {
							logger.Warn("deduping relations failed", slog.String("error", err.Error()))
						}
					} else {
						logger.Warn("deduping relations failed", slog.String("error", err.Error()))
					}
				}
				if err != nil {
					dedupErrChan <- err
					return
				}
				mu.Lock()
				relationAssets = append(relationAssets, newRelations...)
				mu.Unlock()
			} else {
				mu.Lock()
				relationAssets = append(relationAssets, relGroup[0])
				mu.Unlock()
			}
		}(relationWithSameKey)
	}
	doneDedup := make(chan struct{})
	go func() {
		wg.Wait()
		close(doneDedup)
	}()
	select {
	case err := <-dedupErrChan:
		logger.ErrorContext(ctx, "deduping relations failed: "+err.Error())
		return models.GraphAssets{}, err
	case <-doneDedup:
	}

	logger.InfoContext(ctx, "deduping relations finished", slog.Int("entities", len(entityAssets)), slog.Int("relations", len(relationAssets)))
	return models.GraphAssets{
		EntityAssets:   entityAssets,
		RelationAssets: relationAssets,
	}, nil
}
```

- [ ] **Step 4: Use cached tool in `DeduplicateRelations`**

In `DeduplicateRelations`, replace:
```go
relationClusterTool, err := models.GetRelationClusterTool()
if err != nil {
    return []models.RelationAsset{}, err
}
```
With:
```go
relationClusterTool := f.deduplicationTool
```
And remove the now-unused `err` variable (adjust the surrounding code if needed).

- [ ] **Step 5: Use cached tool in `ExtractEntitiesFromQuery`**

In `ExtractEntitiesFromQuery`, replace:
```go
queryExtractionTool, err := models.GetQueryExtractionTool()
if err != nil {
    logger.ErrorContext(ctx, "failed to get query extraction tool: "+err.Error())
    return models.ExtractedQuery{}, err
}
```
With:
```go
queryExtractionTool := f.queryTool
```

- [ ] **Step 6: Run all graph tests**

```bash
go test ./graph/... -v
```
Expected:
```
--- PASS: TestExtractGraphFromChunks_MapChunkIndex (0.00s)
--- PASS: TestExtractGraphFromChunks_InvalidChunkIndex (0.00s)
--- PASS: TestExtractGraph_BatchesChunks (0.00s)
PASS
```

- [ ] **Step 7: Full build check**

```bash
go build ./...
```
Expected: no output.

- [ ] **Step 8: Commit**

```bash
git add graph/extractor.go graph/extractor_test.go
git commit -m "feat: ExtractGraph dispatches batches, use cached tool schemas throughout extractor"
```

---

## Task 7: Add `WithBatchSize` option and wire into service

**Files:**
- Modify: `rag/options.go`

- [ ] **Step 1: Add `BatchSize` to `GraphRAGServiceCompileTimeOptions`**

In `rag/options.go`, update the struct:

```go
type GraphRAGServiceCompileTimeOptions struct {
	MaxConcurrent     int `json:"max_concurrent"`
	BatchSize         int `json:"batch_size"`
	LLM               llm.LLM
	ChunkingExtractor chunk.ChunkExtractor
	GraphExtractor    graph.GraphExtractor
}
```

- [ ] **Step 2: Set default in `NewGraphRAGServiceOptions`**

In `NewGraphRAGServiceOptions`, update the compile-time defaults:

```go
GraphRAGServiceCompileTimeOptions: GraphRAGServiceCompileTimeOptions{
    MaxConcurrent: utils.DefaultMaxConcurrent,
    BatchSize:     utils.DefaultBatchSize,
},
```

- [ ] **Step 3: Add `WithBatchSize` option function**

After the existing `WithMaxConcurrent` function, add:

```go
func WithBatchSize(batchSize int) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.BatchSize = batchSize
		opts.GraphExtractor.SetBatchSize(batchSize)
	}
}
```

- [ ] **Step 4: Verify it compiles and all tests pass**

```bash
go build ./... && go test ./...
```
Expected: clean build, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add rag/options.go
git commit -m "feat: add WithBatchSize option to GraphRAGService"
```

---

## Self-Review Checklist

- [x] **Spec: DefaultBatchSize=5, DefaultMaxConcurrent=3** — Task 1
- [x] **Spec: ChunkIndex on GraphEntity/GraphRelation** — Task 2
- [x] **Spec: Multi-chunk prompt format with chunk attribution instruction** — Task 3
- [x] **Spec: Tools pre-computed at constructor** — Task 4
- [x] **Spec: ExtractGraphFromChunks with chunk_index→ChunkIds mapping** — Task 5
- [x] **Spec: ExtractGraph batches chunks** — Task 6
- [x] **Spec: ExtractGraphFromChunk kept as thin wrapper** — Task 6
- [x] **Spec: WithBatchSize option** — Task 7
- [x] **Spec: ChunkIds preserved for downstream retrieval** — enforced by chunk_index validation
- [x] **Type consistency**: `ExtractGraphFromChunks` signature matches calls from `ExtractGraphFromChunk` and `ExtractGraph`
- [x] **No placeholders or TBDs** — all code blocks are complete
