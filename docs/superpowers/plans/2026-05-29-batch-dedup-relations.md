# Batch Relation Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-pair-group LLM dedup calls with batched calls that process N groups in one request, reducing 50+ LLM calls to ceil(50/10) = 5.

**Architecture:** `FastGraphExtractor` gains a `DedupBatchSize` field (default 10). `ExtractGraph()` splits groups-needing-dedup into sub-batches and calls the new `DeduplicateRelationsBatch()` method—one LLM call per sub-batch—behind the existing `MaxConcurrent` semaphore. Single-relation groups still bypass LLM entirely.

**Tech Stack:** Go 1.22+, `github.com/openai/openai-go`, `github.com/manageai-inet/agentic-assets`, existing `mockLLM` in `graph/extractor_test.go`.

---

## File Map

| File | Change |
|------|--------|
| `utils/const.go` | Add `DefaultDedupBatchSize = 10` |
| `models/graph.go` | Add `PairGroupResult`, `BatchRelationClusters`, `GetBatchRelationClusterTool()` |
| `models/graph_test.go` | New — test `GetBatchRelationClusterTool()` |
| `prompts/prompts.go` | Add `DeduplicateBatchRelationsSystem`, `DeduplicateBatchRelationsPrompt` |
| `graph/extractor.go` | Add `DedupBatchSize` + `batchDeduplicationTool` fields; update `NewFastGraphExtractor()`; add `DeduplicateRelationsBatch()`; replace dedup loop in `ExtractGraph()` |
| `graph/extractor_test.go` | Add tests for `DeduplicateRelationsBatch` and updated `ExtractGraph` |

---

### Task 1: Add constant, model types, and tool function

**Files:**
- Modify: `utils/const.go`
- Modify: `models/graph.go`
- Create: `models/graph_test.go`

- [ ] **Step 1: Add DefaultDedupBatchSize constant to `utils/const.go`**

  Open `utils/const.go`. After the existing `DefaultEmbedBatchSize` line, add:

  ```go
  const DefaultDedupBatchSize = 10
  ```

  The relevant section should look like:
  ```go
  const DefaultEmbedBatchSize = 32
  const DefaultDedupBatchSize = 10
  ```

- [ ] **Step 2: Add new model types and tool function to `models/graph.go`**

  Open `models/graph.go`. After the closing brace of `GetRelationClusterTool()` (currently around line 63), add:

  ```go
  type PairGroupResult struct {
  	GroupId  int                      `json:"group_id"  jsonschema:"The 0-based index of this group, matching the GROUP N label in the input"`
  	Clusters map[string]RelationGroup `json:"clusters"  jsonschema:"Map of relation clusters for this group, key is cluster id"`
  }

  type BatchRelationClusters struct {
  	Groups []PairGroupResult `json:"groups" jsonschema:"Array of results, one per input GROUP; every GROUP N in the input must appear here"`
  }

  func GetBatchRelationClusterTool() (openai.ChatCompletionToolParam, error) {
  	name := "batch_relation_cluster"
  	description := "Cluster relations for multiple entity pairs at once; one entry per GROUP in the input"
  	tool, err := utils.StructToTool[BatchRelationClusters](name, description)
  	if err != nil {
  		return openai.ChatCompletionToolParam{}, err
  	}
  	return tool, nil
  }
  ```

- [ ] **Step 3: Write the failing test in new file `models/graph_test.go`**

  ```go
  package models

  import "testing"

  func TestGetBatchRelationClusterTool_NoError(t *testing.T) {
  	tool, err := GetBatchRelationClusterTool()
  	if err != nil {
  		t.Fatalf("GetBatchRelationClusterTool() error: %v", err)
  	}
  	if tool.Function.Name != "batch_relation_cluster" {
  		t.Errorf("tool name = %q, want %q", tool.Function.Name, "batch_relation_cluster")
  	}
  }
  ```

- [ ] **Step 4: Run the test to verify it passes**

  ```bash
  cd /Users/bestty/Documents/fast-graphrago && go test ./models/... -v
  ```

  Expected output:
  ```
  === RUN   TestGetBatchRelationClusterTool_NoError
  --- PASS: TestGetBatchRelationClusterTool_NoError (0.00s)
  ok      github.com/manageai-inet/fast-graphrago/models
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add utils/const.go models/graph.go models/graph_test.go
  git commit -m "feat: add DefaultDedupBatchSize constant and BatchRelationClusters model types"
  ```

---

### Task 2: Add batch dedup prompt constants

**Files:**
- Modify: `prompts/prompts.go`

- [ ] **Step 1: Add prompt constants to `prompts/prompts.go`**

  Open `prompts/prompts.go`. After the last line of the file (the closing backtick of `DeduplicateRelationsPrompt`), add:

  ```go
  const DeduplicateBatchRelationsSystem = `### **BATCH RELATION DEDUPLICATION & CONTEXT CLUSTERING**

  **Goal**: For each GROUP in the input, cluster relationships between the same entity pair into semantic groups. Process each GROUP independently.

  **Instructions**:
  1. **Scope**: Each GROUP contains relations between one specific entity pair. Process each GROUP independently.
  2. **Contextual Split**:
     * **Merge**: "X founded Y" and "Y was started by X" (same semantic context).
     * **Separate**: "X works at Y" and "X sued Y" (different contexts).
  3. **Synthesis**: For each cluster, write one clear, comprehensive description.
  4. **group_id**: Each entry in your output "groups" array must have a "group_id" matching the GROUP N number from the input.

  **Schema**:
  * Input: multiple GROUPs, each labeled GROUP N with a local 0-based CSV of relations.
  * Output: {"groups": [{"group_id": N, "clusters": {"key": {"source": {...}, "target": {...}, "desc": "...", "indices": [...]}}}, ...]}

  **Constraint**: Return ALL groups — every GROUP N in the input must appear in "groups". Preserve original entity names and types exactly. Return valid JSON only.
  `

  const DeduplicateBatchRelationsPrompt = `
  # INPUT DATA
  <<RELATIONS_START>>
  %s
  <<RELATIONS_END>>
  `
  ```

- [ ] **Step 2: Verify the package compiles**

  ```bash
  cd /Users/bestty/Documents/fast-graphrago && go build ./prompts/...
  ```

  Expected: no output (success).

- [ ] **Step 3: Commit**

  ```bash
  git add prompts/prompts.go
  git commit -m "feat: add DeduplicateBatchRelationsSystem and DeduplicateBatchRelationsPrompt constants"
  ```

---

### Task 3: Add DedupBatchSize field and DeduplicateRelationsBatch method

**Files:**
- Modify: `graph/extractor.go`
- Modify: `graph/extractor_test.go`

- [ ] **Step 1: Update the `FastGraphExtractor` struct in `graph/extractor.go`**

  Find the struct declaration (lines 21–30). Replace it with:

  ```go
  type FastGraphExtractor struct {
  	LLM                    llm.LLM
  	MaxRetryAttempt        int
  	MaxConcurrent          int
  	BatchSize              int
  	DedupBatchSize         int
  	extractionTool         openai.ChatCompletionToolParam
  	deduplicationTool      openai.ChatCompletionToolParam
  	batchDeduplicationTool openai.ChatCompletionToolParam
  	queryTool              openai.ChatCompletionToolParam
  	asset_manager.LoggingCapacity
  }
  ```

- [ ] **Step 2: Update `NewFastGraphExtractor()` to initialize the new fields**

  Find `NewFastGraphExtractor()` (lines 32–55). Replace it with:

  ```go
  func NewFastGraphExtractor(llm llm.LLM) *FastGraphExtractor {
  	extractionTool, err := models.GetGraphExtractionTool()
  	if err != nil {
  		panic("failed to build extraction tool schema: " + err.Error())
  	}
  	deduplicationTool, err := models.GetRelationClusterTool()
  	if err != nil {
  		panic("failed to build deduplication tool schema: " + err.Error())
  	}
  	batchDeduplicationTool, err := models.GetBatchRelationClusterTool()
  	if err != nil {
  		panic("failed to build batch deduplication tool schema: " + err.Error())
  	}
  	queryTool, err := models.GetQueryExtractionTool()
  	if err != nil {
  		panic("failed to build query tool schema: " + err.Error())
  	}
  	return &FastGraphExtractor{
  		LLM:                    llm,
  		MaxRetryAttempt:        utils.DefaultMaxRetryAttempt,
  		MaxConcurrent:          utils.DefaultMaxConcurrent,
  		BatchSize:              utils.DefaultBatchSize,
  		DedupBatchSize:         utils.DefaultDedupBatchSize,
  		extractionTool:         extractionTool,
  		deduplicationTool:      deduplicationTool,
  		batchDeduplicationTool: batchDeduplicationTool,
  		queryTool:              queryTool,
  		LoggingCapacity:        *asset_manager.GetDefaultLoggingCapacity(),
  	}
  }
  ```

- [ ] **Step 3: Add `DeduplicateRelationsBatch()` to `graph/extractor.go`**

  Insert this method immediately after the closing brace of `DeduplicateRelations()` (currently around line 293):

  ```go
  // DeduplicateRelationsBatch sends multiple pair groups in one LLM call and returns
  // deduplicated relations for each group. Indices in each group's clusters are local
  // (0-based within that group's slice), matching the GROUP N CSV in the prompt.
  func (f *FastGraphExtractor) DeduplicateRelationsBatch(ctx context.Context, groups [][]models.RelationAsset) ([][]models.RelationAsset, error) {
  	if len(groups) == 0 {
  		return nil, nil
  	}

  	var sb strings.Builder
  	for gi, group := range groups {
  		src := group[0].Source + "|" + group[0].SourceType
  		tgt := group[0].Target + "|" + group[0].TargetType
  		fmt.Fprintf(&sb, "GROUP %d (%s → %s):\n", gi, src, tgt)
  		sb.WriteString("index,source_name,source_type,target_name,target_type,desc\n")
  		for i, r := range group {
  			fmt.Fprintf(&sb, "%d,%s,%s,%s,%s,%s\n", i, r.Source, r.SourceType, r.Target, r.TargetType, r.Description)
  		}
  		sb.WriteString("\n")
  	}

  	messages := []openai.ChatCompletionMessage{
  		{Role: "system", Content: prompts.DeduplicateBatchRelationsSystem},
  		{Role: "user", Content: fmt.Sprintf(prompts.DeduplicateBatchRelationsPrompt, sb.String())},
  	}
  	toolChoice := openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("required")}
  	generation, err := f.LLM.Generate(ctx, messages, []openai.ChatCompletionToolParam{f.batchDeduplicationTool}, &toolChoice)
  	if err != nil {
  		return nil, err
  	}
  	if len(generation.ToolCalls) == 0 {
  		return nil, fmt.Errorf("no tool calls found")
  	}

  	batchClusters, err := utils.HandleToolCall[models.BatchRelationClusters](generation.ToolCalls[0].Function.Arguments)
  	if err != nil {
  		return nil, err
  	}

  	results := make([][]models.RelationAsset, len(groups))
  	for _, groupResult := range batchClusters.Groups {
  		gi := groupResult.GroupId
  		if gi < 0 || gi >= len(groups) {
  			return nil, fmt.Errorf("invalid group_id %d in LLM response (batch has %d groups)", gi, len(groups))
  		}
  		srcGroup := groups[gi]
  		newRelations := make([]models.RelationAsset, 0, len(groupResult.Clusters))
  		for _, cluster := range groupResult.Clusters {
  			chunkIds := []string{}
  			for _, idx := range cluster.Indices {
  				if idx < 0 || idx >= len(srcGroup) {
  					return nil, fmt.Errorf("invalid relation index %d in group %d (group has %d relations)", idx, gi, len(srcGroup))
  				}
  				chunkIds = append(chunkIds, srcGroup[idx].ChunkIds...)
  			}
  			newRelations = append(newRelations, models.RelationAsset{
  				Source:      utils.NormalizeName(cluster.Source.Name),
  				SourceType:  utils.NormalizeName(cluster.Source.Type),
  				Target:      utils.NormalizeName(cluster.Target.Name),
  				TargetType:  utils.NormalizeName(cluster.Target.Type),
  				Description: cluster.Desc,
  				ChunkIds:    chunkIds,
  			})
  		}
  		results[gi] = newRelations
  	}

  	for gi, r := range results {
  		if r == nil {
  			return nil, fmt.Errorf("LLM did not return result for group %d", gi)
  		}
  	}

  	return results, nil
  }
  ```

- [ ] **Step 4: Write failing tests in `graph/extractor_test.go`**

  Append these tests to `graph/extractor_test.go`:

  ```go
  // TestDeduplicateRelationsBatch_Basic verifies two groups are merged in one LLM call.
  func TestDeduplicateRelationsBatch_Basic(t *testing.T) {
  	groups := [][]models.RelationAsset{
  		{
  			{Source: "alice", SourceType: "person", Target: "bob", TargetType: "person", Description: "works with", ChunkIds: []string{"c0"}},
  			{Source: "alice", SourceType: "person", Target: "bob", TargetType: "person", Description: "collaborated with", ChunkIds: []string{"c1"}},
  		},
  		{
  			{Source: "acme", SourceType: "organization", Target: "globex", TargetType: "organization", Description: "partner of", ChunkIds: []string{"c2"}},
  			{Source: "acme", SourceType: "organization", Target: "globex", TargetType: "organization", Description: "strategic alliance", ChunkIds: []string{"c3"}},
  		},
  	}

  	resp := toolCallMsg(models.BatchRelationClusters{
  		Groups: []models.PairGroupResult{
  			{
  				GroupId: 0,
  				Clusters: map[string]models.RelationGroup{
  					"g0": {Source: models.RelativeEntity{Name: "alice", Type: "person"}, Target: models.RelativeEntity{Name: "bob", Type: "person"}, Desc: "works and collaborates with", Indices: []int{0, 1}},
  				},
  			},
  			{
  				GroupId: 1,
  				Clusters: map[string]models.RelationGroup{
  					"g0": {Source: models.RelativeEntity{Name: "acme", Type: "organization"}, Target: models.RelativeEntity{Name: "globex", Type: "organization"}, Desc: "partner and strategic ally", Indices: []int{0, 1}},
  				},
  			},
  		},
  	})

  	e := newTestExtractor([]openai.ChatCompletionMessage{resp})
  	results, err := e.DeduplicateRelationsBatch(context.Background(), groups)
  	if err != nil {
  		t.Fatalf("unexpected error: %v", err)
  	}
  	if len(results) != 2 {
  		t.Fatalf("expected 2 group results, got %d", len(results))
  	}
  	if len(results[0]) != 1 {
  		t.Errorf("group 0: expected 1 merged relation, got %d", len(results[0]))
  	}
  	if len(results[1]) != 1 {
  		t.Errorf("group 1: expected 1 merged relation, got %d", len(results[1]))
  	}
  	if len(results[0][0].ChunkIds) != 2 {
  		t.Errorf("group 0 ChunkIds: want 2, got %d", len(results[0][0].ChunkIds))
  	}
  	if e.LLM.(*mockLLM).callCount != 1 {
  		t.Errorf("expected 1 LLM call, got %d", e.LLM.(*mockLLM).callCount)
  	}
  }

  // TestDeduplicateRelationsBatch_Empty verifies nil input returns nil without calling LLM.
  func TestDeduplicateRelationsBatch_Empty(t *testing.T) {
  	e := newTestExtractor(nil)
  	results, err := e.DeduplicateRelationsBatch(context.Background(), nil)
  	if err != nil {
  		t.Fatalf("unexpected error: %v", err)
  	}
  	if results != nil {
  		t.Errorf("expected nil results for empty input, got %v", results)
  	}
  	if e.LLM.(*mockLLM).callCount != 0 {
  		t.Errorf("expected 0 LLM calls for empty input, got %d", e.LLM.(*mockLLM).callCount)
  	}
  }

  // TestDeduplicateRelationsBatch_InvalidGroupId verifies an out-of-range group_id causes an error.
  func TestDeduplicateRelationsBatch_InvalidGroupId(t *testing.T) {
  	groups := [][]models.RelationAsset{
  		{
  			{Source: "alice", SourceType: "person", Target: "bob", TargetType: "person", Description: "works with", ChunkIds: []string{"c0"}},
  			{Source: "alice", SourceType: "person", Target: "bob", TargetType: "person", Description: "knows", ChunkIds: []string{"c1"}},
  		},
  	}
  	resp := toolCallMsg(models.BatchRelationClusters{
  		Groups: []models.PairGroupResult{
  			{
  				GroupId: 99,
  				Clusters: map[string]models.RelationGroup{
  					"g0": {Source: models.RelativeEntity{Name: "alice", Type: "person"}, Target: models.RelativeEntity{Name: "bob", Type: "person"}, Desc: "...", Indices: []int{0}},
  				},
  			},
  		},
  	})
  	e := newTestExtractor([]openai.ChatCompletionMessage{resp})
  	_, err := e.DeduplicateRelationsBatch(context.Background(), groups)
  	if err == nil {
  		t.Fatal("expected error for invalid group_id, got nil")
  	}
  }

  // TestDeduplicateRelationsBatch_MissingGroup verifies an error when a group_id is absent from the response.
  func TestDeduplicateRelationsBatch_MissingGroup(t *testing.T) {
  	groups := [][]models.RelationAsset{
  		{
  			{Source: "alice", SourceType: "person", Target: "bob", TargetType: "person", Description: "works with", ChunkIds: []string{"c0"}},
  			{Source: "alice", SourceType: "person", Target: "bob", TargetType: "person", Description: "knows", ChunkIds: []string{"c1"}},
  		},
  		{
  			{Source: "acme", SourceType: "organization", Target: "globex", TargetType: "organization", Description: "partner of", ChunkIds: []string{"c2"}},
  			{Source: "acme", SourceType: "organization", Target: "globex", TargetType: "organization", Description: "ally", ChunkIds: []string{"c3"}},
  		},
  	}
  	// LLM only returns group 0, omits group 1
  	resp := toolCallMsg(models.BatchRelationClusters{
  		Groups: []models.PairGroupResult{
  			{
  				GroupId: 0,
  				Clusters: map[string]models.RelationGroup{
  					"g0": {Source: models.RelativeEntity{Name: "alice", Type: "person"}, Target: models.RelativeEntity{Name: "bob", Type: "person"}, Desc: "...", Indices: []int{0, 1}},
  				},
  			},
  		},
  	})
  	e := newTestExtractor([]openai.ChatCompletionMessage{resp})
  	_, err := e.DeduplicateRelationsBatch(context.Background(), groups)
  	if err == nil {
  		t.Fatal("expected error when group missing from response, got nil")
  	}
  }

  // TestDeduplicateRelationsBatch_InvalidRelationIndex verifies an out-of-range index within a cluster causes an error.
  func TestDeduplicateRelationsBatch_InvalidRelationIndex(t *testing.T) {
  	groups := [][]models.RelationAsset{
  		{
  			{Source: "alice", SourceType: "person", Target: "bob", TargetType: "person", Description: "works with", ChunkIds: []string{"c0"}},
  			{Source: "alice", SourceType: "person", Target: "bob", TargetType: "person", Description: "knows", ChunkIds: []string{"c1"}},
  		},
  	}
  	resp := toolCallMsg(models.BatchRelationClusters{
  		Groups: []models.PairGroupResult{
  			{
  				GroupId: 0,
  				Clusters: map[string]models.RelationGroup{
  					"g0": {Source: models.RelativeEntity{Name: "alice", Type: "person"}, Target: models.RelativeEntity{Name: "bob", Type: "person"}, Desc: "...", Indices: []int{0, 99}},
  				},
  			},
  		},
  	})
  	e := newTestExtractor([]openai.ChatCompletionMessage{resp})
  	_, err := e.DeduplicateRelationsBatch(context.Background(), groups)
  	if err == nil {
  		t.Fatal("expected error for out-of-range relation index, got nil")
  	}
  }
  ```

- [ ] **Step 5: Run the tests to verify they pass**

  ```bash
  cd /Users/bestty/Documents/fast-graphrago && go test ./graph/... -run TestDeduplicateRelationsBatch -v
  ```

  Expected output:
  ```
  === RUN   TestDeduplicateRelationsBatch_Basic
  --- PASS: TestDeduplicateRelationsBatch_Basic (0.00s)
  === RUN   TestDeduplicateRelationsBatch_Empty
  --- PASS: TestDeduplicateRelationsBatch_Empty (0.00s)
  === RUN   TestDeduplicateRelationsBatch_InvalidGroupId
  --- PASS: TestDeduplicateRelationsBatch_InvalidGroupId (0.00s)
  === RUN   TestDeduplicateRelationsBatch_MissingGroup
  --- PASS: TestDeduplicateRelationsBatch_MissingGroup (0.00s)
  === RUN   TestDeduplicateRelationsBatch_InvalidRelationIndex
  --- PASS: TestDeduplicateRelationsBatch_InvalidRelationIndex (0.00s)
  ok      github.com/manageai-inet/fast-graphrago/graph
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add graph/extractor.go graph/extractor_test.go
  git commit -m "feat: add DedupBatchSize field and DeduplicateRelationsBatch method"
  ```

---

### Task 4: Replace the dedup loop in ExtractGraph()

**Files:**
- Modify: `graph/extractor.go`
- Modify: `graph/extractor_test.go`

- [ ] **Step 1: Write the failing integration test first (append to `graph/extractor_test.go`)**

  ```go
  // TestExtractGraph_BatchedDedup verifies ExtractGraph uses one dedup LLM call for a
  // pair group with >1 relation, and that single-relation groups bypass LLM entirely.
  func TestExtractGraph_BatchedDedup(t *testing.T) {
  	// 1 chunk with 2 entities and 2 relations for the same pair (alice→bob)
  	// plus 1 single-relation pair (carol→dave) that must NOT trigger a dedup call.
  	chunks := []asset_manager.ContextualAsset{
  		{AssetId: "c0", Content: "Alice works with Bob. Carol manages Dave."},
  	}

  	extractionResp := toolCallMsg(models.ExtractedGraph{
  		Entities: []models.GraphEntity{
  			{Name: "Alice", Type: "person", Desc: "employee", ChunkIndex: 0},
  			{Name: "Bob", Type: "person", Desc: "employee", ChunkIndex: 0},
  			{Name: "Carol", Type: "person", Desc: "manager", ChunkIndex: 0},
  			{Name: "Dave", Type: "person", Desc: "employee", ChunkIndex: 0},
  		},
  		Relations: []models.GraphRelation{
  			{Source: models.RelativeEntity{Name: "Alice", Type: "person"}, Target: models.RelativeEntity{Name: "Bob", Type: "person"}, Desc: "works with", ChunkIndex: 0},
  			{Source: models.RelativeEntity{Name: "Alice", Type: "person"}, Target: models.RelativeEntity{Name: "Bob", Type: "person"}, Desc: "collaborates with", ChunkIndex: 0},
  			{Source: models.RelativeEntity{Name: "Carol", Type: "person"}, Target: models.RelativeEntity{Name: "Dave", Type: "person"}, Desc: "manages", ChunkIndex: 0},
  		},
  	})

  	dedupResp := toolCallMsg(models.BatchRelationClusters{
  		Groups: []models.PairGroupResult{
  			{
  				GroupId: 0,
  				Clusters: map[string]models.RelationGroup{
  					"g0": {
  						Source:  models.RelativeEntity{Name: "Alice", Type: "person"},
  						Target:  models.RelativeEntity{Name: "Bob", Type: "person"},
  						Desc:    "works and collaborates with",
  						Indices: []int{0, 1},
  					},
  				},
  			},
  		},
  	})

  	e := newTestExtractor([]openai.ChatCompletionMessage{extractionResp, dedupResp})
  	e.BatchSize = 1
  	e.MaxConcurrent = 1
  	e.DedupBatchSize = 10 // both groups fit in one dedup call

  	opts := GraphExtractionOptions{Domain: "test", EntityTypes: []string{"person"}}
  	result, err := e.ExtractGraph(context.Background(), chunks, opts)
  	if err != nil {
  		t.Fatalf("unexpected error: %v", err)
  	}

  	// 1 extraction call + 1 dedup call (for the alice-bob pair) = 2 total
  	if e.LLM.(*mockLLM).callCount != 2 {
  		t.Errorf("expected 2 LLM calls, got %d", e.LLM.(*mockLLM).callCount)
  	}

  	// alice-bob pair merged to 1 relation; carol-dave passes through = 2 total relations
  	if len(result.RelationAssets) != 2 {
  		t.Errorf("expected 2 relations, got %d", len(result.RelationAssets))
  	}

  	// the merged alice-bob relation should reference both source chunks
  	var aliceBobRel *models.RelationAsset
  	for i := range result.RelationAssets {
  		r := &result.RelationAssets[i]
  		if r.Source == "alice" && r.Target == "bob" {
  			aliceBobRel = r
  			break
  		}
  	}
  	if aliceBobRel == nil {
  		t.Fatal("expected alice→bob relation in output, not found")
  	}
  	if len(aliceBobRel.ChunkIds) != 2 {
  		t.Errorf("alice→bob ChunkIds: want 2, got %d", len(aliceBobRel.ChunkIds))
  	}
  }
  ```

- [ ] **Step 2: Run the new test to verify it fails**

  ```bash
  cd /Users/bestty/Documents/fast-graphrago && go test ./graph/... -run TestExtractGraph_BatchedDedup -v
  ```

  Expected: FAIL (the test calls `e.DedupBatchSize = 10` which compiles, but `ExtractGraph` still uses the old per-group loop so the dedup call count won't match).

- [ ] **Step 3: Replace the dedup section in `ExtractGraph()` in `graph/extractor.go`**

  Find the dedup section that begins with:
  ```go
  logger.DebugContext(ctx, "deduping relations", slog.Int("relations", len(relationAssetsGroup)))
  relationAssets := []models.RelationAsset{}
  var dedupWg sync.WaitGroup
  dedupSem := make(chan struct{}, f.MaxConcurrent)
  ```

  And ends with:
  ```go
  case <-doneDedup:
  }
  ```

  Replace that entire block (from `logger.DebugContext` through the closing `}` of the select) with:

  ```go
  logger.DebugContext(ctx, "deduping relations", slog.Int("relation_groups", len(relationAssetsGroup)))
  relationAssets := []models.RelationAsset{}

  // Separate single-relation groups (no LLM needed) from multi-relation groups.
  var needsDedup [][]models.RelationAsset
  for _, relGroup := range relationAssetsGroup {
  	if len(relGroup) > 1 {
  		needsDedup = append(needsDedup, relGroup)
  	} else {
  		relationAssets = append(relationAssets, relGroup[0])
  	}
  }

  dedupBatchSize := f.DedupBatchSize
  if dedupBatchSize <= 0 {
  	dedupBatchSize = utils.DefaultDedupBatchSize
  }

  var dedupWg sync.WaitGroup
  numSubBatches := (len(needsDedup) + dedupBatchSize - 1) / dedupBatchSize
  dedupSem := make(chan struct{}, f.MaxConcurrent)
  dedupErrChan := make(chan error, max(numSubBatches, 1))

  for start := 0; start < len(needsDedup); start += dedupBatchSize {
  	end := min(start+dedupBatchSize, len(needsDedup))
  	groups := needsDedup[start:end]

  	dedupWg.Add(1)
  	dedupSem <- struct{}{}
  	go func(groups [][]models.RelationAsset) {
  		defer dedupWg.Done()
  		defer func() { <-dedupSem }()

  		maxAttempt := max(f.MaxRetryAttempt, 1)
  		var results [][]models.RelationAsset
  		var err error
  		for attempt := 0; attempt < maxAttempt; attempt++ {
  			logger.DebugContext(ctx, "batch deduping relations", slog.Int("groups", len(groups)), slog.Int("attempt", attempt+1))
  			results, err = f.DeduplicateRelationsBatch(ctx, groups)
  			if err != nil {
  				logger.Warn("batch dedup failed: "+err.Error(), slog.Int("attempt", attempt+1))
  				continue
  			}
  			valid := true
  			for _, relGroup := range results {
  				for _, relation := range relGroup {
  					if _, ok := entityKeyToId[relation.Source+"|"+relation.SourceType]; !ok {
  						if attempt == maxAttempt-1 {
  							err = fmt.Errorf("relation source entity %s|%s not found", relation.Source, relation.SourceType)
  						}
  						valid = false
  					}
  					if _, ok := entityKeyToId[relation.Target+"|"+relation.TargetType]; !ok {
  						if attempt == maxAttempt-1 {
  							err = fmt.Errorf("relation target entity %s|%s not found", relation.Target, relation.TargetType)
  						}
  						valid = false
  					}
  				}
  			}
  			if valid {
  				break
  			}
  			if err == nil {
  				err = fmt.Errorf("entity validation failed for dedup batch")
  			}
  			logger.Warn("batch dedup entity validation failed: "+err.Error(), slog.Int("attempt", attempt+1))
  		}
  		if err != nil {
  			dedupErrChan <- err
  			return
  		}
  		mu.Lock()
  		for _, relGroup := range results {
  			relationAssets = append(relationAssets, relGroup...)
  		}
  		mu.Unlock()
  	}(groups)
  }

  doneDedup := make(chan struct{})
  go func() {
  	dedupWg.Wait()
  	close(doneDedup)
  }()
  select {
  case err := <-dedupErrChan:
  	cancel()
  	logger.ErrorContext(ctx, "deduping relations failed: "+err.Error())
  	return models.GraphAssets{}, err
  case <-doneDedup:
  }
  ```

- [ ] **Step 4: Run the new integration test to verify it passes**

  ```bash
  cd /Users/bestty/Documents/fast-graphrago && go test ./graph/... -run TestExtractGraph_BatchedDedup -v
  ```

  Expected:
  ```
  === RUN   TestExtractGraph_BatchedDedup
  --- PASS: TestExtractGraph_BatchedDedup (0.00s)
  ok      github.com/manageai-inet/fast-graphrago/graph
  ```

- [ ] **Step 5: Run the full test suite to verify no regressions**

  ```bash
  cd /Users/bestty/Documents/fast-graphrago && go test ./... -v 2>&1 | tail -30
  ```

  Expected: all tests pass. Look specifically for:
  - `ok  github.com/manageai-inet/fast-graphrago/graph`
  - `ok  github.com/manageai-inet/fast-graphrago/models`
  - `ok  github.com/manageai-inet/fast-graphrago/rag`

- [ ] **Step 6: Commit**

  ```bash
  git add graph/extractor.go graph/extractor_test.go
  git commit -m "feat: batch relation deduplication — N groups per LLM call instead of 1"
  ```
