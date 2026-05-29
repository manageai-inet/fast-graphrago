# Batch Relation Deduplication Design

## Goal

Replace the current one-LLM-call-per-entity-pair deduplication with batched calls that process N pair groups in a single LLM request, reducing total LLM calls from 50+ to ceil(50/batchSize).

## Background

`graph/extractor.go:ExtractGraph()` groups relations by entity pair key. Any pair with >1 relation is sent to `DeduplicateRelations()` — one LLM call per pair. With 50+ groups and MaxConcurrent=5, this requires 10+ serial rounds. Batching N groups into one call collapses this to a single concurrent round.

## Architecture

### New Config Field

`FastGraphExtractor` gains a `DedupBatchSize int` field (default 10, sourced from `utils.DefaultDedupBatchSize`). When >1, `ExtractGraph()` uses `DeduplicateRelationsBatch()`. When ≤1, it falls back to the existing per-group path.

### Data Flow

```
all pair groups with >1 relation (e.g. 50)
  → split into sub-batches of DedupBatchSize (e.g. 10)
  → 5 sub-batches dispatched concurrently (MaxConcurrent semaphore)
  → each sub-batch: 1 LLM call → returns clusters for all groups in the batch
  → results merged into relationAssets
```

### New Model Types (`models/graph.go`)

```go
type PairGroupResult struct {
    GroupId  int                      `json:"group_id"`
    Clusters map[string]RelationGroup `json:"clusters"`
}

type BatchRelationClusters struct {
    Groups []PairGroupResult `json:"groups"`
}
```

`RelationGroup` (already exists) is reused as-is. Indices in each group's clusters are **local** (0-based within that group's relation slice), consistent with current single-group behavior.

### Prompt Format (`prompts/prompts.go`)

Two new constants:

`DeduplicateBatchRelationsSystem` — extends the existing system prompt to explain multi-group input/output format, with `group_id` mapping.

`DeduplicateBatchRelationsPrompt` — template receiving the formatted multi-group CSV block:

```
GROUP 0 (alice|person → bob|person):
index,source_name,source_type,target_name,target_type,desc
0,alice,person,bob,person,works with
1,alice,person,bob,person,collaborated on project

GROUP 1 (company_a|organization → company_b|organization):
index,source_name,source_type,target_name,target_type,desc
0,company_a,organization,company_b,organization,partner of
1,company_a,organization,company_b,organization,strategic alliance with
```

LLM returns `BatchRelationClusters.Groups[i].GroupId` to map back to the correct input group.

### New Method (`graph/extractor.go`)

```go
func (f *FastGraphExtractor) DeduplicateRelationsBatch(
    ctx context.Context,
    groups [][]RelationAsset,
) ([][]RelationAsset, error)
```

- Builds the multi-group CSV prompt
- Calls LLM once with `GetBatchRelationClusterTool()`
- Parses `BatchRelationClusters`, maps each `PairGroupResult` back by `GroupId`
- Returns `[][]RelationAsset` aligned to input `groups` slice order

### `ExtractGraph()` Changes

Replace the per-group goroutine loop for dedup with:

1. Collect all groups with `len(relGroup) > 1` into a slice
2. Split into sub-batches of `DedupBatchSize`
3. For each sub-batch: launch goroutine (existing `dedupSem`), call `DeduplicateRelationsBatch()`
4. Retry logic (`MaxRetryAttempt`) wraps each sub-batch call
5. Entity validation (source/target in `entityKeyToId`) runs per-relation after each sub-batch resolves

Groups with exactly 1 relation continue to pass through without any LLM call (no change).

## Files Changed

| File | Change |
|------|--------|
| `utils/const.go` | Add `DefaultDedupBatchSize = 10` |
| `models/graph.go` | Add `PairGroupResult`, `BatchRelationClusters`, `GetBatchRelationClusterTool()` |
| `prompts/prompts.go` | Add `DeduplicateBatchRelationsSystem`, `DeduplicateBatchRelationsPrompt` |
| `graph/extractor.go` | Add `DedupBatchSize` field to struct + `NewFastGraphExtractor()`, add `DeduplicateRelationsBatch()`, update `ExtractGraph()` dedup loop |

## Error Handling

- If LLM returns a `group_id` not in the current batch: return error (invalid response)
- If entity validation fails for any relation in a sub-batch: retry the whole sub-batch (consistent with current per-group retry)
- Missing `group_id` in LLM response (partial result): return error, trigger retry

## Testing

- Unit test `DeduplicateRelationsBatch` with mock LLM returning well-formed `BatchRelationClusters`
- Test that N groups split into correct number of sub-batches
- Test partial last batch (e.g. 13 groups at batchSize=10 → 2 calls: 10 + 3)
- Test error propagation when LLM returns invalid group_id
- Integration: existing `DeduplicateRelations` tests unchanged (backward compat)

## Backward Compatibility

- `DeduplicateRelations()` kept unchanged
- `DedupBatchSize <= 1` falls back to per-group path
- No changes to external API or option functions needed (internal extractor field)
