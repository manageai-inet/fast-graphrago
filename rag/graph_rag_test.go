package rag

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	asset_manager "github.com/manageai-inet/agentic-assets"
	chunk "github.com/manageai-inet/agentic-assets/chunk"
	"github.com/manageai-inet/fast-graphrago/models"
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

// mockEmbedderWithError always returns an error from EmbedBatch.
type mockEmbedderWithError struct {
	model string
	dim   int
	err   error
}

func (m *mockEmbedderWithError) GetEmbeddingModel() string { return m.model }
func (m *mockEmbedderWithError) GetEmbeddingDim() int      { return m.dim }
func (m *mockEmbedderWithError) Embed(ctx context.Context, content string) ([]float32, error) {
	return nil, m.err
}
func (m *mockEmbedderWithError) EmbedBatch(ctx context.Context, contents []string) ([][]float32, error) {
	return nil, m.err
}

type flakyBatchEmbedder struct {
	mu                sync.Mutex
	model             string
	dim               int
	batchFailuresLeft int
	batchCalls        int
	singleCalls       int
}

func (m *flakyBatchEmbedder) GetEmbeddingModel() string { return m.model }
func (m *flakyBatchEmbedder) GetEmbeddingDim() int      { return m.dim }
func (m *flakyBatchEmbedder) Embed(ctx context.Context, content string) ([]float32, error) {
	m.mu.Lock()
	m.singleCalls++
	m.mu.Unlock()
	return make([]float32, m.dim), nil
}
func (m *flakyBatchEmbedder) EmbedBatch(ctx context.Context, contents []string) ([][]float32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.batchCalls++
	if m.batchFailuresLeft > 0 {
		m.batchFailuresLeft--
		return nil, errors.New("temporary embed batch failure")
	}
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

	vecs, err := batchEmbed(context.Background(), entities, emb, 2)
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

	vecs, err := batchEmbed(context.Background(), entities, emb, 50)
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

	vecs, err := batchEmbed(context.Background(), entities, emb, 3)
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

// TestBatchEmbed_EmbedBatchError verifies that an EmbedBatch error is propagated correctly.
func TestBatchEmbed_EmbedBatchError(t *testing.T) {
	entities := makeTestEntities(3)
	expectedErr := errors.New("embed API unavailable")
	emb := &mockEmbedderWithError{model: "test-model", dim: 3, err: expectedErr}

	// explicit attempts=1/delay=0 keeps this test fast; retry timing itself is covered by
	// TestBatchEmbed_RetriesBatchThenSucceeds and TestBatchEmbed_FallsBackToSingleItemEmbedding.
	vecs, err := batchEmbedWithRetry(context.Background(), entities, emb, 50, nil, 1, 0)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if !strings.Contains(err.Error(), expectedErr.Error()) {
		t.Errorf("expected error containing %q, got %v", expectedErr.Error(), err)
	}
	if vecs != nil {
		t.Errorf("expected nil slice on error, got %v", vecs)
	}
}

func TestBatchEmbed_RetriesBatchThenSucceeds(t *testing.T) {
	entities := makeTestEntities(3)
	emb := &flakyBatchEmbedder{model: "test-model", dim: 4, batchFailuresLeft: 2}

	vecs, err := batchEmbedWithRetry(context.Background(), entities, emb, 50, nil, 3, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vecs) != 3 {
		t.Fatalf("expected 3 VectorAssets, got %d", len(vecs))
	}
	if emb.batchCalls != 3 {
		t.Fatalf("expected 3 batch calls, got %d", emb.batchCalls)
	}
	if emb.singleCalls != 0 {
		t.Fatalf("expected no single-item fallback, got %d single calls", emb.singleCalls)
	}
}

func TestBatchEmbed_FallsBackToSingleItemEmbedding(t *testing.T) {
	entities := makeTestEntities(2)
	emb := &flakyBatchEmbedder{model: "test-model", dim: 2, batchFailuresLeft: 3}

	vecs, err := batchEmbedWithRetry(context.Background(), entities, emb, 50, nil, 3, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("expected 2 VectorAssets, got %d", len(vecs))
	}
	if emb.batchCalls != 3 {
		t.Fatalf("expected 3 batch calls before fallback, got %d", emb.batchCalls)
	}
	if emb.singleCalls != 2 {
		t.Fatalf("expected 2 single-item fallback calls, got %d", emb.singleCalls)
	}
}

func TestPreparePageAssetsPreservesSourceOrder(t *testing.T) {
	page1 := "first page"
	page2 := "second page"
	sourceMetadata := map[string]any{"page": 1}
	sources := []asset_manager.KnowledgeSource{
		{
			SourceName:     "kb:file.pdf:1",
			SourceContents: &page1,
			Metadata:       &sourceMetadata,
		},
		{
			SourceName:     "kb:file.pdf:2",
			SourceContents: &page2,
			Metadata:       &sourceMetadata,
		},
	}
	seedAsset := asset_manager.ContextualAsset{
		KbId:      "kb",
		AssetType: asset_manager.AssetTypeDocument,
		AssetId:   "kb:file.pdf",
		Refs:      &[]asset_manager.AssetRef{},
	}

	pageAssets, err := preparePageAssets("svc", "kb", "file.pdf", 123, sources, nil, &seedAsset)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(pageAssets) != 2 {
		t.Fatalf("expected 2 page assets, got %d", len(pageAssets))
	}
	if pageAssets[0].AssetId != "kb:file.pdf:0" {
		t.Fatalf("first page asset id = %q, want %q", pageAssets[0].AssetId, "kb:file.pdf:0")
	}
	if pageAssets[1].AssetId != "kb:file.pdf:1" {
		t.Fatalf("second page asset id = %q, want %q", pageAssets[1].AssetId, "kb:file.pdf:1")
	}
	if pageAssets[0].Content != page1 || pageAssets[1].Content != page2 {
		t.Fatalf("page asset contents not preserved in order: got [%q, %q]", pageAssets[0].Content, pageAssets[1].Content)
	}
	if seedAsset.Refs == nil || len(*seedAsset.Refs) != 2 {
		t.Fatalf("seed asset refs = %v, want 2 child refs", seedAsset.Refs)
	}
	if (*seedAsset.Refs)[0].AssetId != pageAssets[0].AssetId || (*seedAsset.Refs)[1].AssetId != pageAssets[1].AssetId {
		t.Fatalf("seed refs not appended in source order: got %+v", *seedAsset.Refs)
	}
}

func TestPrepareChunkAssetsAttachToMatchingPage(t *testing.T) {
	page1 := "first page"
	page2 := "second page"
	sourceMetadata := map[string]any{"page": 1}
	sources := []asset_manager.KnowledgeSource{
		{
			SourceName:     "kb:file.pdf:1",
			SourceContents: &page1,
			Metadata:       &sourceMetadata,
		},
		{
			SourceName:     "kb:file.pdf:2",
			SourceContents: &page2,
			Metadata:       &sourceMetadata,
		},
	}
	seedAsset := asset_manager.ContextualAsset{
		KbId:      "kb",
		AssetType: asset_manager.AssetTypeDocument,
		AssetId:   "kb:file.pdf",
		Refs:      &[]asset_manager.AssetRef{},
	}
	pageAssets, err := preparePageAssets("svc", "kb", "file.pdf", 123, sources, nil, &seedAsset)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	chunks := []chunk.ChunkImmediateAsset{
		{PageNumber: 1, StartPos: 0, Content: "chunk from page two"},
	}

	chunkAssets, err := prepareChunkAssets("svc", "kb", 123, chunks, pageAssets)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunkAssets) != 1 {
		t.Fatalf("expected 1 chunk asset, got %d", len(chunkAssets))
	}
	if chunkAssets[0].AssetId != "kb:file.pdf:1[0:19]" {
		t.Fatalf("chunk asset id = %q, want %q", chunkAssets[0].AssetId, "kb:file.pdf:1[0:19]")
	}
	if chunkAssets[0].Refs == nil || len(*chunkAssets[0].Refs) != 1 {
		t.Fatalf("chunk refs = %v, want 1 parent ref", chunkAssets[0].Refs)
	}
	if (*chunkAssets[0].Refs)[0].AssetId != pageAssets[1].AssetId {
		t.Fatalf("chunk parent asset id = %q, want %q", (*chunkAssets[0].Refs)[0].AssetId, pageAssets[1].AssetId)
	}
	if pageAssets[1].Refs == nil || len(*pageAssets[1].Refs) != 2 {
		t.Fatalf("page 2 refs = %v, want document parent + chunk child", pageAssets[1].Refs)
	}
}

type mockQueryVectorStore struct {
	resultsByAsset map[string][]asset_manager.RetrievedVector
}

func (m *mockQueryVectorStore) SetEmbedder(ctx context.Context, embedder asset_manager.Embedder) error {
	return nil
}

func (m *mockQueryVectorStore) Setup(ctx context.Context) error { return nil }

func (m *mockQueryVectorStore) EmbedContent(ctx context.Context, content string) ([]float32, error) {
	return nil, nil
}

func (m *mockQueryVectorStore) EmbedAsset(ctx context.Context, asset *asset_manager.ContextualAsset, contentConstructorFn *func(asset_manager.ContextualAsset) string) (asset_manager.VectorAsset, error) {
	return asset_manager.VectorAsset{}, nil
}

func (m *mockQueryVectorStore) QueryVectors(ctx context.Context, queryVector []float32, topK *int, threshold *float32, filter *asset_manager.VectorQueryFilter) ([]asset_manager.RetrievedVector, error) {
	return m.resultsByAsset[fmt.Sprint(queryVector)], nil
}

func (m *mockQueryVectorStore) GetVersions(ctx context.Context, kbId string) ([]int, error) {
	return nil, nil
}

func (m *mockQueryVectorStore) InsertVector(ctx context.Context, vectorAsset asset_manager.VectorAsset) (bool, error) {
	return true, nil
}

func (m *mockQueryVectorStore) InsertBatchVectors(ctx context.Context, vectorAssets []asset_manager.VectorAsset) (int, error) {
	return len(vectorAssets), nil
}

func (m *mockQueryVectorStore) DeleteVector(ctx context.Context, kbId string, assetId string, version *int) (bool, error) {
	return true, nil
}

func (m *mockQueryVectorStore) DeleteVectorsByKbId(ctx context.Context, kbId string) (int, error) {
	return 0, nil
}

func (m *mockQueryVectorStore) DeleteVectorsByKbIdAndVersion(ctx context.Context, kbId string, version *int) (int, error) {
	return 0, nil
}

// flakyVectorStore embeds mockQueryVectorStore and overrides EmbedAsset to fail a
// fixed number of times before succeeding, simulating a transient/malformed embedder response.
type flakyVectorStore struct {
	mockQueryVectorStore
	mu           sync.Mutex
	failuresLeft int
	calls        int
}

func (m *flakyVectorStore) EmbedAsset(ctx context.Context, asset *asset_manager.ContextualAsset, contentConstructorFn *func(asset_manager.ContextualAsset) string) (asset_manager.VectorAsset, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.calls++
	if m.failuresLeft > 0 {
		m.failuresLeft--
		return asset_manager.VectorAsset{}, errors.New("invalid value in embedding array at index 0, expected float64, got <nil>")
	}
	return asset_manager.VectorAsset{AssetId: asset.AssetId, EmbededVector: []float32{1, 2, 3}}, nil
}

func TestEmbedAssetWithRetry_RetriesThenSucceeds(t *testing.T) {
	store := &flakyVectorStore{failuresLeft: 2}
	entity := &asset_manager.ContextualAsset{AssetId: "entity-0"}

	v, err := embedAssetWithRetry(context.Background(), store, entity, nil, 3, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if store.calls != 3 {
		t.Fatalf("expected 3 calls, got %d", store.calls)
	}
	if v.AssetId != "entity-0" {
		t.Fatalf("AssetId: got %q, want %q", v.AssetId, "entity-0")
	}
}

func TestEmbedAssetWithRetry_FailsAfterExhaustingRetries(t *testing.T) {
	store := &flakyVectorStore{failuresLeft: 5}
	entity := &asset_manager.ContextualAsset{AssetId: "entity-0"}

	_, err := embedAssetWithRetry(context.Background(), store, entity, nil, 3, 0)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if store.calls != 3 {
		t.Fatalf("expected 3 calls, got %d", store.calls)
	}
}

// partialFailureVectorStore embeds mockQueryVectorStore; EmbedAsset always fails for asset IDs
// in failIDs (simulating a deterministic malformed embedder response) and succeeds for the rest.
type partialFailureVectorStore struct {
	mockQueryVectorStore
	mu      sync.Mutex
	failIDs map[string]bool
	calls   map[string]int
}

func (m *partialFailureVectorStore) EmbedAsset(ctx context.Context, asset *asset_manager.ContextualAsset, contentConstructorFn *func(asset_manager.ContextualAsset) string) (asset_manager.VectorAsset, error) {
	m.mu.Lock()
	if m.calls == nil {
		m.calls = map[string]int{}
	}
	m.calls[asset.AssetId]++
	m.mu.Unlock()
	if m.failIDs[asset.AssetId] {
		return asset_manager.VectorAsset{}, errors.New("invalid value in embedding array at index 0, expected float64, got <nil>")
	}
	return asset_manager.VectorAsset{AssetId: asset.AssetId, EmbededVector: []float32{1, 2, 3}}, nil
}

func TestEmbedEntitiesIndividually_SkipsPermanentlyFailingEntity(t *testing.T) {
	entities := makeTestEntities(3) // entity-0, entity-1, entity-2
	store := &partialFailureVectorStore{failIDs: map[string]bool{"entity-1": true}}

	vectors := embedEntitiesIndividually(context.Background(), entities, store, 2, nil, 2, 0)

	if len(vectors) != 2 {
		t.Fatalf("expected 2 vectors (one entity skipped), got %d", len(vectors))
	}
	for _, v := range vectors {
		if v.AssetId == "entity-1" {
			t.Fatalf("expected entity-1 to be skipped, but it was embedded")
		}
	}
	if store.calls["entity-1"] != 2 {
		t.Fatalf("expected entity-1 to be retried 2 times before being skipped, got %d", store.calls["entity-1"])
	}
}

func TestDedupeContextualAssetsPreservesOrder(t *testing.T) {
	input := []asset_manager.ContextualAsset{
		{AssetId: "a"},
		{AssetId: "b"},
		{AssetId: "a"},
		{AssetId: "c"},
	}

	deduped, byID, duplicates := dedupeContextualAssetsByID(input)

	if len(deduped) != 3 {
		t.Fatalf("expected 3 assets, got %d", len(deduped))
	}
	if deduped[0].AssetId != "a" || deduped[1].AssetId != "b" || deduped[2].AssetId != "c" {
		t.Fatalf("unexpected order after dedupe: %+v", deduped)
	}
	if len(byID) != 3 {
		t.Fatalf("expected 3 map entries, got %d", len(byID))
	}
	if len(duplicates) != 1 || duplicates[0] != "a" {
		t.Fatalf("unexpected duplicates: %+v", duplicates)
	}
}

func TestGetSeedVectorAccumulatesWithoutLookupSlices(t *testing.T) {
	scoreA := float32(0.9)
	scoreB := float32(0.6)
	scoreC := float32(0.5)
	svc := &GraphRAGServiceImpl{
		VectorStore: &mockQueryVectorStore{
			resultsByAsset: map[string][]asset_manager.RetrievedVector{
				"[1]": {
					{VectorAsset: asset_manager.VectorAsset{AssetId: "entity-a"}, Score: &scoreA},
				},
				"[2]": {
					{VectorAsset: asset_manager.VectorAsset{AssetId: "entity-b"}, Score: &scoreB},
				},
				"[3]": {
					{VectorAsset: asset_manager.VectorAsset{AssetId: "entity-c"}, Score: &scoreC},
				},
			},
		},
		GraphRAGServiceCompileTimeOptions: GraphRAGServiceCompileTimeOptions{MaxConcurrent: 1},
	}

	queryEntities := []models.QueryEntity{
		{Name: "query", Type: models.QueryEntityType, Embedding: []float32{1}},
		{Name: "named", Type: models.NamedEntityType, Embedding: []float32{2}},
		{Name: "generic", Type: models.GenericEntityType, Embedding: []float32{3}},
	}
	matrix := models.TransitionMatrix{
		Dim:              3,
		EntityIndiceToId: []string{"entity-a", "entity-b", "entity-c"},
		EntityIDToIndex: map[string]int{
			"entity-a": 0,
			"entity-b": 1,
			"entity-c": 2,
		},
	}

	weights := [3]float32{0.5, 0.2, 0.3}
	seed, err := svc.getSeedVector(context.Background(), "kb", queryEntities, matrix, 5, 0, weights, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	wantRaw := []float32{scoreA * weights[2], scoreB * weights[0], scoreC * weights[1]}
	sum := wantRaw[0] + wantRaw[1] + wantRaw[2]
	want := []float32{wantRaw[0] / sum, wantRaw[1] / sum, wantRaw[2] / sum}

	for i := range want {
		if diff := seed[i] - want[i]; diff > 0.0001 || diff < -0.0001 {
			t.Fatalf("seed[%d] = %f, want %f", i, seed[i], want[i])
		}
	}
}
