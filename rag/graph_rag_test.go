package rag

import (
	"context"
	"errors"
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

// TestBatchEmbed_EmbedBatchError verifies that an EmbedBatch error is propagated correctly.
func TestBatchEmbed_EmbedBatchError(t *testing.T) {
	entities := makeTestEntities(3)
	expectedErr := errors.New("embed API unavailable")
	emb := &mockEmbedderWithError{model: "test-model", dim: 3, err: expectedErr}

	vecs, err := batchEmbed(context.Background(), entities, emb, 50, 1)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if err != expectedErr {
		t.Errorf("expected %v, got %v", expectedErr, err)
	}
	if vecs != nil {
		t.Errorf("expected nil slice on error, got %v", vecs)
	}
}
