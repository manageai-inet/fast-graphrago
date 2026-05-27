package graph

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/manageai-inet/fast-graphrago/models"
	"github.com/manageai-inet/fast-graphrago/utils"

	"github.com/openai/openai-go"

	asset_manager "github.com/manageai-inet/agentic-assets"
)

// mockLLM replays pre-set responses in order.
type mockLLM struct {
	responses   []openai.ChatCompletionMessage
	callCount   int
	errToReturn error
}

func (m *mockLLM) Generate(
	_ context.Context,
	_ []openai.ChatCompletionMessage,
	_ []openai.ChatCompletionToolParam,
	_ *openai.ChatCompletionToolChoiceOptionUnionParam,
) (*openai.ChatCompletionMessage, error) {
	if m.errToReturn != nil {
		return nil, m.errToReturn
	}
	if m.callCount >= len(m.responses) {
		return nil, fmt.Errorf("mockLLM: unexpected call %d (only %d responses configured)", m.callCount, len(m.responses))
	}
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
			{Name: "Acme", Type: "organization", Desc: "a company", ChunkIndex: 0},
			{Name: "Bob", Type: "person", Desc: "a founder", ChunkIndex: 1},
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
	relationChunkIds := map[string]string{}
	for _, ra := range result.RelationAssets {
		if len(ra.ChunkIds) != 1 {
			t.Errorf("relation %s→%s: expected 1 ChunkId, got %d", ra.Source, ra.Target, len(ra.ChunkIds))
		} else {
			key := ra.Source + "→" + ra.Target
			relationChunkIds[key] = ra.ChunkIds[0]
		}
	}
	if relationChunkIds["alice→acme"] != "kb:file.txt:chunk-0" {
		t.Errorf("alice→acme ChunkId = %q, want kb:file.txt:chunk-0", relationChunkIds["alice→acme"])
	}
	if relationChunkIds["bob→globex"] != "kb:file.txt:chunk-1" {
		t.Errorf("bob→globex ChunkId = %q, want kb:file.txt:chunk-1", relationChunkIds["bob→globex"])
	}

	// Check ChunkIds for organization entities
	if entityChunkIds["acme"] != "kb:file.txt:chunk-0" {
		t.Errorf("acme ChunkId = %q, want kb:file.txt:chunk-0", entityChunkIds["acme"])
	}
	if entityChunkIds["globex"] != "kb:file.txt:chunk-1" {
		t.Errorf("globex ChunkId = %q, want kb:file.txt:chunk-1", entityChunkIds["globex"])
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

// TestExtractGraphFromChunks_NegativeChunkIndex verifies a negative chunk_index causes an error.
func TestExtractGraphFromChunks_NegativeChunkIndex(t *testing.T) {
	chunks := []asset_manager.ContextualAsset{
		{AssetId: "kb:file.txt:chunk-0", Content: "Alice works at Acme."},
	}
	llmResp := toolCallMsg(models.ExtractedGraph{
		Entities: []models.GraphEntity{
			{Name: "Alice", Type: "person", Desc: "an employee", ChunkIndex: -1},
		},
		Relations: []models.GraphRelation{},
	})

	e := newTestExtractor([]openai.ChatCompletionMessage{llmResp})
	opts := GraphExtractionOptions{Domain: "test", EntityTypes: []string{"person"}}

	_, err := e.ExtractGraphFromChunks(context.Background(), chunks, opts)
	if err == nil {
		t.Fatal("expected error for negative chunk_index, got nil")
	}
}

// TestExtractGraphFromChunks_LLMError verifies that LLM errors propagate correctly.
func TestExtractGraphFromChunks_LLMError(t *testing.T) {
	chunks := []asset_manager.ContextualAsset{
		{AssetId: "kb:file.txt:chunk-0", Content: "Alice works at Acme."},
	}

	e := NewFastGraphExtractor(&mockLLM{errToReturn: errors.New("rate limit exceeded")})
	opts := GraphExtractionOptions{Domain: "test", EntityTypes: []string{"person"}}

	_, err := e.ExtractGraphFromChunks(context.Background(), chunks, opts)
	if err == nil {
		t.Fatal("expected error from LLM, got nil")
	}
	if err.Error() != "rate limit exceeded" {
		t.Errorf("error = %q, want %q", err.Error(), "rate limit exceeded")
	}
}
