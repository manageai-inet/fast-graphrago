package rag

import (
	"context"
	asset_manager "github.com/manageai-inet/agentic-assets"
)

// GraphRAGService is the main port (interface) to interact with the system.
type GraphRAGService interface {
	Index(ctx context.Context, kbId string, sources []asset_manager.KnowledgeSource, labels *[]string, metadata *map[string]any, config *map[string]any) ([]asset_manager.ContextualAsset, []asset_manager.VectorAsset, error)
	// return retrieved assets and propagated assets
	Retrieve(ctx context.Context, query string, kbIds []string, seedAssets []asset_manager.ContextualAsset, config *map[string]any) ([]asset_manager.RetrievedAsset, []asset_manager.ContextualAsset, error)
	asset_manager.Loggable
}