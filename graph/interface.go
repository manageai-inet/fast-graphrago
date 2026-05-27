package graph

import (
	"context"
	"github.com/manageai-inet/fast-graphrago/llm"
	"github.com/manageai-inet/fast-graphrago/models"
	"github.com/manageai-inet/fast-graphrago/utils"

	asset_manager "github.com/manageai-inet/agentic-assets"
)

type GraphExtractionOptions struct {
	Domain string
	EntityTypes []string
}

func NewGraphExtractionOptions() GraphExtractionOptions {
	return GraphExtractionOptions{
		Domain: utils.DefaultDomain,
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