package fastgraphrag

import (
	"cmp"
	"context"

	"log/slog"
	"slices"
	"time"

	"github.com/manageai-inet/fast-graphrago/llm"
	"github.com/manageai-inet/fast-graphrago/rag"
	am "github.com/manageai-inet/agentic-assets"
)

// FastGraphIndexer represents the initialization block for the RAG package.
type FastGraphIndexer struct {
	Rag    rag.GraphRAGService
	Assets am.AssetStorage
	Vectors am.VectorStorage
	am.LoggingCapacity
}

func New(llm llm.LLM, vecStore am.VectorStorage, assetStore am.AssetStorage, opts ...rag.Option) (*FastGraphIndexer, error) {
	graphRag := rag.NewGraphRAGService(llm, vecStore, assetStore, opts...)
	return &FastGraphIndexer{
		Rag:             graphRag,
		Assets:          assetStore,
		Vectors:         vecStore,
		LoggingCapacity: *am.GetDefaultLoggingCapacity(),
	}, nil
}

func (f *FastGraphIndexer) SetLogger(logger *slog.Logger) {
	f.LoggingCapacity.SetLogger(logger)
	f.Rag.SetLogger(logger)
}

func (f *FastGraphIndexer) Index(ctx context.Context, kbId string, sources []am.KnowledgeSource, labels *[]string, metadata *map[string]any, config *map[string]any) (am.IndexingResult, error) {
	logger := am.GetLogger(f)
	logger.InfoContext(ctx, "starting to index", slog.String("kbId", kbId), slog.Any("sources", sources), slog.Any("labels", labels), slog.Any("metadata", metadata), slog.Any("config", config))
	// assets and vectors already inserted to vector storage and asset storage
	pageSources := f.FilterSupportedSources(sources)
	// sort page by source name, source name will be `kbId:filename.ext:page_number`
	slices.SortFunc(pageSources, func(a, b am.KnowledgeSource) int {
		return cmp.Compare(a.SourceName, b.SourceName)
	})
	assets, vectors, err := f.Rag.Index(ctx, kbId, pageSources, labels, metadata, config)
	if err != nil {
		err_msg := err.Error()
		logger.ErrorContext(ctx, "failed to index: "+err_msg, slog.String("kbId", kbId))
		return am.IndexingResult{
			KbId:      kbId,
			Status:    false,
			IndexedAt: time.Now(),
			Error:     &err_msg,
		}, err
	}
	if len(assets) == 0 {
		err_msg := "No assets were indexed, might be something wrong."
		logger.ErrorContext(ctx, "failed to index: "+err_msg, slog.String("kbId", kbId))
		return am.IndexingResult{
			KbId:      kbId,
			Status:    false,
			IndexedAt: time.Now(),
			Error:     &err_msg,
		}, nil
	}
	version := assets[0].Version
	asset_count := map[string]int{}
	for _, asset := range assets {
		asset_count[asset.AssetType]++
	}
	var embedModel *string
	var embedDim int
	if len(vectors) >= 0 {
		// All vector should be embedded by the same embedding model
		asset_count[am.AssetTypeVector] = len(vectors)
		embedModel = vectors[0].EmbeddingModel
		embedDim = len(vectors[0].EmbededVector)
	}
	logger.InfoContext(ctx, "successfully indexed", slog.String("kbId", kbId), slog.Int("version", version), slog.Any("assets", asset_count))
	return am.IndexingResult{
		KbId:           kbId,
		Status:         true,
		IndexedAt:      time.Now(),
		Version:        &version,
		EmbeddingModel: embedModel,
		EmbeddingDim:   &embedDim,
		AssetsCount:    &asset_count,
	}, nil
}

// config accept: top_k int, threshold float32, version *int, label *string
func (f *FastGraphIndexer) Retrieve(ctx context.Context, query string, kbIds []string, seedAssets []am.ContextualAsset, config *map[string]any) (am.RetrieveResult, error) {
	logger := am.GetLogger(f)
	logger.InfoContext(ctx, "starting to retrieve", slog.Any("kbIds", kbIds), slog.Any("config", config))
	retrievedAssets, propagatedAssets, err := f.Rag.Retrieve(ctx, query, kbIds, seedAssets, config)
	if err != nil {
		err_msg := err.Error()
		logger.ErrorContext(ctx, "failed to retrieve: "+err_msg, slog.Any("kbIds", kbIds))
		return am.RetrieveResult{
			KbIds:      kbIds,
			Query:      query,
			Config:     config,
			Status:     false,
			RetrieveAt: time.Now(),
			Error:      &err_msg,
		}, err
	}
	for _, asset := range propagatedAssets {
		retrievedAssets = append(retrievedAssets, am.RetrievedAsset{
			ContextualAsset: asset,
			Score:           nil,
		})
	}
	logger.InfoContext(ctx, "successfully retrieved", slog.Any("kbIds", kbIds), slog.Int("retrievedAssets", len(retrievedAssets)))
	return am.RetrieveResult{
		KbIds:           kbIds,
		Query:           query,
		Config:          config,
		Status:          true,
		RetrieveAt:      time.Now(),
		RetrievedAssets: &retrievedAssets,
	}, nil
}

func (f *FastGraphIndexer) getSupportedAssetTypes() []string {
	return []string{am.AssetTypePage}
}

func (f *FastGraphIndexer) FilterSupportedSources(sources []am.KnowledgeSource) []am.KnowledgeSource {
	filteredSources := []am.KnowledgeSource{}
	supportedAssetTypes := f.getSupportedAssetTypes()
	for _, source := range sources {
		if slices.Contains(supportedAssetTypes, source.SourceType) {
			filteredSources = append(filteredSources, source)
		}
	}
	return filteredSources
}

func (f *FastGraphIndexer) getRetrivalSeedAssetTypes() []string {
	return []string{am.AssetTypeDocument}
}

func (f *FastGraphIndexer) GetRetrievalSeedAssets(ctx context.Context, query string, kbIds []string, version *int, label *string) []am.ContextualAsset {
	seedTypes := f.getRetrivalSeedAssetTypes()
	seedAssets := []am.ContextualAsset{}
	for _, kbId := range kbIds {
		for _, seedType := range seedTypes {
			assets, err := f.Assets.GetAssets(ctx, kbId, seedType, nil, version, label)
			if err == nil {
				seedAssets = append(seedAssets, assets...)
			}
		}
	}
	return seedAssets
}
