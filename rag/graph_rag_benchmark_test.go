package rag

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"testing"

	"github.com/manageai-inet/fast-graphrago/graph"
	"github.com/manageai-inet/fast-graphrago/llm"
	"github.com/manageai-inet/fast-graphrago/models"

	asset_manager "github.com/manageai-inet/agentic-assets"
	chunk "github.com/manageai-inet/agentic-assets/chunk"
)

type benchmarkChunkExtractor struct {
	chunksPerPage int
	chunkSize     int
}

func (e *benchmarkChunkExtractor) Extract(ctx context.Context, kbId string, sources []asset_manager.KnowledgeSource) ([]chunk.ChunkImmediateAsset, error) {
	chunks := make([]chunk.ChunkImmediateAsset, 0, len(sources)*e.chunksPerPage)
	for pageIdx, source := range sources {
		content := ""
		if source.SourceContents != nil {
			content = *source.SourceContents
		}
		for i := 0; i < e.chunksPerPage; i++ {
			start := i * e.chunkSize
			end := start + e.chunkSize
			if start >= len(content) {
				start = len(content)
			}
			if end > len(content) {
				end = len(content)
			}
			chunks = append(chunks, chunk.ChunkImmediateAsset{
				KbId:       kbId,
				AssetId:    fmt.Sprintf("%s:%d:%d", kbId, pageIdx, i),
				Content:    content[start:end],
				PageNumber: pageIdx,
				StartPos:   start,
			})
		}
	}
	return chunks, nil
}

func (e *benchmarkChunkExtractor) Compare(ctx context.Context, oldChunks, newChunks []chunk.ChunkImmediateAsset) ([]chunk.ChunkImmediateAsset, error) {
	return newChunks, nil
}

type benchmarkGraphExtractor struct {
	asset_manager.LoggingCapacity
}

func (e *benchmarkGraphExtractor) SetLLM(model llm.LLM) {}

func (e *benchmarkGraphExtractor) SetBatchSize(batchSize int) {}

func (e *benchmarkGraphExtractor) SetLogger(logger *slog.Logger) {
	e.LoggingCapacity.SetLogger(logger)
}

func (e *benchmarkGraphExtractor) GetLogger() *slog.Logger {
	return e.LoggingCapacity.GetLogger()
}

func (e *benchmarkGraphExtractor) ExtractGraph(ctx context.Context, chunks []asset_manager.ContextualAsset, options graph.GraphExtractionOptions) (models.GraphAssets, error) {
	entityAssets := make([]models.EntityAsset, 0, len(chunks)*2)
	relationAssets := make([]models.RelationAsset, 0, len(chunks))
	for i, c := range chunks {
		personName := fmt.Sprintf("person-%d", i)
		orgName := fmt.Sprintf("org-%d", i)
		entityAssets = append(entityAssets,
			models.EntityAsset{
				Name:        personName,
				Type:        "Person",
				Description: "benchmark entity person",
				ChunkIds:    []string{c.AssetId},
			},
			models.EntityAsset{
				Name:        orgName,
				Type:        "Organization",
				Description: "benchmark entity organization",
				ChunkIds:    []string{c.AssetId},
			},
		)
		relationAssets = append(relationAssets, models.RelationAsset{
			Source:      personName,
			SourceType:  "Person",
			Target:      orgName,
			TargetType:  "Organization",
			Description: "works_at",
			ChunkIds:    []string{c.AssetId},
		})
	}
	return models.GraphAssets{EntityAssets: entityAssets, RelationAssets: relationAssets}, nil
}

func (e *benchmarkGraphExtractor) ExtractGraphFromChunk(ctx context.Context, chunk asset_manager.ContextualAsset, options graph.GraphExtractionOptions) (models.GraphAssets, error) {
	return e.ExtractGraph(ctx, []asset_manager.ContextualAsset{chunk}, options)
}

func (e *benchmarkGraphExtractor) ExtractEntitiesFromQuery(ctx context.Context, query string) (models.ExtractedQuery, error) {
	return models.ExtractedQuery{}, nil
}

type benchmarkRetrieveGraphExtractor struct {
	benchmarkGraphExtractor
}

func (e *benchmarkRetrieveGraphExtractor) ExtractEntitiesFromQuery(ctx context.Context, query string) (models.ExtractedQuery, error) {
	return models.ExtractedQuery{
		NamedEntities:   []string{"named"},
		GenericEntities: []string{"generic"},
	}, nil
}

type benchmarkEmbedder struct {
	dim int
}

func (e *benchmarkEmbedder) GetEmbeddingModel() string { return "benchmark-embedder" }

func (e *benchmarkEmbedder) GetEmbeddingDim() int { return e.dim }

func (e *benchmarkEmbedder) Embed(ctx context.Context, content string) ([]float32, error) {
	return make([]float32, e.dim), nil
}

func (e *benchmarkEmbedder) EmbedBatch(ctx context.Context, contents []string) ([][]float32, error) {
	vectors := make([][]float32, len(contents))
	for i := range contents {
		vectors[i] = make([]float32, e.dim)
	}
	return vectors, nil
}

type benchmarkVectorStore struct {
	inserted             []asset_manager.VectorAsset
	embedder             asset_manager.Embedder
	queryResultsByMarker map[float32][]asset_manager.RetrievedVector
}

func (s *benchmarkVectorStore) SetEmbedder(ctx context.Context, embedder asset_manager.Embedder) error {
	s.embedder = embedder
	return nil
}

func (s *benchmarkVectorStore) Setup(ctx context.Context) error { return nil }

func (s *benchmarkVectorStore) EmbedContent(ctx context.Context, content string) ([]float32, error) {
	switch content {
	case "benchmark-query":
		return []float32{1}, nil
	case "named":
		return []float32{2}, nil
	case "generic":
		return []float32{3}, nil
	}
	if s.embedder == nil {
		return []float32{}, nil
	}
	return s.embedder.Embed(ctx, content)
}

func (s *benchmarkVectorStore) EmbedAsset(ctx context.Context, asset *asset_manager.ContextualAsset, contentConstructorFn *func(asset_manager.ContextualAsset) string) (asset_manager.VectorAsset, error) {
	content := asset.Content
	if contentConstructorFn != nil {
		content = (*contentConstructorFn)(*asset)
	}
	vector, err := s.EmbedContent(ctx, content)
	if err != nil {
		return asset_manager.VectorAsset{}, err
	}
	model := ""
	if s.embedder != nil {
		model = s.embedder.GetEmbeddingModel()
	}
	return asset_manager.VectorAsset{
		KbId:           asset.KbId,
		AssetId:        asset.AssetId,
		Version:        asset.Version,
		Content:        content,
		Refs:           asset.Refs,
		Labels:         asset.Labels,
		Metadata:       asset.Metadata,
		EmbeddingModel: &model,
		EmbededVector:  vector,
	}, nil
}

func (s *benchmarkVectorStore) QueryVectors(ctx context.Context, queryVector []float32, topK *int, threshold *float32, filter *asset_manager.VectorQueryFilter) ([]asset_manager.RetrievedVector, error) {
	if len(queryVector) > 0 && s.queryResultsByMarker != nil {
		if results, ok := s.queryResultsByMarker[queryVector[0]]; ok {
			return results, nil
		}
	}
	return nil, nil
}

func (s *benchmarkVectorStore) GetVersions(ctx context.Context, kbId string) ([]int, error) {
	return nil, nil
}

func (s *benchmarkVectorStore) InsertVector(ctx context.Context, vectorAsset asset_manager.VectorAsset) (bool, error) {
	s.inserted = append(s.inserted, vectorAsset)
	return true, nil
}

func (s *benchmarkVectorStore) InsertBatchVectors(ctx context.Context, vectorAssets []asset_manager.VectorAsset) (int, error) {
	s.inserted = append(s.inserted[:0], vectorAssets...)
	return len(vectorAssets), nil
}

func (s *benchmarkVectorStore) DeleteVector(ctx context.Context, kbId string, assetId string, version *int) (bool, error) {
	return true, nil
}

func (s *benchmarkVectorStore) DeleteVectorsByKbId(ctx context.Context, kbId string) (int, error) {
	return 0, nil
}

func (s *benchmarkVectorStore) DeleteVectorsByKbIdAndVersion(ctx context.Context, kbId string, version *int) (int, error) {
	return 0, nil
}

type benchmarkAssetStore struct {
	inserted       []asset_manager.ContextualAsset
	assetsByType   map[string][]asset_manager.ContextualAsset
	assetsByID     map[string]asset_manager.ContextualAsset
	relationAssets []asset_manager.ContextualAsset
}

func (s *benchmarkAssetStore) Setup(ctx context.Context) error { return nil }

func (s *benchmarkAssetStore) GetVersions(ctx context.Context, kbId string) ([]int, error) {
	return nil, nil
}

func (s *benchmarkAssetStore) GetAssets(ctx context.Context, kbId string, assetType string, assetIds *[]string, version *int, label *string) ([]asset_manager.ContextualAsset, error) {
	assets := s.assetsByType[assetType]
	if assetIds == nil {
		return append([]asset_manager.ContextualAsset(nil), assets...), nil
	}
	filtered := make([]asset_manager.ContextualAsset, 0, len(*assetIds))
	for _, assetID := range *assetIds {
		if asset, ok := s.assetsByID[assetID]; ok && asset.AssetType == assetType {
			filtered = append(filtered, asset)
		}
	}
	return filtered, nil
}

func (s *benchmarkAssetStore) GetAssetsByRefs(ctx context.Context, kbId string, assetType string, refIds *[]string, refTypes *[]string, version *int, label *string) ([]asset_manager.ContextualAsset, error) {
	if assetType != asset_manager.AssetTypeRelation {
		return nil, nil
	}
	if refIds == nil || refTypes == nil {
		return append([]asset_manager.ContextualAsset(nil), s.relationAssets...), nil
	}
	refIDSet := make(map[string]struct{}, len(*refIds))
	for _, refID := range *refIds {
		refIDSet[refID] = struct{}{}
	}
	refTypeSet := make(map[string]struct{}, len(*refTypes))
	for _, refType := range *refTypes {
		refTypeSet[refType] = struct{}{}
	}
	filtered := make([]asset_manager.ContextualAsset, 0, len(s.relationAssets))
	for _, asset := range s.relationAssets {
		if asset.Refs == nil {
			continue
		}
		for _, ref := range *asset.Refs {
			if _, ok := refIDSet[ref.AssetId]; !ok {
				continue
			}
			if _, ok := refTypeSet[ref.RefType]; !ok {
				continue
			}
			filtered = append(filtered, asset)
			break
		}
	}
	return filtered, nil
}

func (s *benchmarkAssetStore) GetAssetsByKbId(ctx context.Context, kbId string) ([]asset_manager.ContextualAsset, error) {
	return nil, nil
}

func (s *benchmarkAssetStore) CheckAssetsExist(ctx context.Context, kbId string, assetType string, assetIds []string) (map[string]bool, error) {
	return map[string]bool{}, nil
}

func (s *benchmarkAssetStore) InsertAsset(ctx context.Context, asset asset_manager.ContextualAsset) (bool, error) {
	s.inserted = append(s.inserted, asset)
	return true, nil
}

func (s *benchmarkAssetStore) InsertBatchAssets(ctx context.Context, assets []asset_manager.ContextualAsset) (int, error) {
	s.inserted = append(s.inserted[:0], assets...)
	return len(assets), nil
}

func (s *benchmarkAssetStore) UpdateAsset(ctx context.Context, kbId string, assetType string, assetId string, version *int, updatedAsset asset_manager.UpdatedContextualAsset) (*asset_manager.ContextualAsset, error) {
	return nil, nil
}

func (s *benchmarkAssetStore) DeleteAsset(ctx context.Context, kbId string, assetType string, assetId string, version *int) (bool, error) {
	return true, nil
}

func (s *benchmarkAssetStore) DeleteAssetsByKbId(ctx context.Context, kbId string) (int, error) {
	return 0, nil
}

func (s *benchmarkAssetStore) DeleteAssetsByKbIdAndVersion(ctx context.Context, kbId string, version *int) (int, error) {
	return 0, nil
}

func (s *benchmarkAssetStore) DeleteAssetsByKbIdAndAssetType(ctx context.Context, kbId string, assetType string) (int, error) {
	return 0, nil
}

func makeBenchmarkSources(pageCount, chunksPerPage, chunkSize int) []asset_manager.KnowledgeSource {
	sources := make([]asset_manager.KnowledgeSource, 0, pageCount)
	pageText := strings.Repeat("a", chunksPerPage*chunkSize)
	for i := 0; i < pageCount; i++ {
		content := pageText
		metadata := map[string]any{"page": i}
		sources = append(sources, asset_manager.KnowledgeSource{
			SourceType:     asset_manager.AssetTypePage,
			SourceName:     fmt.Sprintf("kb:benchmark.pdf:%d", i+1),
			SourceContents: &content,
			Metadata:       &metadata,
		})
	}
	return sources
}

func newBenchmarkService(pageCount, chunksPerPage, chunkSize int) (*GraphRAGServiceImpl, *benchmarkAssetStore, *benchmarkVectorStore, []asset_manager.KnowledgeSource) {
	assetStore := &benchmarkAssetStore{}
	vectorStore := &benchmarkVectorStore{}
	embedder := &benchmarkEmbedder{dim: 64}
	vectorStore.embedder = embedder

	service := &GraphRAGServiceImpl{
		serviceName: "benchmark",
		VectorStore: vectorStore,
		AssetStore:  assetStore,
		GraphRAGServiceCompileTimeOptions: GraphRAGServiceCompileTimeOptions{
			MaxConcurrent:     1,
			BatchSize:         8,
			EmbedBatchSize:    32,
			ChunkingExtractor: &benchmarkChunkExtractor{chunksPerPage: chunksPerPage, chunkSize: chunkSize},
			GraphExtractor:    &benchmarkGraphExtractor{LoggingCapacity: *asset_manager.GetDefaultLoggingCapacity()},
			Embedder:          embedder,
		},
		GraphRAGServiceRuntimeOptions: *NewGraphRAGServiceRuntimeOptionsFromConfig(nil, &NewGraphRAGServiceOptions().GraphRAGServiceRuntimeOptions),
		LoggingCapacity:               *asset_manager.GetDefaultLoggingCapacity(),
	}

	return service, assetStore, vectorStore, makeBenchmarkSources(pageCount, chunksPerPage, chunkSize)
}

func newBenchmarkRetrieveService(entityCount int) (*GraphRAGServiceImpl, []string, []asset_manager.ContextualAsset) {
	assetStore := &benchmarkAssetStore{
		assetsByType:   make(map[string][]asset_manager.ContextualAsset),
		assetsByID:     make(map[string]asset_manager.ContextualAsset),
		relationAssets: make([]asset_manager.ContextualAsset, 0, entityCount),
	}
	vectorStore := &benchmarkVectorStore{}
	embedder := &benchmarkEmbedder{dim: 64}
	vectorStore.embedder = embedder

	kbID := "kb-retrieve"
	version := 1
	seedAssets := []asset_manager.ContextualAsset{
		{
			KbId:      kbID,
			AssetType: asset_manager.AssetTypeDocument,
			AssetId:   kbID + ":doc",
			Version:   version,
			Content:   "seed",
			Refs:      &[]asset_manager.AssetRef{},
		},
	}

	queryResults := make([]asset_manager.RetrievedVector, 0, min(entityCount, 32))
	for i := 0; i < entityCount; i++ {
		chunkID := fmt.Sprintf("%s:chunk-%d", kbID, i)
		chunkRefs := []asset_manager.AssetRef{}
		chunkAsset := asset_manager.ContextualAsset{
			KbId:      kbID,
			AssetType: asset_manager.AssetTypeChunk,
			AssetId:   chunkID,
			Version:   version,
			Content:   fmt.Sprintf("chunk-%d", i),
			Refs:      &chunkRefs,
		}
		assetStore.assetsByType[asset_manager.AssetTypeChunk] = append(assetStore.assetsByType[asset_manager.AssetTypeChunk], chunkAsset)
		assetStore.assetsByID[chunkID] = chunkAsset

		entityRefs := []asset_manager.AssetRef{
			{
				KbId:      kbID,
				AssetType: asset_manager.AssetTypeChunk,
				AssetId:   chunkID,
				RefType:   asset_manager.AssetRefTypeParent,
			},
		}
		entityID := fmt.Sprintf("%s:entity-%d", kbID, i)
		entityAsset := asset_manager.ContextualAsset{
			KbId:      kbID,
			AssetType: asset_manager.AssetTypeEntity,
			AssetId:   entityID,
			Version:   version,
			Content:   fmt.Sprintf("entity-%d", i),
			Refs:      &entityRefs,
		}
		assetStore.assetsByType[asset_manager.AssetTypeEntity] = append(assetStore.assetsByType[asset_manager.AssetTypeEntity], entityAsset)
		assetStore.assetsByID[entityID] = entityAsset

		if i < cap(queryResults) {
			score := float32(entityCount-i) / float32(entityCount)
			queryResults = append(queryResults, asset_manager.RetrievedVector{
				VectorAsset: asset_manager.VectorAsset{AssetId: entityID},
				Score:       &score,
			})
		}
	}

	for i := 0; i < entityCount; i++ {
		sourceID := fmt.Sprintf("%s:entity-%d", kbID, i)
		targetID := fmt.Sprintf("%s:entity-%d", kbID, (i+1)%entityCount)
		chunkID := fmt.Sprintf("%s:chunk-%d", kbID, i)
		relationRefs := []asset_manager.AssetRef{
			{
				KbId:      kbID,
				AssetType: asset_manager.AssetTypeEntity,
				AssetId:   sourceID,
				RefType:   asset_manager.AssetRefTypeSource,
			},
			{
				KbId:      kbID,
				AssetType: asset_manager.AssetTypeEntity,
				AssetId:   targetID,
				RefType:   asset_manager.AssetRefTypeTarget,
			},
			{
				KbId:      kbID,
				AssetType: asset_manager.AssetTypeChunk,
				AssetId:   chunkID,
				RefType:   asset_manager.AssetRefTypeParent,
			},
		}
		relationAsset := asset_manager.ContextualAsset{
			KbId:      kbID,
			AssetType: asset_manager.AssetTypeRelation,
			AssetId:   fmt.Sprintf("%s:relation-%d", kbID, i),
			Version:   version,
			Content:   fmt.Sprintf("rel-%d", i),
			Refs:      &relationRefs,
		}
		assetStore.relationAssets = append(assetStore.relationAssets, relationAsset)
		assetStore.assetsByType[asset_manager.AssetTypeRelation] = append(assetStore.assetsByType[asset_manager.AssetTypeRelation], relationAsset)
		assetStore.assetsByID[relationAsset.AssetId] = relationAsset
	}

	vectorStore.queryResultsByMarker = map[float32][]asset_manager.RetrievedVector{
		1: queryResults,
		2: queryResults,
		3: queryResults,
	}

	service := &GraphRAGServiceImpl{
		serviceName: "benchmark",
		VectorStore: vectorStore,
		AssetStore:  assetStore,
		GraphRAGServiceCompileTimeOptions: GraphRAGServiceCompileTimeOptions{
			MaxConcurrent: 1,
			BatchSize:     8,
			GraphExtractor: &benchmarkRetrieveGraphExtractor{
				benchmarkGraphExtractor{LoggingCapacity: *asset_manager.GetDefaultLoggingCapacity()},
			},
			Embedder: embedder,
		},
		GraphRAGServiceRuntimeOptions: *NewGraphRAGServiceRuntimeOptionsFromConfig(nil, &NewGraphRAGServiceOptions().GraphRAGServiceRuntimeOptions),
		LoggingCapacity:               *asset_manager.GetDefaultLoggingCapacity(),
	}

	return service, []string{kbID}, seedAssets
}

func TestBenchmarkIndexFixtureSanity(t *testing.T) {
	service, assetStore, vectorStore, sources := newBenchmarkService(3, 2, 32)

	assets, vectors, err := service.Index(context.Background(), "kb", sources, nil, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(assets) != 28 {
		t.Fatalf("expected 28 assets, got %d", len(assets))
	}
	if len(vectors) != 12 {
		t.Fatalf("expected 12 vectors, got %d", len(vectors))
	}
	if len(assetStore.inserted) != len(assets) {
		t.Fatalf("asset store inserted %d assets, want %d", len(assetStore.inserted), len(assets))
	}
	if len(vectorStore.inserted) != len(vectors) {
		t.Fatalf("vector store inserted %d vectors, want %d", len(vectorStore.inserted), len(vectors))
	}
}

func TestBenchmarkRetrieveFixtureSanity(t *testing.T) {
	service, kbIDs, seedAssets := newBenchmarkRetrieveService(32)

	results, propagated, err := service.Retrieve(context.Background(), "benchmark-query", kbIDs, seedAssets, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected retrieved assets")
	}
	if len(propagated) == 0 {
		t.Fatal("expected propagated assets")
	}
}

func BenchmarkGraphRAGIndex(b *testing.B) {
	cases := []struct {
		name          string
		pageCount     int
		chunksPerPage int
		chunkSize     int
	}{
		{
			name:          "pages=200/chunks-per-page=4/chunk-size=256",
			pageCount:     200,
			chunksPerPage: 4,
			chunkSize:     256,
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			service, _, _, sources := newBenchmarkService(tc.pageCount, tc.chunksPerPage, tc.chunkSize)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				kbID := fmt.Sprintf("kb-%d", i)
				if _, _, err := service.Index(context.Background(), kbID, sources, nil, nil, nil); err != nil {
					b.Fatalf("unexpected error: %v", err)
				}
			}
		})
	}
}

func BenchmarkGraphRAGRetrieve(b *testing.B) {
	cases := []struct {
		name        string
		entityCount int
	}{
		{
			name:        "entities=512/ring-relations=512/query-results=32",
			entityCount: 512,
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			service, kbIDs, seedAssets := newBenchmarkRetrieveService(tc.entityCount)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, _, err := service.Retrieve(context.Background(), "benchmark-query", kbIDs, seedAssets, nil); err != nil {
					b.Fatalf("unexpected error: %v", err)
				}
			}
		})
	}
}
