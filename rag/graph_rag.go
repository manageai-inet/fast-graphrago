package rag

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/manageai-inet/fast-graphrago/graph"
	"github.com/manageai-inet/fast-graphrago/llm"
	"github.com/manageai-inet/fast-graphrago/models"
	"github.com/manageai-inet/fast-graphrago/utils"

	"github.com/james-bowman/sparse"

	asset_manager "github.com/manageai-inet/agentic-assets"
	chunk "github.com/manageai-inet/agentic-assets/chunk"
)

const LogNoNewChunksBypassExtraction = "No new chunks found, bypassing extraction"

type EntityMetadata struct {
	EntityName  string   `json:"entity_name"`
	EntityType  string   `json:"entity_type"`
	Description string   `json:"description"`
	SourceChunk []string `json:"source_chunk"`
}

type RelationMetadata struct {
	RelationSource string   `json:"relation_source"`
	RelationTarget string   `json:"relation_target"`
	Description    string   `json:"description"`
	SourceChunk    []string `json:"source_chunk"`
}

// GraphRAGServiceImpl coordinates extraction and storage.
type GraphRAGServiceImpl struct {
	serviceName string
	VectorStore asset_manager.VectorStorage
	AssetStore  asset_manager.AssetStorage
	GraphRAGServiceCompileTimeOptions
	GraphRAGServiceRuntimeOptions
	asset_manager.LoggingCapacity
}

func NewGraphRAGService(
	llm llm.LLM,
	vecStore asset_manager.VectorStorage,
	assetStore asset_manager.AssetStorage,
	opts ...Option,
) *GraphRAGServiceImpl {
	options := NewGraphRAGServiceOptions()
	chunkingExtractor := chunk.NewSimpleChunkExtractor(nil, nil)
	graphExtractor := graph.NewFastGraphExtractor(llm)
	options.Apply(WithChunkingExtractor(chunkingExtractor), WithGraphExtractor(graphExtractor))
	for _, opt := range opts {
		opt(options)
	}
	return &GraphRAGServiceImpl{
		serviceName:                       "fastgraphrag",
		VectorStore:                       vecStore,
		AssetStore:                        assetStore,
		GraphRAGServiceCompileTimeOptions: options.GraphRAGServiceCompileTimeOptions,
		GraphRAGServiceRuntimeOptions:     options.GraphRAGServiceRuntimeOptions,
		LoggingCapacity:                   *asset_manager.GetDefaultLoggingCapacity(),
	}
}

// batchEmbed embeds entities in batches using EmbedBatch, returning one VectorAsset per entity.
// Batches are processed sequentially; results are ordered to match entities.
func batchEmbed(ctx context.Context, entities []asset_manager.ContextualAsset, embedder asset_manager.Embedder, batchSize int) ([]asset_manager.VectorAsset, error) {
	return batchEmbedWithRetry(ctx, entities, embedder, batchSize, nil, utils.DefaultEmbedRetryAttempts, utils.DefaultEmbedRetryDelay)
}

func batchEmbedWithRetry(
	ctx context.Context,
	entities []asset_manager.ContextualAsset,
	embedder asset_manager.Embedder,
	batchSize int,
	logger *slog.Logger,
	retryAttempts int,
	retryDelay time.Duration,
) ([]asset_manager.VectorAsset, error) {
	if batchSize <= 0 {
		batchSize = utils.DefaultEmbedBatchSize
	}
	if retryAttempts <= 0 {
		retryAttempts = utils.DefaultEmbedRetryAttempts
	}
	if retryDelay < 0 {
		retryDelay = 0
	}
	n := len(entities)
	numBatches := (n + batchSize - 1) / batchSize
	model := embedder.GetEmbeddingModel()

	vectorAssets := make([]asset_manager.VectorAsset, n)

	for b := 0; b < numBatches; b++ {
		start := b * batchSize
		end := min(start+batchSize, n)
		batch := entities[start:end]

		contents := make([]string, len(batch))
		for i, e := range batch {
			contents[i] = e.Content
		}
		vectors, err := embedBatchVectorsWithRetry(ctx, embedder, contents, logger, model, retryAttempts, retryDelay)
		if err != nil {
			if len(batch) == 1 {
				return nil, err
			}
			if logger != nil {
				logger.WarnContext(ctx, "embedding batch failed after retries, falling back to per-item embedding",
					slog.String("embeddingModel", model),
					slog.Int("batchSize", len(batch)),
					slog.Int("retryAttempts", retryAttempts),
					slog.Duration("retryDelay", retryDelay),
					slog.String("error", err.Error()),
				)
			}
			vectors, err = embedIndividuallyWithRetry(ctx, embedder, contents, logger, model, retryAttempts, retryDelay)
			if err != nil {
				return nil, err
			}
		}
		if len(vectors) != len(batch) {
			return nil, fmt.Errorf("embedder returned %d vectors for %d inputs", len(vectors), len(batch))
		}
		for i, e := range batch {
			parentRef := asset_manager.AssetRef{
				KbId:      e.KbId,
				AssetType: e.AssetType,
				AssetId:   e.AssetId,
				RefType:   asset_manager.AssetRefTypeParent,
			}
			refs := []asset_manager.AssetRef{parentRef}
			m := model
			vectorAssets[start+i] = asset_manager.VectorAsset{
				KbId:           e.KbId,
				AssetId:        e.AssetId,
				Version:        e.Version,
				Content:        e.Content,
				Refs:           &refs,
				Labels:         e.Labels,
				Metadata:       e.Metadata,
				EmbeddingModel: &m,
				EmbededVector:  vectors[i],
			}
		}
	}
	return vectorAssets, nil
}

func embedBatchVectorsWithRetry(ctx context.Context, embedder asset_manager.Embedder, contents []string, logger *slog.Logger, model string, attempts int, delay time.Duration) ([][]float32, error) {
	var lastErr error
	for attempt := 1; attempt <= attempts; attempt++ {
		vectors, err := embedder.EmbedBatch(ctx, contents)
		if err == nil {
			return vectors, nil
		}
		lastErr = err
		if attempt == attempts {
			break
		}
		if logger != nil {
			logger.WarnContext(ctx, "retrying embedding batch after failure",
				slog.String("embeddingModel", model),
				slog.Int("inputCount", len(contents)),
				slog.Int("attempt", attempt),
				slog.Int("maxAttempts", attempts),
				slog.Duration("retryDelay", delay),
				slog.String("error", err.Error()),
			)
		}
		if err := sleepWithContext(ctx, delay); err != nil {
			return nil, err
		}
	}
	return nil, lastErr
}

func embedIndividuallyWithRetry(ctx context.Context, embedder asset_manager.Embedder, contents []string, logger *slog.Logger, model string, attempts int, delay time.Duration) ([][]float32, error) {
	vectors := make([][]float32, len(contents))
	for i, content := range contents {
		vector, err := embedSingleWithRetry(ctx, embedder, content, logger, model, i, attempts, delay)
		if err != nil {
			return nil, err
		}
		vectors[i] = vector
	}
	return vectors, nil
}

func embedSingleWithRetry(ctx context.Context, embedder asset_manager.Embedder, content string, logger *slog.Logger, model string, index int, attempts int, delay time.Duration) ([]float32, error) {
	var lastErr error
	for attempt := 1; attempt <= attempts; attempt++ {
		vector, err := embedder.Embed(ctx, content)
		if err == nil {
			return vector, nil
		}
		lastErr = err
		if attempt == attempts {
			break
		}
		if logger != nil {
			logger.WarnContext(ctx, "retrying single-item embedding after failure",
				slog.String("embeddingModel", model),
				slog.Int("itemIndex", index),
				slog.Int("attempt", attempt),
				slog.Int("maxAttempts", attempts),
				slog.Duration("retryDelay", delay),
				slog.String("error", err.Error()),
			)
		}
		if err := sleepWithContext(ctx, delay); err != nil {
			return nil, err
		}
	}
	if logger != nil {
		logger.ErrorContext(ctx, "single-item embedding failed after retries",
			slog.String("embeddingModel", model),
			slog.Int("itemIndex", index),
			slog.String("inputSnippet", truncateEmbeddingLog(content, 300)),
			slog.String("error", lastErr.Error()),
		)
	}
	return nil, fmt.Errorf("single-item embedding failed at batch index %d: %w", index, lastErr)
}

func sleepWithContext(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return nil
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func truncateEmbeddingLog(value string, limit int) string {
	clean := strings.TrimSpace(value)
	if len(clean) <= limit {
		return clean
	}
	return clean[:limit] + "...<truncated>"
}

func (g *GraphRAGServiceImpl) SetLogger(logger *slog.Logger) {
	g.LoggingCapacity.SetLogger(logger)
	g.GraphExtractor.SetLogger(logger)
}

func preparePageAssets(serviceName, kbId, filename string, version int, sources []asset_manager.KnowledgeSource, labels *[]string, seedAsset *asset_manager.ContextualAsset) ([]asset_manager.ContextualAsset, error) {
	pageAssets := make([]asset_manager.ContextualAsset, 0, len(sources))
	for i, source := range sources {
		if source.SourceName == "" {
			return nil, fmt.Errorf("page source of kbId %s at position %d has empty source name", kbId, i)
		}
		if source.SourceContents == nil || *source.SourceContents == "" {
			return nil, fmt.Errorf("page source name %s of kbId %s at position %d has empty source contents", source.SourceName, kbId, i)
		}
		pageId := fmt.Sprintf("%s:%s:%d", kbId, filename, i)
		pageRefs := []asset_manager.AssetRef{
			{
				KbId:      kbId,
				AssetId:   seedAsset.AssetId,
				AssetType: asset_manager.AssetTypeDocument,
				RefType:   asset_manager.AssetRefTypeParent,
			},
		}
		pageAssets = append(pageAssets, asset_manager.ContextualAsset{
			IndexedBy: serviceName,
			KbId:      kbId,
			AssetType: asset_manager.AssetTypePage,
			AssetId:   pageId,
			Version:   version,
			Content:   *source.SourceContents,
			Metadata:  source.Metadata,
			Labels:    labels,
			Refs:      &pageRefs,
		})
		*seedAsset.Refs = append(*seedAsset.Refs, asset_manager.AssetRef{
			KbId:      kbId,
			AssetId:   pageId,
			AssetType: asset_manager.AssetTypePage,
			RefType:   asset_manager.AssetRefTypeChild,
		})
	}
	return pageAssets, nil
}

func prepareChunkAssets(serviceName, kbId string, version int, chunks []chunk.ChunkImmediateAsset, pageAssets []asset_manager.ContextualAsset) ([]asset_manager.ContextualAsset, error) {
	chunkAssets := make([]asset_manager.ContextualAsset, 0, len(chunks))
	for _, c := range chunks {
		pageNumber := c.PageNumber
		if pageNumber >= len(pageAssets) {
			return nil, fmt.Errorf("page asset not found for page number %d (page assets length: %d)", pageNumber, len(pageAssets))
		}
		pageRefs := pageAssets[pageNumber].Refs
		if pageRefs == nil {
			refs := []asset_manager.AssetRef{}
			pageRefs = &refs
			pageAssets[pageNumber].Refs = pageRefs
		}
		page := pageAssets[pageNumber]
		refs := []asset_manager.AssetRef{
			{
				KbId:      kbId,
				AssetId:   page.AssetId,
				AssetType: page.AssetType,
				RefType:   asset_manager.AssetRefTypeParent,
			},
		}
		chunkId := fmt.Sprintf("%s[%d:%d]", page.AssetId, c.StartPos, c.StartPos+len(c.Content))
		chunkAssets = append(chunkAssets, asset_manager.ContextualAsset{
			IndexedBy: serviceName,
			KbId:      kbId,
			AssetType: asset_manager.AssetTypeChunk,
			AssetId:   chunkId,
			Version:   version,
			Content:   c.Content,
			Metadata:  c.Metadata,
			Refs:      &refs,
		})
		*pageRefs = append(*pageRefs, asset_manager.AssetRef{
			KbId:      kbId,
			AssetId:   chunkId,
			AssetType: asset_manager.AssetTypeChunk,
			RefType:   asset_manager.AssetRefTypeChild,
		})
	}
	return chunkAssets, nil
}

func dedupeContextualAssetsByID(assets []asset_manager.ContextualAsset) ([]asset_manager.ContextualAsset, map[string]asset_manager.ContextualAsset, []string) {
	deduped := make([]asset_manager.ContextualAsset, 0, len(assets))
	byID := make(map[string]asset_manager.ContextualAsset, len(assets))
	duplicates := make([]string, 0)
	for _, asset := range assets {
		if _, exists := byID[asset.AssetId]; exists {
			duplicates = append(duplicates, asset.AssetId)
			continue
		}
		byID[asset.AssetId] = asset
		deduped = append(deduped, asset)
	}
	return deduped, byID, duplicates
}

func (g *GraphRAGServiceImpl) Index(ctx context.Context, kbId string, sources []asset_manager.KnowledgeSource, labels *[]string, metadata *map[string]any, config *map[string]any) ([]asset_manager.ContextualAsset, []asset_manager.VectorAsset, error) {
	logger := asset_manager.GetLogger(&g.LoggingCapacity)
	logger.InfoContext(ctx, "starting to index", slog.String("kbId", kbId), slog.Int("sources", len(sources)))
	if kbId == "" {
		logger.ErrorContext(ctx, "kbId is empty")
		return []asset_manager.ContextualAsset{}, []asset_manager.VectorAsset{}, fmt.Errorf("kbId is empty")
	}
	if len(sources) == 0 {
		logger.ErrorContext(ctx, "input sources is empty", slog.String("kbId", kbId))
		return []asset_manager.ContextualAsset{}, []asset_manager.VectorAsset{}, fmt.Errorf("sources is empty")
	}
	splitedSourceName := strings.Split(sources[0].SourceName, ":")
	if len(splitedSourceName) < 3 {
		logger.ErrorContext(ctx, "page source name is not in format kbid:filename.ext:n", slog.String("kbId", kbId), slog.String("sourceName", sources[0].SourceName))
		return []asset_manager.ContextualAsset{}, []asset_manager.VectorAsset{}, fmt.Errorf("page source name %s of kbId %s is not in format kbid:filename.ext:n", sources[0].SourceName, kbId)
	}
	runOpts := NewGraphRAGServiceRuntimeOptionsFromConfig(config, &g.GraphRAGServiceRuntimeOptions)
	logger.DebugContext(ctx, "indexing with options", slog.Any("options", runOpts))

	filename := strings.Join(splitedSourceName[1:len(splitedSourceName)-1], ":")
	tsVersion := int(time.Now().UnixMilli())
	seedAsset := asset_manager.ContextualAsset{
		IndexedBy: g.serviceName,
		KbId:      kbId,
		AssetType: asset_manager.AssetTypeDocument,
		AssetId:   fmt.Sprintf("%s:%s", kbId, filename),
		Version:   tsVersion,
		Content:   fmt.Sprintf("id: %s, file: %s, pages: %d", kbId, filename, len(sources)), // it just used for anchor point for retrieving
		Refs:      &[]asset_manager.AssetRef{},
	}
	runOpts.Version = &tsVersion
	logger.DebugContext(ctx, "indexing new version from seed asset", slog.String("kbId", kbId), slog.Int("version", tsVersion), slog.String("filename", filename), slog.Int("pages", len(sources)))

	// 1. Prepare documents
	pageAssets, err := preparePageAssets(g.serviceName, kbId, filename, tsVersion, sources, labels, &seedAsset)
	if err != nil {
		logger.ErrorContext(ctx, err.Error(), slog.String("kbId", kbId))
		return []asset_manager.ContextualAsset{}, []asset_manager.VectorAsset{}, err
	}
	logger.DebugContext(ctx, "prepared page assets successfully", slog.String("kbId", kbId), slog.Int("pages", len(pageAssets)))

	// 2. Extract Chunks
	logger.DebugContext(ctx, "extracting chunks", slog.String("kbId", kbId))
	chunks, err := g.ChunkingExtractor.Extract(ctx, kbId, sources)
	if err != nil {
		logger.ErrorContext(ctx, err.Error(), slog.String("kbId", kbId))
		return []asset_manager.ContextualAsset{}, []asset_manager.VectorAsset{}, err
	}
	logger.DebugContext(ctx, "extracted chunks successfully", slog.String("kbId", kbId), slog.Int("chunks", len(chunks)))
	chunkAssets, err := prepareChunkAssets(g.serviceName, kbId, tsVersion, chunks, pageAssets)
	if err != nil {
		logger.ErrorContext(ctx, err.Error(), slog.String("kbId", kbId))
		return []asset_manager.ContextualAsset{}, []asset_manager.VectorAsset{}, err
	}
	logger.DebugContext(ctx, "prepared chunk assets successfully", slog.String("kbId", kbId), slog.Int("chunks", len(chunkAssets)))
	// 3. Extract Entities & Relations
	logger.DebugContext(ctx, "extracting entities and relations", slog.String("kbId", kbId), slog.Int("chunks", len(chunkAssets)), slog.String("domain", g.Domain))
	graph, err := g.GraphExtractor.ExtractGraph(
		ctx,
		chunkAssets,
		graph.GraphExtractionOptions{
			Domain:      runOpts.Domain,
			EntityTypes: runOpts.EntityTypes,
		},
	)
	if err != nil {
		logger.ErrorContext(ctx, err.Error(), slog.String("kbId", kbId))
		return []asset_manager.ContextualAsset{}, []asset_manager.VectorAsset{}, err
	}
	logger.DebugContext(ctx, "extracted entities and relations successfully", slog.String("kbId", kbId), slog.Int("entities", len(graph.EntityAssets)), slog.Int("relations", len(graph.RelationAssets)))
	entityKeyToIds := make(map[string]string)
	var entityAssets []asset_manager.ContextualAsset
	for i, ea := range graph.EntityAssets {
		refs := []asset_manager.AssetRef{}
		for _, chunkId := range ea.ChunkIds {
			refs = append(refs, asset_manager.AssetRef{
				KbId:      kbId,
				AssetId:   chunkId,
				AssetType: asset_manager.AssetTypeChunk,
				RefType:   asset_manager.AssetRefTypeParent,
			})
		}
		metadata := map[string]any{
			"index":        i,
			"entity_name":  ea.Name,
			"entity_type":  ea.Type,
			"description":  ea.Description,
			"source_chunk": ea.ChunkIds,
		}
		entityId := fmt.Sprintf("%s:%s:%s:%d", kbId, filename, asset_manager.AssetTypeEntity, i)
		entityContent := fmt.Sprintf("%s | %s | %s", ea.Name, ea.Type, ea.Description)
		entityAssets = append(entityAssets, asset_manager.ContextualAsset{
			IndexedBy: g.serviceName,
			KbId:      kbId,
			AssetType: asset_manager.AssetTypeEntity,
			AssetId:   entityId,
			Version:   tsVersion,
			Content:   entityContent,
			Metadata:  &metadata,
			Refs:      &refs,
		})
		entityKey := fmt.Sprintf("%s|%s", ea.Name, ea.Type)
		entityKeyToIds[entityKey] = entityId
	}
	logger.DebugContext(ctx, "prepared entity assets successfully", slog.String("kbId", kbId), slog.Int("entities", len(entityAssets)))

	relationAssets := []asset_manager.ContextualAsset{}
	for i, ra := range graph.RelationAssets {
		refs := []asset_manager.AssetRef{}
		for _, chunkId := range ra.ChunkIds {
			refs = append(refs, asset_manager.AssetRef{
				KbId:      kbId,
				AssetId:   chunkId,
				AssetType: asset_manager.AssetTypeChunk,
				RefType:   asset_manager.AssetRefTypeParent,
			})
		}
		sourceEntityKey := fmt.Sprintf("%s|%s", ra.Source, ra.SourceType)
		targetEntityKey := fmt.Sprintf("%s|%s", ra.Target, ra.TargetType)
		sourceEntityId, ok := entityKeyToIds[sourceEntityKey]
		if !ok {
			msg := fmt.Sprintf("relation bind to source entity %s which is not found", sourceEntityKey)
			logger.ErrorContext(ctx, msg, slog.String("kbId", kbId))
			return []asset_manager.ContextualAsset{}, []asset_manager.VectorAsset{}, errors.New(msg)
		}
		targetEntityId, ok := entityKeyToIds[targetEntityKey]
		if !ok {
			msg := fmt.Sprintf("relation bind to target entity %s which is not found", targetEntityKey)
			logger.ErrorContext(ctx, msg, slog.String("kbId", kbId))
			return []asset_manager.ContextualAsset{}, []asset_manager.VectorAsset{}, errors.New(msg)
		}
		refs = append(refs, asset_manager.AssetRef{
			KbId:      kbId,
			AssetId:   sourceEntityId,
			AssetType: asset_manager.AssetTypeEntity,
			RefType:   asset_manager.AssetRefTypeSource,
		})
		refs = append(refs, asset_manager.AssetRef{
			KbId:      kbId,
			AssetId:   targetEntityId,
			AssetType: asset_manager.AssetTypeEntity,
			RefType:   asset_manager.AssetRefTypeTarget,
		})

		metadata := map[string]any{
			"relation_source":      ra.Source,
			"relation_source_type": ra.SourceType,
			"relation_target":      ra.Target,
			"relation_target_type": ra.TargetType,
			"description":          ra.Description,
			"source_chunk":         ra.ChunkIds,
		}
		relationId := fmt.Sprintf("%s:%s:%s:%d", kbId, filename, asset_manager.AssetTypeRelation, i)
		relationContent := fmt.Sprintf("%s | %s | %s | %s | %s", ra.Source, ra.SourceType, ra.Target, ra.TargetType, ra.Description)
		relationAssets = append(relationAssets, asset_manager.ContextualAsset{
			IndexedBy: g.serviceName,
			KbId:      kbId,
			AssetType: asset_manager.AssetTypeRelation,
			AssetId:   relationId,
			Version:   tsVersion,
			Content:   relationContent,
			Metadata:  &metadata,
			Refs:      &refs,
		})
	}
	logger.DebugContext(ctx, "prepared relation assets successfully", slog.String("kbId", kbId), slog.Int("relations", len(relationAssets)))

	// 4. Generate Embeddings for Nodes (Entities)
	logger.DebugContext(ctx, "generating embeddings for entities", slog.String("kbId", kbId), slog.Int("entities", len(entityAssets)))
	var vectorAssets []asset_manager.VectorAsset
	if g.Embedder != nil {
		var embedErr error
		vectorAssets, embedErr = batchEmbedWithRetry(ctx, entityAssets, g.Embedder, g.EmbedBatchSize, asset_manager.GetLogger(g), g.EmbedRetryAttempts, g.EmbedRetryDelay)
		if embedErr != nil {
			logger.ErrorContext(ctx, embedErr.Error(), slog.String("kbId", kbId))
			return []asset_manager.ContextualAsset{}, []asset_manager.VectorAsset{}, embedErr
		}
	} else {
		vectorAssets = make([]asset_manager.VectorAsset, 0, len(entityAssets))
		var wg sync.WaitGroup
		var mu sync.Mutex
		sem := make(chan struct{}, g.MaxConcurrent)
		errsCh := make(chan error, len(entityAssets))
		for _, entity := range entityAssets {
			wg.Add(1)
			sem <- struct{}{}
			go func(vecArray *[]asset_manager.VectorAsset, entity *asset_manager.ContextualAsset) {
				defer wg.Done()
				defer func() { <-sem }()
				v, err := g.VectorStore.EmbedAsset(ctx, entity, nil)
				if err != nil {
					errsCh <- err
					return
				}
				entity.EmbeddingModel = v.EmbeddingModel
				mu.Lock()
				*vecArray = append(*vecArray, v)
				mu.Unlock()
			}(&vectorAssets, &entity)
		}
		done := make(chan struct{})
		go func() {
			wg.Wait()
			close(done)
		}()
		select {
		case err := <-errsCh:
			logger.ErrorContext(ctx, err.Error(), slog.String("kbId", kbId))
			return []asset_manager.ContextualAsset{}, []asset_manager.VectorAsset{}, err
		case <-done:
		}
	}
	logger.DebugContext(ctx, "generated embeddings for entities successfully", slog.String("kbId", kbId), slog.Int("entities", len(vectorAssets)))

	allAssets := make([]asset_manager.ContextualAsset, 0, 1+len(pageAssets)+len(chunkAssets)+len(entityAssets)+len(relationAssets))
	allAssets = append(allAssets, seedAsset)
	allAssets = append(allAssets, pageAssets...)
	allAssets = append(allAssets, chunkAssets...)
	allAssets = append(allAssets, entityAssets...)
	allAssets = append(allAssets, relationAssets...)
	logger.DebugContext(ctx, "prepared all assets successfully", slog.String("kbId", kbId), slog.Int("assets", len(allAssets)))

	// 5. Store Vectors
	logger.DebugContext(ctx, "storing vectors", slog.String("kbId", kbId), slog.Int("vectors", len(vectorAssets)))
	_, err = g.VectorStore.InsertBatchVectors(ctx, vectorAssets)
	if err != nil {
		logger.ErrorContext(ctx, err.Error(), slog.String("kbId", kbId))
		return []asset_manager.ContextualAsset{}, []asset_manager.VectorAsset{}, err
	}
	logger.DebugContext(ctx, "stored vectors successfully", slog.String("kbId", kbId), slog.Int("vectors", len(vectorAssets)))

	// 6. Store Assets
	logger.DebugContext(ctx, "storing assets", slog.String("kbId", kbId), slog.Int("assets", len(allAssets)))
	_, err = g.AssetStore.InsertBatchAssets(ctx, allAssets)
	if err != nil {
		// delete vectors
		_, derr := g.VectorStore.DeleteVectorsByKbIdAndVersion(ctx, kbId, &tsVersion)
		if derr != nil {
			msg := fmt.Sprintf("failed to delete vectors: %s", derr.Error())
			logger.ErrorContext(ctx, msg, slog.String("kbId", kbId))
		}
		logger.ErrorContext(ctx, err.Error(), slog.String("kbId", kbId))
		return []asset_manager.ContextualAsset{}, []asset_manager.VectorAsset{}, err
	}
	logger.DebugContext(ctx, "stored assets successfully", slog.String("kbId", kbId), slog.Int("assets", len(allAssets)))

	logger.InfoContext(ctx, "finished indexing for kbId", slog.String("kbId", kbId))
	return allAssets, vectorAssets, nil
}

func (g *GraphRAGServiceImpl) getTransitionMatrixForKbId(ctx context.Context, entityAssets []asset_manager.ContextualAsset, label *string) (models.TransitionMatrix, error) {
	logger := asset_manager.GetLogger(g)
	kbId := entityAssets[0].KbId
	version := entityAssets[0].Version

	entityIndiceToId := make([]string, 0, len(entityAssets))
	entityIDToIndex := make(map[string]int, len(entityAssets))
	for idx, entityAsset := range entityAssets {
		entityIndiceToId = append(entityIndiceToId, entityAsset.AssetId)
		entityIDToIndex[entityAsset.AssetId] = idx
	}
	pDOK := sparse.NewDOK(len(entityIndiceToId), len(entityIndiceToId))
	rDim, cDim := pDOK.Dims()
	relationAssets, err := g.AssetStore.GetAssetsByRefs(
		ctx,
		kbId,
		asset_manager.AssetTypeRelation,
		&entityIndiceToId,
		&[]string{asset_manager.AssetRefTypeSource, asset_manager.AssetRefTypeTarget},
		&version,
		label,
	)
	if err != nil {
		logger.ErrorContext(ctx, err.Error(), slog.String("kbId", kbId))
		return models.TransitionMatrix{}, err
	}
	if len(relationAssets) == 0 {
		msg := "not found any relation asset when search by refs"
		logger.ErrorContext(ctx, msg, slog.String("kbId", kbId))
		return models.TransitionMatrix{}, errors.New(msg)
	}

	for _, relationAsset := range relationAssets {
		var sourceRef asset_manager.AssetRef
		var targetRef asset_manager.AssetRef
		hasSource := false
		hasTarget := false
		for _, ref := range *relationAsset.Refs {
			if ref.RefType == asset_manager.AssetRefTypeSource {
				sourceRef = ref
				hasSource = true
			} else if ref.RefType == asset_manager.AssetRefTypeTarget {
				targetRef = ref
				hasTarget = true
			}
			if hasSource && hasTarget {
				break
			}
		}
		if !(hasSource && hasTarget) {
			msg := "relation asset is not complete, missing source or target entity, skipping"
			logger.WarnContext(
				ctx, msg,
				slog.String("kbId", kbId),
				slog.String("assetId", relationAsset.AssetId),
			)
			continue
		}
		sourceIndex, sourceOK := entityIDToIndex[sourceRef.AssetId]
		targetIndex, targetOK := entityIDToIndex[targetRef.AssetId]
		if !sourceOK || !targetOK {
			msg := "relation asset is not complete, source or target entity not found in entityAssets, skipping"
			logger.WarnContext(
				ctx, msg,
				slog.String("kbId", kbId),
				slog.String("assetId", relationAsset.AssetId),
				slog.String("sourceRef", sourceRef.AssetId),
				slog.String("targetRef", targetRef.AssetId),
			)
			continue
		}
		// check if index not out of bound
		if sourceIndex >= rDim || targetIndex >= cDim {
			msg := "relation asset is not complete, source or target entity index out of bound, skipping"
			logger.WarnContext(
				ctx, msg,
				slog.String("kbId", kbId),
				slog.String("assetId", relationAsset.AssetId),
				slog.Int("sourceIndex", sourceIndex),
				slog.Int("targetIndex", targetIndex),
				slog.Int("rDim", rDim),
				slog.Int("cDim", cDim),
			)
			continue
		}
		val := pDOK.At(sourceIndex, targetIndex)
		pDOK.Set(sourceIndex, targetIndex, val+1)
	}
	pCSR := pDOK.ToCSR()
	if pCSR == nil {
		return models.TransitionMatrix{}, fmt.Errorf("failed to create transition matrix for kbId: %s", kbId)
	}
	rm := pCSR.RawMatrix()
	data := rm.Data
	indptr := rm.Indptr

	for i := 0; i < len(indptr)-1; i++ {
		start, end := indptr[i], indptr[i+1]
		if start == end {
			continue
		} // Skip empty rows

		// Step 1: Accumulate log-weights for the row
		var rowSum float64
		for k := start; k < end; k++ {
			// Weight Clipping to prevent node with high degree from dominating the random walk
			// Log1p is the "FastGraphRAG" standard for weight clipping
			data[k] = math.Log1p(data[k])
			rowSum += data[k]
		}

		// Step 2: Scale the row to sum to 1.0
		if rowSum > 0 {
			invSum := 1.0 / rowSum // Multiply by inverse is faster than repeated division
			for k := start; k < end; k++ {
				data[k] *= invSum
			}
		}
	}

	return models.TransitionMatrix{CSR: pCSR, EntityIndiceToId: entityIndiceToId, EntityIDToIndex: entityIDToIndex, Dim: len(entityIndiceToId)}, nil
}

func (g *GraphRAGServiceImpl) getSeedVector(
	ctx context.Context,
	kbId string,
	queryEntities []models.QueryEntity,
	transitionMatrix models.TransitionMatrix,
	topK int,
	threshold float32,
	weights [3]float32,
	version *int,
	label *string,
) ([]float32, error) {
	logger := asset_manager.GetLogger(g)
	filter := asset_manager.VectorQueryFilter{
		KbId:    &kbId,
		Version: version,
		Label:   label,
	}

	sVec := make([]float32, transitionMatrix.Dim)
	var countNamed int
	var countGeneric int
	for _, queryEntity := range queryEntities {
		if queryEntity.Type == models.NamedEntityType {
			countNamed++
		}
		if queryEntity.Type == models.GenericEntityType {
			countGeneric++
		}
	}
	var w [3]float32
	w[0] = weights[0] / float32(countNamed)
	w[1] = weights[1] / float32(countGeneric)
	w[2] = weights[2]

	var wg sync.WaitGroup
	var mu sync.Mutex
	// query in concurrency
	sem := make(chan struct{}, g.MaxConcurrent)
	for _, queryEntity := range queryEntities {
		wg.Add(1)
		sem <- struct{}{}
		go func(qe models.QueryEntity) {
			defer wg.Done()
			defer func() { <-sem }()
			retrievedAssets, err := g.VectorStore.QueryVectors(ctx, qe.Embedding, &topK, &threshold, &filter)
			if err != nil {
				// if failed to retrieve, skip this query entity
				msg := "failed to retrieve vectors for query entity, skipping, cause: " + err.Error()
				logger.WarnContext(
					ctx, msg,
					slog.String("kbId", kbId),
					slog.String("entity", qe.Name),
					slog.String("entityType", qe.Type),
				)
				return
			}
			weight := w[2]
			if qe.Type == models.NamedEntityType {
				weight = w[0]
			} else if qe.Type == models.GenericEntityType {
				weight = w[1]
			}
			mu.Lock()
			for _, retrievedAsset := range retrievedAssets {
				idx, ok := transitionMatrix.EntityIDToIndex[retrievedAsset.AssetId]
				if !ok {
					continue
				}
				if retrievedAsset.Score == nil {
					msg := "retrieved asset has no score, skipping"
					logger.WarnContext(
						ctx, msg,
						slog.String("kbId", kbId),
						slog.String("entity", retrievedAsset.AssetId),
						slog.String("entityType", qe.Type),
					)
					continue
				}
				sVec[idx] += float32(*retrievedAsset.Score) * weight
			}
			mu.Unlock()
		}(queryEntity)
	}
	wg.Wait()
	// normalize the seed vector
	sVecSum := float32(0.0)
	for _, v := range sVec {
		sVecSum += v
	}
	if sVecSum != 0.0 {
		normFactor := 1.0 / sVecSum
		for i := range sVec {
			sVec[i] *= normFactor
		}
	}
	// if sum is 0 mean seed is all zero, no need to normalize
	return sVec, nil
}

func (g *GraphRAGServiceImpl) ppr(ctx context.Context, s []float32, P *sparse.CSR, alpha float32) ([]float32, error) {
	// Ensure s and P dimension
	rm := P.RawMatrix()
	if rm.I != rm.J {
		return []float32{}, fmt.Errorf("P is not a square matrix")
	}
	if len(s) != rm.I {
		return []float32{}, fmt.Errorf("s and P dimension mismatch")
	}
	data := rm.Data
	indptr := rm.Indptr
	indices := rm.Ind
	_, cols := P.Dims()

	// 2. Initialize v as the seed vector
	v := make([]float32, cols)
	copy(v, s)
	vNext := make([]float32, cols)

	oneMinusAlpha := 1.0 - alpha

	// 3. Power Iteration Loop
	for it := 0; it < g.MaxWalks; it++ {
		for i := range vNext {
			vNext[i] = 0
		}

		// 1. Random Walk Step: vNext = v * P
		for i := 0; i < len(v); i++ {
			if v[i] == 0 {
				continue
			}

			start, end := indptr[i], indptr[i+1]
			for k := start; k < end; k++ {
				colIdx := indices[k]
				vNext[colIdx] += v[i] * float32(data[k])
			}
		}

		// 2. Personalization Step: v = (alpha * vNext) + ((1-alpha) * s)
		for i := 0; i < cols; i++ {
			v[i] = (alpha * vNext[i]) + (oneMinusAlpha * s[i])
		}
	}

	return v, nil
}

func (g *GraphRAGServiceImpl) retrieveForKbId(ctx context.Context, queryEntities []models.QueryEntity, kbId string, seedAssets []asset_manager.ContextualAsset, config *map[string]any) ([]asset_manager.RetrievedAsset, error) {
	logger := asset_manager.GetLogger(g)
	logger.InfoContext(ctx, "starting to retrieve", slog.String("kbId", kbId), slog.Int("queryEntities", len(queryEntities)), slog.Int("seedAssets", len(seedAssets)))
	if len(queryEntities) == 0 {
		logger.ErrorContext(ctx, "no query provided", slog.String("kbId", kbId))
		return []asset_manager.RetrievedAsset{}, fmt.Errorf("no query provided")
	}
	filteredSeedAssets := []asset_manager.ContextualAsset{}
	for _, seedAsset := range seedAssets {
		if seedAsset.KbId == kbId {
			filteredSeedAssets = append(filteredSeedAssets, seedAsset)
		}
	}
	if len(filteredSeedAssets) == 0 {
		logger.ErrorContext(ctx, "no seed assets found for kbId", slog.String("kbId", kbId))
		return []asset_manager.RetrievedAsset{}, fmt.Errorf("no seed assets found for kbId: %s", kbId)
	}
	logger.DebugContext(ctx, "filtered seed assets successfully", slog.Any("filteredSeedAssets", filteredSeedAssets))
	runOpts := NewGraphRAGServiceRuntimeOptionsFromConfig(config, &g.GraphRAGServiceRuntimeOptions)

	version := filteredSeedAssets[0].Version
	runOpts.Version = &version

	logger.DebugContext(ctx, "find entities for kbId", slog.String("kbId", kbId), slog.Any("version", version), slog.Any("label", runOpts.Label))
	entityAssets, err := g.AssetStore.GetAssets(ctx, kbId, asset_manager.AssetTypeEntity, nil, runOpts.Version, runOpts.Label)
	if err != nil {
		logger.ErrorContext(ctx, err.Error(), slog.String("kbId", kbId))
		return []asset_manager.RetrievedAsset{}, err
	}
	if len(entityAssets) == 0 {
		logger.ErrorContext(ctx, "no entity assets found for kbId", slog.String("kbId", kbId))
		return []asset_manager.RetrievedAsset{}, fmt.Errorf("no entity assets found for kbId: %s", kbId)
	}
	version = entityAssets[0].Version
	runOpts.Version = &version
	logger.DebugContext(ctx, "found entities for kbId", slog.String("kbId", kbId), slog.Any("version", version))

	logger.DebugContext(ctx, "deduplicate entity assets", slog.String("kbId", kbId))
	dedupedEntityAssets, entitiesMap, duplicateEntityIDs := dedupeContextualAssetsByID(entityAssets)
	if len(dedupedEntityAssets) != len(entityAssets) {
		for _, entityID := range duplicateEntityIDs {
			logger.WarnContext(ctx, "duplicate entity asset found for kbId, skipping", slog.String("kbId", kbId), slog.String("entityId", entityID))
		}
		logger.WarnContext(ctx, "entity assets were deduplicated, preserving first-seen order", slog.String("kbId", kbId), slog.Int("originalCount", len(entityAssets)), slog.Int("deduplicatedCount", len(dedupedEntityAssets)))
		entityAssets = dedupedEntityAssets
	}
	logger.DebugContext(ctx, "deduplicated entity assets", slog.String("kbId", kbId), slog.Int("count", len(entityAssets)))

	logger.InfoContext(ctx, "create transition matrix for kbId", slog.String("kbId", kbId), slog.Int("expectedDim", len(entityAssets)))
	transitionMatrix, err := g.getTransitionMatrixForKbId(ctx, entityAssets, runOpts.Label)
	if err != nil {
		logger.ErrorContext(ctx, err.Error(), slog.String("kbId", kbId))
		return []asset_manager.RetrievedAsset{}, err
	}
	logger.InfoContext(ctx, "created transition matrix for kbId successfully", slog.String("kbId", kbId), slog.Any("dim", transitionMatrix.Dim))

	logger.InfoContext(
		ctx, "create seed vector for kbId",
		slog.String("kbId", kbId),
		slog.Int("topK", *runOpts.TopK),
		slog.Float64("threshold", float64(*runOpts.Threshold)),
		slog.Any("queryWeights", runOpts.QueryWeights),
		slog.Any("version", runOpts.Version),
		slog.Any("label", runOpts.Label),
	)
	seedVector, err := g.getSeedVector(ctx, kbId, queryEntities, transitionMatrix, *runOpts.TopK, *runOpts.Threshold, runOpts.QueryWeights, runOpts.Version, runOpts.Label)
	if err != nil {
		logger.ErrorContext(ctx, err.Error(), slog.String("kbId", kbId))
		return []asset_manager.RetrievedAsset{}, err
	}
	logger.InfoContext(ctx, "created seed vector for kbId successfully", slog.String("kbId", kbId), slog.Any("dim", len(seedVector)))

	logger.InfoContext(
		ctx,
		"starting to run ppr for kbId",
		slog.String("kbId", kbId),
		slog.Int("seedDim", len(seedVector)),
		slog.Int("transDim", transitionMatrix.Dim),
		slog.Float64("dampingFactor", float64(runOpts.DampingFactor)),
	)
	rankedVector, err := g.ppr(ctx, seedVector, transitionMatrix.CSR, runOpts.DampingFactor)
	if err != nil {
		logger.ErrorContext(ctx, err.Error(), slog.String("kbId", kbId))
		return []asset_manager.RetrievedAsset{}, err
	}
	logger.InfoContext(ctx, "run ppr for kbId successfully", slog.String("kbId", kbId), slog.Any("dim", len(rankedVector)))

	eid2idx := transitionMatrix.EntityIndiceToId
	scoredEnities := make([]asset_manager.RetrievedAsset, 0, len(entityAssets))
	for idx, score := range rankedVector {
		entityId := eid2idx[idx]
		entityAsset := entitiesMap[entityId]
		retrievedAsset := asset_manager.RetrievedAsset{
			ContextualAsset: entityAsset,
			Score:           &score,
		}
		logger.Debug("scored entity", slog.Float64("score", float64(score)), slog.String("content", retrievedAsset.Content))
		scoredEnities = append(scoredEnities, retrievedAsset)
	}
	slices.SortFunc(scoredEnities, func(a, b asset_manager.RetrievedAsset) int {
		if a.Score == nil && b.Score == nil {
			return 0
		}
		if a.Score == nil {
			return -1
		}
		if b.Score == nil {
			return 1
		}
		return cmp.Compare(*b.Score, *a.Score)
	})
	logger.InfoContext(
		ctx,
		"scored entities for kbId successfully",
		slog.String("kbId", kbId),
		slog.Int("entities", len(scoredEnities)),
		slog.Float64("maxScore", float64(*scoredEnities[0].Score)),
		slog.Float64("minScore", float64(*scoredEnities[len(scoredEnities)-1].Score)),
	)
	return scoredEnities, nil
}

func (g *GraphRAGServiceImpl) Retrieve(ctx context.Context, query string, kbIds []string, seedAssets []asset_manager.ContextualAsset, config *map[string]any) ([]asset_manager.RetrievedAsset, []asset_manager.ContextualAsset, error) {
	logger := asset_manager.GetLogger(g)
	logger.InfoContext(ctx, "starting to retrieve for kbIds", slog.String("query", query), slog.Any("kbIds", kbIds), slog.Any("config", config))
	if len(kbIds) == 0 {
		msg := "no kbIds provided"
		logger.ErrorContext(ctx, msg)
		return []asset_manager.RetrievedAsset{}, []asset_manager.ContextualAsset{}, errors.New(msg)
	}
	if len(query) == 0 {
		msg := "no query provided"
		logger.ErrorContext(ctx, msg, slog.Any("kbIds", kbIds), slog.Any("config", config))
		return []asset_manager.RetrievedAsset{}, []asset_manager.ContextualAsset{}, errors.New(msg)
	}
	if len(seedAssets) == 0 {
		msg := "no seedAssets provided"
		logger.ErrorContext(ctx, msg, slog.String("query", query), slog.Any("kbIds", kbIds), slog.Any("config", config))
		return []asset_manager.RetrievedAsset{}, []asset_manager.ContextualAsset{}, errors.New(msg)
	}
	runOpts := NewGraphRAGServiceRuntimeOptionsFromConfig(config, &g.GraphRAGServiceRuntimeOptions)

	// 1. Extract Entities from Query
	logger.InfoContext(ctx, "extracting entities from query", slog.String("query", query), slog.Any("kbIds", kbIds))
	extractedQEntities, err := g.GraphExtractor.ExtractEntitiesFromQuery(ctx, query)
	if err != nil {
		logger.ErrorContext(ctx, err.Error(), slog.String("query", query), slog.Any("kbIds", kbIds))
		return []asset_manager.RetrievedAsset{}, []asset_manager.ContextualAsset{}, err
	}
	logger.InfoContext(
		ctx, "extracted entities from query successfully",
		slog.String("query", query),
		slog.Any("kbIds", kbIds),
		slog.Any("namedEntities", extractedQEntities.NamedEntities),
		slog.Any("genericEntities", extractedQEntities.GenericEntities),
	)
	// These extracted entities should be already normalized
	queryEntities := []models.QueryEntity{
		{
			Name: query,
			Type: models.QueryEntityType,
		},
	}
	for _, e := range extractedQEntities.NamedEntities {
		queryEntities = append(queryEntities, models.QueryEntity{
			Name: e,
			Type: models.NamedEntityType,
		})
	}
	for _, e := range extractedQEntities.GenericEntities {
		queryEntities = append(queryEntities, models.QueryEntity{
			Name: e,
			Type: models.GenericEntityType,
		})
	}

	var wg sync.WaitGroup
	var mu sync.Mutex

	embeddedQueryEntities := []models.QueryEntity{}
	// 2. Generate Embeddings for Query Entities
	embedSem := make(chan struct{}, g.MaxConcurrent)
	errChan := make(chan error, len(queryEntities))
	for _, qe := range queryEntities {
		wg.Add(1)
		embedSem <- struct{}{}
		go func(qe *models.QueryEntity) {
			defer wg.Done()
			defer func() { <-embedSem }()
			vector, err := g.VectorStore.EmbedContent(ctx, qe.Name)
			if err != nil {
				if qe.Type != models.QueryEntityType {
					logger.Warn("failed to embed entity extracted from query, skipping this entity: "+err.Error(), slog.String("entity", qe.Name), slog.String("entityType", qe.Type))
					return
				}
				errChan <- err
			}
			mu.Lock()
			qe.Embedding = vector
			logger.Debug("set embedding for query entity", slog.String("entity", qe.Name), slog.String("entityType", qe.Type), slog.Int("embeddingDim", len(qe.Embedding)))
			embeddedQueryEntities = append(embeddedQueryEntities, *qe)
			mu.Unlock()
		}(&qe)
	}
	wg.Wait()
	if len(errChan) > 0 {
		var err error
		for e := range errChan {
			err = fmt.Errorf("%w\n%s", err, e.Error())
		}
		return []asset_manager.RetrievedAsset{}, []asset_manager.ContextualAsset{}, err
	}

	if len(embeddedQueryEntities) == 0 {
		msg := "no embedded query entities found"
		logger.ErrorContext(ctx, msg, slog.String("query", query), slog.Any("kbIds", kbIds))
		return []asset_manager.RetrievedAsset{}, []asset_manager.ContextualAsset{}, errors.New(msg)
	}

	sem := make(chan struct{}, min(max(g.MaxConcurrent, 1), len(kbIds)))
	retrievedAssets := []asset_manager.RetrievedAsset{}
	for _, kbId := range kbIds {
		wg.Add(1)
		sem <- struct{}{}
		go func(kbId string) {
			defer wg.Done()
			defer func() { <-sem }()
			assets, err := g.retrieveForKbId(ctx, embeddedQueryEntities, kbId, seedAssets, config)
			if err != nil {
				msg := "failed to retrieve for kbId, skipping this kbId, cause: " + err.Error()
				logger.ErrorContext(ctx, msg, slog.String("query", query), slog.Any("kbId", kbId), slog.Any("config", config))
				return
			}
			mu.Lock()
			retrievedAssets = append(retrievedAssets, assets...)
			mu.Unlock()
		}(kbId)
	}
	wg.Wait()
	if len(retrievedAssets) == 0 {
		msg := "no assets found for kbIds: " + strings.Join(kbIds, ", ")
		logger.ErrorContext(ctx, msg, slog.String("query", query), slog.Any("kbIds", kbIds), slog.Any("config", config))
		return []asset_manager.RetrievedAsset{}, []asset_manager.ContextualAsset{}, errors.New(msg)
	}
	// filter by threshold
	filteredEntityAssets := make([]asset_manager.RetrievedAsset, 0, len(retrievedAssets))
	for _, asset := range retrievedAssets {
		if asset.Score != nil && *asset.Score >= 0 {
			filteredEntityAssets = append(filteredEntityAssets, asset)
		}
	}
	if len(filteredEntityAssets) == 0 {
		msg := "no retrieved assets found for kbIds: " + strings.Join(kbIds, ", ") + ", may be threshold is too high"
		logger.WarnContext(ctx, msg, slog.String("query", query), slog.Any("kbIds", kbIds), slog.Float64("threshold", float64(*runOpts.Threshold)))
		return []asset_manager.RetrievedAsset{}, []asset_manager.ContextualAsset{}, nil
	}
	// sort by score
	slices.SortFunc(filteredEntityAssets, func(a, b asset_manager.RetrievedAsset) int {
		if a.Score == nil && b.Score == nil {
			return 0
		}
		if a.Score == nil {
			return -1
		}
		if b.Score == nil {
			return 1
		}
		return cmp.Compare(*b.Score, *a.Score)
	})
	logger.DebugContext(ctx, "filtered entity assets", slog.String("query", query), slog.Any("kbIds", kbIds), slog.Int("filteredEntityAssets", len(filteredEntityAssets)))
	// take topK assets
	topKAssets := make([]asset_manager.RetrievedAsset, 0, *runOpts.TopK)
	for i := 0; i < *runOpts.TopK && i < len(filteredEntityAssets); i++ {
		topKAssets = append(topKAssets, filteredEntityAssets[i])
	}
	logger.DebugContext(ctx, "trucated topK assets", slog.String("query", query), slog.Any("kbIds", kbIds), slog.Int("topKAssets", len(topKAssets)))

	kbIdMapRefEntities := make(map[string][]asset_manager.RetrievedAsset)
	for _, asset := range topKAssets {
		kbIdMapRefEntities[asset.KbId] = append(kbIdMapRefEntities[asset.KbId], asset)
	}

	logger.DebugContext(ctx, "propagating assets", slog.String("query", query), slog.Any("kbIds", kbIds))
	propagatedAssets := []asset_manager.ContextualAsset{}
	sem = make(chan struct{}, min(max(g.MaxConcurrent, 1), len(kbIdMapRefEntities)))
	for kbId, refAssets := range kbIdMapRefEntities {
		wg.Add(1)
		sem <- struct{}{}
		go func(kbId string, refAssets []asset_manager.RetrievedAsset) {
			defer wg.Done()
			defer func() { <-sem }()
			version := refAssets[0].Version

			refEntityIds := make([]string, 0, len(refAssets))
			for _, asset := range refAssets {
				refEntityIds = append(refEntityIds, asset.AssetId)
			}
			logger.DebugContext(ctx, "propagating relation assets for kbId by source and target", slog.String("kbId", kbId), slog.Int("refIds", len(refEntityIds)))
			relatedAssets, err := g.AssetStore.GetAssetsByRefs(
				ctx,
				kbId,
				asset_manager.AssetTypeRelation,
				&refEntityIds,
				&[]string{asset_manager.AssetRefTypeSource, asset_manager.AssetRefTypeTarget},
				&version,
				runOpts.Label,
			)
			if err == nil {
				logger.DebugContext(ctx, "propagated relation assets for kbId", slog.String("kbId", kbId), slog.Int("relatedAssets", len(relatedAssets)))
				mu.Lock()
				propagatedAssets = append(propagatedAssets, relatedAssets...)
				mu.Unlock()
			} else {
				// Skip this kbId if err, assume something wrong with this kbId but service should keep running
				// TODO: Logging
				msg := "failed to propagate relation assets for kbId, skipping this kbId, cause: " + err.Error()
				logger.ErrorContext(ctx, msg, slog.String("kbId", kbId), slog.Any("refIds", refEntityIds))
			}

			// propagate chunk assets
			chunkIDsSet := make(map[string]struct{})
			chunkIds := make([]string, 0, len(refAssets))
			for _, asset := range refAssets {
				if asset.Refs != nil {
					for _, ref := range *asset.Refs {
						if ref.AssetType == asset_manager.AssetTypeChunk {
							if _, exists := chunkIDsSet[ref.AssetId]; !exists {
								chunkIDsSet[ref.AssetId] = struct{}{}
								chunkIds = append(chunkIds, ref.AssetId)
							}
						}
					}
				}
			}
			for _, asset := range relatedAssets {
				if asset.Refs != nil {
					for _, ref := range *asset.Refs {
						if ref.AssetType == asset_manager.AssetTypeChunk {
							if _, exists := chunkIDsSet[ref.AssetId]; !exists {
								chunkIDsSet[ref.AssetId] = struct{}{}
								chunkIds = append(chunkIds, ref.AssetId)
							}
						}
					}
				}
			}
			if len(chunkIds) == 0 {
				msg := "no chunk ids found for kbId: " + kbId
				logger.WarnContext(ctx, msg, slog.String("kbId", kbId), slog.Any("refAssets", refAssets))
			} else {
				logger.DebugContext(ctx, "propagating chunk assets for kbId", slog.String("kbId", kbId), slog.Int("chunkIds", len(chunkIds)))
				chunkAssets, err := g.AssetStore.GetAssets(
					ctx,
					kbId,
					asset_manager.AssetTypeChunk,
					&chunkIds,
					&version,
					runOpts.Label,
				)
				if err == nil {
					logger.DebugContext(ctx, "propagated chunk assets for kbId", slog.String("kbId", kbId), slog.Int("chunkAssets", len(chunkAssets)))
					mu.Lock()
					propagatedAssets = append(propagatedAssets, chunkAssets...)
					mu.Unlock()
				} else {
					msg := "failed to propagate chunk assets for kbId, skipping this kbId, cause: " + err.Error()
					logger.ErrorContext(ctx, msg, slog.String("kbId", kbId), slog.Any("chunkIds", len(chunkIds)))
				}
			}

		}(kbId, refAssets)
	}
	wg.Wait()

	logger.InfoContext(ctx, "retrieved assets successfully", slog.String("query", query), slog.Any("kbIds", kbIds), slog.Int("retrievedAssets", len(topKAssets)), slog.Int("propagatedAssets", len(propagatedAssets)))
	return topKAssets, propagatedAssets, nil
}
