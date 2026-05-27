package graph

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"

	"github.com/manageai-inet/fast-graphrago/llm"
	"github.com/manageai-inet/fast-graphrago/models"
	"github.com/manageai-inet/fast-graphrago/prompts"
	"github.com/manageai-inet/fast-graphrago/utils"

	"github.com/openai/openai-go"

	asset_manager "github.com/manageai-inet/agentic-assets"
)

type FastGraphExtractor struct {
	LLM               llm.LLM
	MaxRetryAttempt   int
	MaxConcurrent     int
	BatchSize         int
	extractionTool    openai.ChatCompletionToolParam
	deduplicationTool openai.ChatCompletionToolParam
	queryTool         openai.ChatCompletionToolParam
	asset_manager.LoggingCapacity
}

func NewFastGraphExtractor(llm llm.LLM) *FastGraphExtractor {
	extractionTool, err := models.GetGraphExtractionTool()
	if err != nil {
		panic("failed to build extraction tool schema: " + err.Error())
	}
	deduplicationTool, err := models.GetRelationClusterTool()
	if err != nil {
		panic("failed to build deduplication tool schema: " + err.Error())
	}
	queryTool, err := models.GetQueryExtractionTool()
	if err != nil {
		panic("failed to build query tool schema: " + err.Error())
	}
	return &FastGraphExtractor{
		LLM:               llm,
		MaxRetryAttempt:   utils.DefaultMaxRetryAttempt,
		MaxConcurrent:     utils.DefaultMaxConcurrent,
		BatchSize:         utils.DefaultBatchSize,
		extractionTool:    extractionTool,
		deduplicationTool: deduplicationTool,
		queryTool:         queryTool,
		LoggingCapacity:   *asset_manager.GetDefaultLoggingCapacity(),
	}
}

func (f *FastGraphExtractor) SetLLM(llm llm.LLM) {
	f.LLM = llm
}

func (f *FastGraphExtractor) SetBatchSize(batchSize int) {
	f.BatchSize = batchSize
}

// Extract Graph From given batch of chunks, including deduplication and merge processes for entities and relations
func (f *FastGraphExtractor) ExtractGraph(ctx context.Context, chunks []asset_manager.ContextualAsset, options GraphExtractionOptions) (models.GraphAssets, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var mu sync.Mutex
	var wg sync.WaitGroup

	logger := asset_manager.GetLogger(f)
	logger.InfoContext(ctx, "starting to extract graph", slog.Int("chunks", len(chunks)), slog.Any("options", options))

	entityAssetsGroup := make(map[string]models.EntityAsset)
	relationAssetsGroup := make(map[string][]models.RelationAsset)

	batchSize := f.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}
	batches := make([][]asset_manager.ContextualAsset, 0, (len(chunks)+batchSize-1)/batchSize)
	for i := 0; i < len(chunks); i += batchSize {
		end := min(i+batchSize, len(chunks))
		batches = append(batches, chunks[i:end])
	}

	errChan := make(chan error, len(batches))
	sem := make(chan struct{}, f.MaxConcurrent)
	for _, batch := range batches {
		wg.Add(1)
		sem <- struct{}{}
		go func(batch []asset_manager.ContextualAsset) {
			defer wg.Done()
			defer func() { <-sem }()

			maxAttempt := max(f.MaxRetryAttempt, 1)
			var graph models.GraphAssets
			var err error
			for i := 0; i < maxAttempt; i++ {
				logger.DebugContext(ctx, "extracting graph from batch", slog.Int("batchSize", len(batch)), slog.Int("attempt", i+1))
				graph, err = f.ExtractGraphFromChunks(ctx, batch, options)
				if err == nil {
					break
				}
				logger.Warn("extracting graph from batch failed: "+err.Error(), slog.Int("batchSize", len(batch)), slog.Int("attempt", i+1))
			}
			if err != nil {
				errChan <- err
				return
			}
			mu.Lock()
			for _, e := range graph.EntityAssets {
				entityKey := e.Name + "|" + e.Type
				if _, ok := entityAssetsGroup[entityKey]; !ok {
					entityAssetsGroup[entityKey] = e
				} else {
					logger.Warn("found duplicate entity, merging", slog.String("entity", e.Name), slog.String("type", e.Type))
					existing := entityAssetsGroup[entityKey]
					existing.Description = existing.Description + "\n" + e.Description
					existing.ChunkIds = append(existing.ChunkIds, e.ChunkIds...)
					entityAssetsGroup[entityKey] = existing
				}
			}
			for _, r := range graph.RelationAssets {
				keys := []string{r.Source + "|" + r.SourceType, r.Target + "|" + r.TargetType}
				slices.Sort(keys)
				relationKey := strings.Join(keys, "|")
				relationAssetsGroup[relationKey] = append(relationAssetsGroup[relationKey], r)
			}
			mu.Unlock()
		}(batch)
	}
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case err := <-errChan:
		cancel()
		logger.ErrorContext(ctx, "extracting graph from batch failed: "+err.Error())
		return models.GraphAssets{}, err
	case <-done:
	}

	entityKeyToId := make(map[string]string)
	entityAssets := []models.EntityAsset{}
	for _, entity := range entityAssetsGroup {
		entityKeyToId[entity.Name+"|"+entity.Type] = entity.Name
		entityAssets = append(entityAssets, entity)
	}

	logger.DebugContext(ctx, "deduping relations", slog.Int("relations", len(relationAssetsGroup)))
	relationAssets := []models.RelationAsset{}
	var dedupWg sync.WaitGroup
	dedupSem := make(chan struct{}, f.MaxConcurrent)
	dedupErrChan := make(chan error, len(relationAssetsGroup))
	for rk, relationWithSameKey := range relationAssetsGroup {
		dedupWg.Add(1)
		dedupSem <- struct{}{}
		go func(relGroup []models.RelationAsset) {
			defer dedupWg.Done()
			defer func() { <-dedupSem }()

			if len(relGroup) > 1 {
				var newRelations []models.RelationAsset
				var err error
				for i := 0; i < f.MaxRetryAttempt; i++ {
					logger.Debug("deduping relations in a group", slog.String("relationGroupKey", rk), slog.Int("relations", len(relGroup)), slog.Int("attempt", i+1))
					newRelations, err = f.DeduplicateRelations(ctx, relGroup)
					if err == nil {
						isFailed := false
						for _, relation := range newRelations {
							sourceKey := relation.Source + "|" + relation.SourceType
							targetKey := relation.Target + "|" + relation.TargetType
							if _, ok := entityKeyToId[sourceKey]; !ok {
								if i == f.MaxRetryAttempt-1 {
									err = fmt.Errorf("relation bind to source entity %s which is not found", sourceKey)
								}
								isFailed = true
							}
							if _, ok := entityKeyToId[targetKey]; !ok {
								if i == f.MaxRetryAttempt-1 {
									err = fmt.Errorf("relation bind to target entity %s which is not found", targetKey)
								}
								isFailed = true
							}
						}
						if !isFailed {
							break
						} else {
							logger.Warn("deduping relations failed", slog.String("error", err.Error()))
						}
					} else {
						logger.Warn("deduping relations failed", slog.String("error", err.Error()))
					}
				}
				if err != nil {
					dedupErrChan <- err
					return
				}
				mu.Lock()
				relationAssets = append(relationAssets, newRelations...)
				mu.Unlock()
			} else {
				mu.Lock()
				relationAssets = append(relationAssets, relGroup[0])
				mu.Unlock()
			}
		}(relationWithSameKey)
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

	logger.InfoContext(ctx, "deduping relations finished", slog.Int("entities", len(entityAssets)), slog.Int("relations", len(relationAssets)))
	return models.GraphAssets{
		EntityAssets:   entityAssets,
		RelationAssets: relationAssets,
	}, nil
}

func (f *FastGraphExtractor) DeduplicateRelations(ctx context.Context, relations []models.RelationAsset) ([]models.RelationAsset, error) {
	relationClusterTool := f.deduplicationTool
	inputCSV := ""
	for i, relation := range relations {
		inputCSV += fmt.Sprintf(
			"%d,%s,%s,%s,%s,%s\n",
			i,
			relation.Source,
			relation.SourceType,
			relation.Target,
			relation.TargetType,
			relation.Description,
		)
	}
	systemPrompt := prompts.DeduplicateRelationsSystem
	relationClusterPrompt := fmt.Sprintf(prompts.DeduplicateRelationsPrompt, inputCSV)
	messages := []openai.ChatCompletionMessage{
		{
			Role:    "system",
			Content: systemPrompt,
		},
		{
			Role:    "user",
			Content: relationClusterPrompt,
		},
	}
	toolChoice := openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("required"),}
	generation, err := f.LLM.Generate(ctx, messages, []openai.ChatCompletionToolParam{relationClusterTool}, &toolChoice)
	if err != nil {
		return []models.RelationAsset{}, err
	}
	if len(generation.ToolCalls) == 0 {
		return []models.RelationAsset{}, fmt.Errorf("no tool calls found")
	}
	args := generation.ToolCalls[0].Function.Arguments
	relationClusters, err := utils.HandleToolCall[models.RelationClusters](args)
	if err != nil {
		return []models.RelationAsset{}, err
	}
	newRelations := []models.RelationAsset{}
	for _, relationCluster := range relationClusters.Clusters {
		chunkIds := []string{}
		for _, i := range relationCluster.Indices {
			if i >= len(relations) {
				return []models.RelationAsset{}, fmt.Errorf("invalid relation index: %d", i)
			}
			r := relations[i]
			chunkIds = append(chunkIds, r.ChunkIds...)
		}
		newRelations = append(newRelations, models.RelationAsset{
			Source:      utils.NormalizeName(relationCluster.Source.Name),
			SourceType:  utils.NormalizeName(relationCluster.Source.Type),
			Target:      utils.NormalizeName(relationCluster.Target.Name),
			TargetType:  utils.NormalizeName(relationCluster.Target.Type),
			Description: relationCluster.Desc,
			ChunkIds:    chunkIds,
		})
	}
	return newRelations, nil
}

// extractEntityContext searches for entityName in chunkContent and returns a
// surrounding text window as a best-effort description. Returns "" if not found.
// Uses rune-safe slicing so multi-byte characters (e.g. Thai) are never split.
func extractEntityContext(chunkContent, entityName string) string {
	byteIdx := strings.Index(strings.ToLower(chunkContent), entityName)
	if byteIdx < 0 {
		return ""
	}
	runes := []rune(chunkContent)
	runeIdx := len([]rune(chunkContent[:byteIdx]))
	entityRuneLen := len([]rune(entityName))
	const window = 80
	start := max(0, runeIdx-window)
	end := min(len(runes), runeIdx+entityRuneLen+window)
	return strings.TrimSpace(string(runes[start:end]))
}

func (f *FastGraphExtractor) inferMissingEntity(ctx context.Context, logger interface{ WarnContext(context.Context, string, ...any) }, name, typ string, chunkIdx int, chunks []asset_manager.ContextualAsset) models.EntityAsset {
	desc := extractEntityContext(chunks[chunkIdx].Content, name)
	logger.WarnContext(ctx, "relation references undeclared entity, inferring from chunk",
		slog.String("entity", name),
		slog.String("type", typ),
		slog.Bool("desc_found", desc != ""))
	return models.EntityAsset{
		Name:        name,
		Type:        typ,
		Description: desc,
		ChunkIds:    []string{chunks[chunkIdx].AssetId},
	}
}

func (f *FastGraphExtractor) ExtractGraphFromChunks(ctx context.Context, chunks []asset_manager.ContextualAsset, options GraphExtractionOptions) (models.GraphAssets, error) {
	logger := asset_manager.GetLogger(f)
	var sb strings.Builder
	for i, ch := range chunks {
		fmt.Fprintf(&sb, "[chunk_%d]\n%s\n\n", i, ch.Content)
	}
	chunksContent := strings.TrimRight(sb.String(), "\n")

	systemPrompt := fmt.Sprintf(prompts.EntityRelationshipExtractionSystem, options.Domain)
	entityTypes := strings.Join(options.EntityTypes, ", ")
	extractionPrompt := fmt.Sprintf(prompts.EntityRelationshipExtractionPrompt, entityTypes, len(chunks), chunksContent)

	messages := []openai.ChatCompletionMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: extractionPrompt},
	}
	toolChoice := openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("required")}

	generation, err := f.LLM.Generate(ctx, messages, []openai.ChatCompletionToolParam{f.extractionTool}, &toolChoice)
	if err != nil {
		return models.GraphAssets{}, err
	}
	if len(generation.ToolCalls) == 0 {
		return models.GraphAssets{}, fmt.Errorf("no tool calls found")
	}

	graph, err := utils.HandleToolCall[models.ExtractedGraph](generation.ToolCalls[0].Function.Arguments)
	if err != nil {
		return models.GraphAssets{}, err
	}

	entityKeyToId := make(map[string]string)
	entityAssets := make([]models.EntityAsset, 0, len(graph.Entities))
	relationAssets := make([]models.RelationAsset, 0, len(graph.Relations))

	for _, entity := range graph.Entities {
		entity.Name = utils.NormalizeName(entity.Name)
		entity.Type = utils.NormalizeName(entity.Type)
		if entity.ChunkIndex < 0 || entity.ChunkIndex >= len(chunks) {
			return models.GraphAssets{}, fmt.Errorf("entity %q has invalid chunk_index %d (batch size %d)", entity.Name, entity.ChunkIndex, len(chunks))
		}
		entityAssets = append(entityAssets, models.EntityAsset{
			Name:        entity.Name,
			Type:        entity.Type,
			Description: entity.Desc,
			ChunkIds:    []string{chunks[entity.ChunkIndex].AssetId},
		})
		entityKeyToId[entity.Name+"|"+entity.Type] = entity.Name
	}

	for _, relation := range graph.Relations {
		relation.Source.Name = utils.NormalizeName(relation.Source.Name)
		relation.Source.Type = utils.NormalizeName(relation.Source.Type)
		relation.Target.Name = utils.NormalizeName(relation.Target.Name)
		relation.Target.Type = utils.NormalizeName(relation.Target.Type)
		if relation.ChunkIndex < 0 || relation.ChunkIndex >= len(chunks) {
			return models.GraphAssets{}, fmt.Errorf("relation %q→%q has invalid chunk_index %d (batch size %d)", relation.Source.Name, relation.Target.Name, relation.ChunkIndex, len(chunks))
		}
		if _, ok := entityKeyToId[relation.Source.Name+"|"+relation.Source.Type]; !ok {
			inferred := f.inferMissingEntity(ctx, logger, relation.Source.Name, relation.Source.Type, relation.ChunkIndex, chunks)
			entityAssets = append(entityAssets, inferred)
			entityKeyToId[inferred.Name+"|"+inferred.Type] = inferred.Name
		}
		if _, ok := entityKeyToId[relation.Target.Name+"|"+relation.Target.Type]; !ok {
			inferred := f.inferMissingEntity(ctx, logger, relation.Target.Name, relation.Target.Type, relation.ChunkIndex, chunks)
			entityAssets = append(entityAssets, inferred)
			entityKeyToId[inferred.Name+"|"+inferred.Type] = inferred.Name
		}
		relationAssets = append(relationAssets, models.RelationAsset{
			Source:      relation.Source.Name,
			SourceType:  relation.Source.Type,
			Target:      relation.Target.Name,
			TargetType:  relation.Target.Type,
			Description: relation.Desc,
			ChunkIds:    []string{chunks[relation.ChunkIndex].AssetId},
		})
	}

	return models.GraphAssets{EntityAssets: entityAssets, RelationAssets: relationAssets}, nil
}

func (f *FastGraphExtractor) ExtractGraphFromChunk(ctx context.Context, chunk asset_manager.ContextualAsset, options GraphExtractionOptions) (models.GraphAssets, error) {
	return f.ExtractGraphFromChunks(ctx, []asset_manager.ContextualAsset{chunk}, options)
}

func (f *FastGraphExtractor) ExtractEntitiesFromQuery(ctx context.Context, query string) (models.ExtractedQuery, error) {
	logger := asset_manager.GetLogger(f)
	logger.InfoContext(ctx, "extracting entities from query", slog.String("query", query))
	queryExtractionTool := f.queryTool
	entityExtractionPrompt := fmt.Sprintf(prompts.EntityExtractionQuery, query)

	messages := []openai.ChatCompletionMessage{
		{
			Role:    "user",
			Content: entityExtractionPrompt,
		},
	}
	maxAttempt := max(f.MaxRetryAttempt, 1)
	var generation *openai.ChatCompletionMessage
	var err error
	var generationErr error
	var extracted *models.ExtractedQuery
	for i := 0; i < maxAttempt; i++ {
		logger.Debug("extracting entities from query", slog.String("query", query), slog.Int("attempt", i+1))
		toolChoice := openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("required"),}
		generation, err = f.LLM.Generate(ctx, messages, []openai.ChatCompletionToolParam{queryExtractionTool}, &toolChoice)
		if err != nil {
			logger.ErrorContext(ctx, "failed to extract entities from query", slog.String("query", query), slog.Int("attempt", i+1), slog.String("error", err.Error()))
			return models.ExtractedQuery{}, err
		}
		if len(generation.ToolCalls) > 0 {
			args := generation.ToolCalls[0].Function.Arguments
			extracted, generationErr = utils.HandleToolCall[models.ExtractedQuery](args)
			if generationErr == nil {
				for i, e := range extracted.NamedEntities {
					extracted.NamedEntities[i] = utils.NormalizeName(e)
				}
				for i, e := range extracted.GenericEntities {
					extracted.GenericEntities[i] = utils.NormalizeName(e)
				}
				logger.InfoContext(ctx, "extracted entities from query successfully", slog.String("query", query), slog.Int("attempt", i+1), slog.Any("extracted", extracted))
				return *extracted, nil
			}
			logger.WarnContext(ctx, "failed to extract entities from query: "+generationErr.Error(), slog.String("query", query), slog.Int("attempt", i+1))
		} else {
			logger.WarnContext(ctx, "failed to extract entities from query: llm did not return any tool calls", slog.String("query", query), slog.Int("attempt", i+1))
		}
	}
	if generationErr != nil {
		logger.ErrorContext(ctx, "failed to extract entities from query: "+generationErr.Error(), slog.String("query", query))
		return models.ExtractedQuery{}, generationErr
	} else {
		logger.ErrorContext(ctx, "failed to extract entities from query because unknown reason", slog.String("query", query))
		return models.ExtractedQuery{}, fmt.Errorf("failed to extract entities from query because unknown reason")
	}
}
