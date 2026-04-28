package main

import (
	"context"
	"errors"
	"fastgraph-example/models"
	"fmt"
	"log/slog"

	am "github.com/manageai-inet/agentic-assets"
	amk "github.com/manageai-inet/agentic-assets/knowledge_extractor"

	"github.com/manageai-inet/fast-graphrago"
)

type FastGraphService struct {
	// key is file type
	extractors map[string]amk.KnowledgeExtractor
	indexer    fastgraphrag.FastGraphIndexer
	am.LoggingCapacity
}

func NewFastGraphService(
	extractors map[string]amk.KnowledgeExtractor,
	indexer fastgraphrag.FastGraphIndexer,
) *FastGraphService {
	return &FastGraphService{
		extractors: extractors,
		indexer:    indexer,
		LoggingCapacity: *am.GetDefaultLoggingCapacity(),
	}
}

func (f *FastGraphService) String() string {
	return "FastGraphService"
}

func (f *FastGraphService) Extract(
	ctx context.Context, 
	sourceFile models.IndexingFile,
	) ([]am.KnowledgeSource, error) {
	logger := am.GetLogger(f)
	logger.InfoContext(ctx, "Extracting content from file", slog.String("filename", sourceFile.FileName), slog.String("fileType", sourceFile.FileType))
	extractor, ok := f.extractors[sourceFile.FileType]
	if !ok {
		msg := fmt.Sprintf("unsupported file type: %s", sourceFile.FileType)
		logger.ErrorContext(ctx, msg, slog.String("filename", sourceFile.FileName))
		return nil, errors.New(msg)
	}
	if sourceFile.FileUrl == nil {
		msg := "file url is required for extraction"
		logger.ErrorContext(ctx, msg, slog.String("filename", sourceFile.FileName))
		return nil, errors.New(msg)
	}
	return extractor.Extract(ctx, "", sourceFile.FileName, *sourceFile.FileUrl, nil)
}

// config accept: domain string, entity_types []string
func (f *FastGraphService) Index(
	ctx context.Context, 
	sourceFile models.IndexingFile,
	kbId string, 
	labels *[]string, 
	metadata *map[string]any, 
	config *map[string]any,
	) (am.IndexingResult, error) {
	logger := am.GetLogger(f)
	logger.InfoContext(
		ctx, "Indexing file", 
		slog.String("kbId", kbId), 
		slog.String("filename", sourceFile.FileName),
		slog.String("fileType", sourceFile.FileType),
		slog.Any("labels", labels),
		slog.Any("metadata", metadata),
	)
	extractor, ok := f.extractors[sourceFile.FileType]
	if !ok {
		msg := fmt.Sprintf("unsupported file type: %s", sourceFile.FileType)
		logger.ErrorContext(ctx, msg, slog.String("kbId", kbId), slog.String("filename", sourceFile.FileName))
		return am.IndexingResult{}, errors.New(msg)
	}
	var sources []am.KnowledgeSource
	var err error
	if len(sourceFile.FileContent) == 0 {
		if sourceFile.FileUrl == nil {
			msg := "file url is required when file content is empty"
			logger.ErrorContext(ctx, msg, slog.String("kbId", kbId), slog.String("filename", sourceFile.FileName))
			return am.IndexingResult{}, errors.New(msg)
		}
		logger.InfoContext(ctx, "Extracting content from file", slog.String("kbId", kbId), slog.String("filename", sourceFile.FileName), slog.String("fileUrl", *sourceFile.FileUrl))
		sources, err = extractor.Extract(ctx, kbId, sourceFile.FileName, *sourceFile.FileUrl, metadata)
		if err != nil {
			return am.IndexingResult{}, err
		}
	} else {
		logger.InfoContext(ctx, "Using provided file content", slog.String("kbId", kbId), slog.String("filename", sourceFile.FileName), slog.Int("contentCount", len(sourceFile.FileContent)))
		for i, content := range sourceFile.FileContent {
			sources = append(sources, am.KnowledgeSource{
				SourceType: am.AssetTypePage,
				SourceName: fmt.Sprintf("%s:%s:%d", kbId, sourceFile.FileName, i+1),
				SourceContents: &content,
				SourceUrl: sourceFile.FileUrl,
				Metadata: metadata,
			})
		}
	}
	logger.InfoContext(ctx, "Indexing file", slog.String("kbId", kbId), slog.String("filename", sourceFile.FileName), slog.Int("sources", len(sources)))
	indexingResult, err := f.indexer.Index(ctx, kbId, sources, labels, metadata, config)
	if err != nil {
		logger.ErrorContext(ctx, "Failed to index file", slog.String("kbId", kbId), slog.String("filename", sourceFile.FileName), slog.String("error", err.Error()))
		return indexingResult, err
	}
	logger.InfoContext(ctx, "Indexing file complete", slog.String("kbId", kbId), slog.String("filename", sourceFile.FileName), slog.Any("indexingResult", indexingResult))
	return indexingResult, nil
}

// config accept: damping_factor float32, query_weights [3]float32, top_k int, threshold float32, version *int, label *string
func (f *FastGraphService) Retrieve(ctx context.Context, query string, kbIds []string, config *map[string]any) (am.RetrieveResult, error) {
	logger := am.GetLogger(f)
	logger.InfoContext(ctx, "Retrieving", slog.String("query", query), slog.Any("kbIds", kbIds))
	var version *int
	var label *string
	if config != nil {
		if v, ok := (*config)["version"]; ok {
			version = v.(*int)
		}
		if v, ok := (*config)["label"]; ok {
			label = v.(*string)
		}
	}
	seedAssets := f.indexer.GetRetrievalSeedAssets(ctx, query, kbIds, version, label)
	retrieveResult, err := f.indexer.Retrieve(ctx, query, kbIds, seedAssets, config)
	if err != nil {
		logger.ErrorContext(ctx, "Failed to retrieve", slog.String("query", query), slog.Any("kbIds", kbIds), slog.String("error", err.Error()))
		return retrieveResult, err
	}
	logger.InfoContext(ctx, "Retrieve complete", slog.String("query", query), slog.Any("kbIds", kbIds), slog.Any("retrieveResult", retrieveResult))
	return retrieveResult, nil
}

func (a *FastGraphService) GetAssetsByKbId(ctx context.Context, kbId string) ([]am.ContextualAsset, error) {
	logger := am.GetLogger(a)
	logger.InfoContext(ctx, "GetAssetsByKbId", slog.String("kbId", kbId))
	return a.indexer.Assets.GetAssetsByKbId(ctx, kbId)
}

func (a *FastGraphService) GetVersions(ctx context.Context, kbId string) ([]int, error) {
	logger := am.GetLogger(a)
	logger.InfoContext(ctx, "GetVersions", slog.String("kbId", kbId))
	return a.indexer.Assets.GetVersions(ctx, kbId)
}

func (a *FastGraphService) GetAssets(ctx context.Context, kbId string, assetType string, assetIds []string, version *int, label *string) ([]am.ContextualAsset, error) {
	logger := am.GetLogger(a)
	logargs := []any{
		slog.String("kbId", kbId),
		slog.String("assetType", assetType),
	}
	if len(assetIds) == 0 {
		logargs = append(logargs, slog.String("assetIds", "*"))
	} else {
		logargs = append(logargs, slog.String("assetIds", fmt.Sprintf("[%v ... %v]", assetIds[0], assetIds[len(assetIds)-1])))
	}
	if version != nil {
		logargs = append(logargs, slog.Int("version", *version))
	} else {
		logargs = append(logargs, slog.String("version", "latest"))
	}
	if label != nil {
		logargs = append(logargs, slog.String("label", *label))
	} else {
		logargs = append(logargs, slog.String("label", "*"))
	}
	logger.InfoContext(ctx, "GetAssets", logargs...)
	
	return a.indexer.Assets.GetAssets(ctx, kbId, assetType, &assetIds, version, label)
}

func (a *FastGraphService) GetAssetsByRefs(ctx context.Context, kbId string, assetType string, refIds *[]string, refTypes *[]string, version *int, label *string) ([]am.ContextualAsset, error) {
	logger := am.GetLogger(a)
	logargs := []any{
		slog.String("kbId", kbId),
		slog.String("assetType", assetType),
	}
	if refIds == nil || len(*refIds) == 0 {
		logargs = append(logargs, slog.String("refIds", "*"))
	} else {
		logargs = append(logargs, slog.String("refIds", fmt.Sprintf("[%v ... %v]", (*refIds)[0], (*refIds)[len(*refIds)-1])))
	}
	if refTypes == nil || len(*refTypes) == 0 {
		logargs = append(logargs, slog.String("refTypes", "*"))
	} else {
		logargs = append(logargs, slog.String("refTypes", fmt.Sprintf("[%v ... %v]", (*refTypes)[0], (*refTypes)[len(*refTypes)-1])))
	}
	if version != nil {
		logargs = append(logargs, slog.Int("version", *version))
	} else {
		logargs = append(logargs, slog.String("version", "latest"))
	}
	if label != nil {
		logargs = append(logargs, slog.String("label", *label))
	} else {
		logargs = append(logargs, slog.String("label", "*"))
	}
	logger.InfoContext(ctx, "GetAssetsByRefs", logargs...)
	return a.indexer.Assets.GetAssetsByRefs(ctx, kbId, assetType, refIds, refTypes, version, label)
}

func (a *FastGraphService) DeleteAssetsByKbId(ctx context.Context, kbId string) (int, error) {
	logger := am.GetLogger(a)
	logger.InfoContext(ctx, "DeleteAssetsByKbId", slog.String("kbId", kbId))
	total, err := a.indexer.Assets.DeleteAssetsByKbId(ctx, kbId)
	if err != nil {
		return total, err
	}
	logger.InfoContext(ctx, "DeleteVectorsByKbId", slog.String("kbId", kbId))
	vtotal, err := a.indexer.Vectors.DeleteVectorsByKbId(ctx, kbId)
	if err != nil {
		return total, err
	}
	return total + vtotal, nil
}

func (a *FastGraphService) DeleteAssetsByKbIdAndVersion(ctx context.Context, kbId string, version *int) (int, error) {
	logger := am.GetLogger(a)
	logargs := []any{
		slog.String("kbId", kbId),
	}
	if version != nil {
		logargs = append(logargs, slog.Int("version", *version))
	} else {
		logargs = append(logargs, slog.String("version", "latest"))
	}
	logger.InfoContext(ctx, "DeleteAssetsByKbIdAndVersion", logargs...)
	total, err := a.indexer.Assets.DeleteAssetsByKbIdAndVersion(ctx, kbId, version)
	if err != nil {
		return total, err
	}
	logger.InfoContext(ctx, "DeleteVectorsByKbIdAndVersion", slog.String("kbId", kbId), slog.Int("version", *version))
	vtotal, err := a.indexer.Vectors.DeleteVectorsByKbIdAndVersion(ctx, kbId, version)
	if err != nil {
		return total, err
	}
	return total + vtotal, nil
}
