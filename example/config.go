package main

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"time"

	"fastgraph-example/loaders"
	"github.com/kelseyhightower/envconfig"

	"github.com/elastic/go-elasticsearch/v8"
	"github.com/lmittmann/tint"

	"github.com/manageai-inet/agentic-assets/chunk"

	"github.com/manageai-inet/fast-graphrago"
	"github.com/manageai-inet/fast-graphrago/graph"
	// "github.com/manageai-inet/fast-graphrago/llm" // uncomment this if you want to use OpenAI as the LLM and Embedder
	"github.com/manageai-inet/fast-graphrago/rag"

	"github.com/manageai-inet/agentic-assets-elasticsearch/v8"

	am "github.com/manageai-inet/agentic-assets"
	amk "github.com/manageai-inet/agentic-assets/knowledge_extractor"
	mai "github.com/manageai-inet/agentic-assets-manageai"
	amai "github.com/manageai-inet/agentic-assets-manageai"
	maiclient "github.com/manageai-inet/Eino-ManageAI-Extension/components/core/manageai"
)

type AppConfig struct {
	// Log level
	LogLevel string `envconfig:"log_level" default:"INFO"`

	// Chunk size for text splitting
	ChunkSize int `envconfig:"chunk_size" default:"500"`
	// Chunk overlap for text splitting
	ChunkOverlap int `envconfig:"chunk_overlap" default:"50"`

	// Elasticsearch configuration
	ElasticHost []string `envconfig:"elastic_host" default:"http://localhost:9200"`
	ElasticUser string   `envconfig:"elastic_user" default:""`
	ElasticPassword string `envconfig:"elastic_password" default:""`
	ElasticAPIKey string `envconfig:"elastic_api_key" default:""`
	ElasticServiceToken string `envconfig:"elastic_service_token" default:""`

	// Index name for storing assets in Elasticsearch
	AssetsStorageIndex string `envconfig:"assets_storage_index" default:"fast-graphrago-assets"`
	VectorStorageIndex string `envconfig:"vector_storage_index" default:"fast-graphrago-vectors"`

	// Maximum number of concurrent operations for RAG
	MaxConcurrent int `envconfig:"max_concurrent" default:"1"`
	// LLM HTTP timeout in seconds (default 5 minutes to support large batch extraction)
	LLMTimeoutSeconds int `envconfig:"llm_timeout_seconds" default:"300"`

	// OCR Service Configuration
	OcrServiceProjectId string `envconfig:"ocr_service_project_id" default:""`
	OcrServiceLocation string `envconfig:"ocr_service_location" default:""`
	OcrServiceAPIPath string `envconfig:"ocr_service_api_path" default:""`
	OcrServiceAPIKey string `envconfig:"ocr_service_api_key" default:""`

	am.LoggingCapacity
}

func LoadConfig(prefix string) (*AppConfig, error) {
	cfg := &AppConfig{LoggingCapacity: *am.GetDefaultLoggingCapacity()}
	err := envconfig.Process(prefix, cfg)
	if err != nil {
		return nil, err
	}
	// Setup Logger
	var level slog.Level
	err = level.UnmarshalText([]byte(strings.ToUpper(cfg.LogLevel)))
	if err != nil {
		return nil, err
	}
	logHand := tint.NewHandler(os.Stdout, &tint.Options{
		Level:      level,
		TimeFormat: time.Kitchen, // Makes it look clean like Fiber
		AddSource:  true,         // Shows the file and line number
	})
	logger := slog.New(logHand)
	cfg.SetLogger(logger)
	
	return cfg, nil
}

func InitializeFastGraphIndexerFromConfig(cfg *AppConfig) (*fastgraphrag.FastGraphIndexer, error) {
	ctx := context.Background()
	logger := am.GetLogger(cfg)
	// FastGraphRAG Application Setup
	logger.Info("Setting up FastGraphRAG Application", slog.String("log_level", cfg.LogLevel))

	// LLM Setup
	// You must set the MAI_BASE_URL, MAI_MODEL_ID and MAI_API_KEY environment variables before running this application.
	maiC, err := maiclient.GetDefaultClient()
	if err != nil {
		logger.Error("Failed to initialize ManageAI Client", slog.Any("error", err))
		return nil, err
	}
	if _, err := maiC.WithTimeout(time.Duration(cfg.LLMTimeoutSeconds) * time.Second); err != nil {
		logger.Error("Failed to set LLM timeout", slog.Any("error", err))
		return nil, err
	}
	chatModel := mai.NewLLM(maiC)

	// To use OpenAI, you can uncomment the following lines and make sure to set the OPENAI_MODEL_ID and OPENAI_API_KEY environment variables before running this application.
	// chatModel := llm.NewOpenAIChatModelFromEnv()

	// Setup Graph Extraction Service
	graphExt := graph.NewFastGraphExtractor(chatModel)
	graphExt.SetLogger(logger)

	// Setup Elasticsearch Asset Store

	// Setup Elasticsearch Client
	elsCfg := elasticsearch.Config{
		Addresses: cfg.ElasticHost,
		Username:  cfg.ElasticUser,
		Password:  cfg.ElasticPassword,
		APIKey:    cfg.ElasticAPIKey,
		ServiceToken: cfg.ElasticServiceToken,
	}
	elsCli, err := elasticsearch.NewTypedClient(elsCfg)
	if err != nil {
		logger.Error("Failed to initialize Elasticsearch Client", slog.Any("error", err))
		return nil, err
	}
	// Initialize Elasticsearch Asset Store
	assetStore := v8.NewElasticsearchV8AssetManagerRepo(cfg.AssetsStorageIndex, elsCli)
	assetStore.SetLogger(logger)
	err = assetStore.Setup(ctx)
	if err != nil {
		logger.Error("Failed to initialize Asset Storage", slog.Any("error", err))
		return nil, err
	}

	// Setup Vector Store

	// Embedder Setup
	// You must set the MAI_BASE_URL, MAI_EMBEDDING_MODEL, MAI_EMBEDDING_DIM and MAI_API_KEY environment variables before running this application.
	embedder := mai.NewEmbedderFromEnv()

	// You must set the OPENAI_EMBEDDING_MODEL_ID, OPENAI_EMBEDDING_DIM, and OPENAI_API_KEY environment variables before running this application.
	// embedder := llm.NewOpenAIEmbedderFromEnv()

	// Initialize Elasticsearch Vector Store
	vectorStore := v8.NewElasticsearchV8VectorRepo(cfg.VectorStorageIndex, elsCli, embedder, "cosine")
	vectorStore.SetLogger(logger)
	err = vectorStore.Setup(ctx)
	if err != nil {
		logger.Error("Failed to initialize Vector Storage", slog.Any("error", err))
		return nil, err
	}

	// Setup Simple Chunk Extractor
	// This is just simple text splitter that will split text into chunks of size `ChunkSize` with an overlap of `ChunkOverlap`. 
	// You can implement your own chunk extractor if you want to do more complex processing (e.g. using a language model to extract chunks).
	chunkExt := chunk.NewSimpleChunkExtractor(&cfg.ChunkSize, &cfg.ChunkOverlap)
	chunkExt.SetLogger(logger)

	// Setup FastGraphRAG Indexer
	opts := []rag.Option{}
	opts = append(opts, rag.WithMaxConcurrent(cfg.MaxConcurrent))
	opts = append(opts, rag.WithChunkingExtractor(chunkExt))
	opts = append(opts, rag.WithGraphExtractor(graphExt))
	opts = append(opts, rag.WithLLM(chatModel))
	// Initialize the FastGraphRAG Indexer with the chat model, vector store, and asset store.
	fgIndexer, err := fastgraphrag.New(chatModel, vectorStore, assetStore, opts...)
	if err != nil {
		logger.Error("Failed to initialize FastGraphRAG", slog.Any("error", err))
		return nil, err
	}
	fgIndexer.SetLogger(logger)
	
	return fgIndexer, nil
}

func InitializeKnowledgeExtractorsFromConfig(cfg *AppConfig) (*map[string]amk.KnowledgeExtractor, error) {
	logger := am.GetLogger(cfg)
	
	logger.Info("Setting up Knowledge Extractors", slog.String("log_level", cfg.LogLevel))

	// Initialize OCR Knowledge Extractor
	logger.Info("Initializing OCR Service")
	ocrCfg := amai.OcrLayoutApiRequestConfig{
		ProjectId: cfg.OcrServiceProjectId,
		Location: cfg.OcrServiceLocation,
	}
	ocrRepo := mai.NewOcrLayoutApiRepo(cfg.OcrServiceAPIPath, cfg.OcrServiceAPIKey, &ocrCfg)
	ocrRepo.SetLogger(logger)

	// key is used for matching with input FileType, 
	// not need to be the same as actual file extension, 
	// but it's better to set it to something meaningful for better logging and debugging.
	extractors := make(map[string]amk.KnowledgeExtractor)

	logger.Info("Initializing Plain Text Knowledge Extractor using default implementation in agentic-assets")
	// Plain Text Converter using the default implementation in agentic-assets, 
	// which simply returns the original text content without any processing. 
	// You can implement your own converter if you want to do some custom processing (e.g. cleaning, formatting) on the text content.
	textConv := amk.NewPlainTextConverter()
	textConv.SetLogger(logger)

	// Plain Text Loader to load plain text files from HTTP URLs
	textLoadHttpLoader := amk.NewHttpFileLoader(nil)
	textLoadHttpLoader.SetLogger(logger)

	// As mentioned above, key is used for matching with input FileType, not need to be the same as actual file extension, 
	// but it's better to set it to something meaningful for better logging and debugging.
	textExt := amk.NewDocumentExtractor(map[amk.KnowledgeLoader]amk.KnowledgeConverter{
		textLoadHttpLoader: textConv,
	})
	textExt.SetLogger(logger)
	for _, ext := range []string{"txt", "text", "md", "csv", "json", "yaml", "yml", "xml", "log", "html"} {
		extractors[ext] = textExt
	}

	// Initialize PDF Knowledge Extractor with OCR support
	logger.Info("Initializing PDF Knowledge Extractor with OCR support")
	// PDF Converter using OCR for layout analysis and text extraction
	pdfConv := mai.NewOcrLayoutApiConverter("pdf", ocrRepo)
	pdfConv.SetLogger(logger)
	
	// PDF Loader to load PDF files from HTTP URLs
	pdfLoadHttpLoader := loaders.NewHttpPDFLoader(nil)
	pdfLoadHttpLoader.SetLogger(logger)

	// Following is an example of how to add another PDF loader that loads PDF files from local file system.
	// This is just for demonstration purpose, you can choose to implement it or not based on your use case. 
	// If you want to support loading PDF files from local file system, 
	// you can uncomment the following lines and make sure to put your PDF files under the specified root path.

	// PDF Loader to load PDF files from local file system
	// pdfLoadLocalLoader := loaders.NewLocalPDFLoader("./static/path/to/pdf/directory")
	// pdfLoadLocalLoader.SetLogger(logger)

	pdfExt := amk.NewDocumentExtractor(map[amk.KnowledgeLoader]amk.KnowledgeConverter{
		pdfLoadHttpLoader: pdfConv,
		// pdfLoadLocalLoader: pdfConv, // uncomment this if you want to support loading PDF files from local file system
	})
	pdfExt.SetLogger(logger)
	extractors["pdf"] = pdfExt

	return &extractors, nil
}