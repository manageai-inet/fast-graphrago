package models

import (
	"time"
	am "github.com/manageai-inet/agentic-assets"
)

type TextExtractionResult struct {
	// filename to be indexed, including extension
	FileName string `json:"file_name"`
	// file type to be indexed
	FileType string `json:"file_type"`
	// extracted text content, each element represents a fragment of given file (however, it will be considered as a page).
	TextContent string `json:"text_content"`
}

type IndexingResult struct {
	// knowledge base id
	KbId           string          `json:"kb_id"`
	// indexed at
	IndexedAt      time.Time       `json:"indexed_at"`
	// version
	Version        *int            `json:"version"`
	// embedding model
	EmbeddingModel *string         `json:"embedding_model"`
	// embedding dimension
	EmbeddingDim   *int            `json:"embedding_dim"`
	// assets count by type, key is asset type and value is count
	AssetsCount    *map[string]int `json:"assets_count"`
}

type RetrieveResult struct {
	// query
	Query           string               `json:"query"`
	// knowledge base ids
	KbIds           []string             `json:"kb_ids"`
	// retrieved at
	RetrieveAt      time.Time            `json:"retrieve_at"`
	// retrieved count
	RetrievedCount  int                  `json:"retrieved_count"`
	// retrieved assets
	RetrievedAssets *[]am.RetrievedAsset `json:"retrieved_assets"`
	// config
	Config          *map[string]any      `json:"config"`
}

type AssetsResponse struct {
	// knowledge base id
	KbId           string          `json:"kb_id"`
	// assets count
	AssetsCount    int             `json:"assets_count"`
	// assets
	Assets         []am.ContextualAsset `json:"assets"`
}

type ErrorResponse struct {
	FailedField string `json:"failed_field"`
	Tag         string `json:"tag"`
	Value       string `json:"value"`
}