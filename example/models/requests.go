package models

type IndexingRequest struct {
	// files to be indexed
	File      	IndexingFile          `json:"file" validate:"required"`
	// knowledge base id to be indexed
	KbId        string                `json:"kb_id" validate:"required" example:"kb_1234567890"`
	// domain to be indexed, value should be a single word, phrase, or sentence.
	// it's used to provide context to llm for extracting entities and relations.
	Domain      *string               `json:"domain" validate:"optional"`
	// entity types to be indexed, each element should be a single word (no space or using underscore instead)
	// it's used to provide context to llm for extracting entities and relations.
	EntityTypes *[]string             `json:"entity_types" validate:"optional"`
	// labels to be indexed
	Labels      *[]string             `json:"labels" validate:"optional"`
	// metadata to be indexed
	Metadata    map[string]any        `json:"metadata" validate:"optional"`
	// config to be indexed
	Config      map[string]any        `json:"config" validate:"optional"`
}

type IndexingFile struct {
	// filename to be indexed, including extension
	FileName string `json:"file_name" validate:"required" example:"document.pdf"`
	// file type to be indexed
	FileType string `json:"file_type" validate:"required" example:"pdf" enum:"pdf,docx,md,markdown,txt"`
	// file content, if empty, it will download from fileUrl and extract content from it instead
	// each element represents a fragment of given file (however, it will be considered as a page).
	FileContent []string `json:"file_content" validate:"optional"`
	// url to download file from, if file content is empty, it will download from this url and extract content from it instead
	FileUrl *string `json:"file_url" validate:"optional" example:"https://example.com/document.pdf"`
}

type RetrieveRequest struct {
	// query to be retrieved
	Query			string 			`json:"query" validate:"required" minLength:"1"`
	// knowledge base ids to be retrieved
	KbIds 			[]string 		`json:"kb_ids" validate:"required" minLength:"1"`
	
	// label to filter assets
	Label 			*string 		`json:"label" validate:"optional"`
	// version of the knowledge base to retrieve
	Version 		*int 			`json:"version" validate:"optional"`
	// damping factor for PageRank (also known as alpha)
	DampingFactor 	*float32 		`json:"damping_factor" validate:"optional" minimum:"0" maximum:"1"`
	// relative weight for entities extracted from query by type, order is [named_entities, generic_entities, relations]
	// they will be divided by number of entities in each type
	QueryWeights 	*[]float32 		`json:"query_weights" validate:"optional"`
	// Top K for PageRank
	TopK 			*int 			`json:"top_k" validate:"optional"`
	// Threshold for PageRank
	Threshold 		*float32 		`json:"threshold" validate:"optional" minimum:"0" maximum:"1"`
	// config for retrieval process
	Config      	map[string]any 	`json:"config,omitempty" validate:"optional"`
}

type GetAssetsRequest struct {
	// knowledge base id to be retrieved
	KbId 		string 		`uri:"kb_id"`
	// asset type to be retrieved
	AssetType 	string 		`uri:"asset_type"`
}

type GetAssetsQueryParams struct {
	// asset id to be retrieved
	AssetIds 	[]string 	`query:"asset_id"`
	// version of the asset to retrieve
	Version 	*int 		`query:"version"`
	// label to filter assets
	Label 		*string 	`query:"label"`
}

type GetAssetsByRefsQueryParams struct {
	// reference ids to be retrieved
	RefIds 		[]string 	`query:"ref_id"`
	// reference type to filter assets
	RefTypes 	[]string 	`query:"ref_type"`
	// label to filter assets
	Label 		*string 	`query:"label"`
	// version of the asset to retrieve
	Version 	*int 		`query:"version"`
}