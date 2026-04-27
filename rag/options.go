package rag

import (
	"encoding/json"

	"github.com/manageai-inet/fast-graphrago/graph"
	"github.com/manageai-inet/fast-graphrago/llm"
	"github.com/manageai-inet/fast-graphrago/utils"

	chunk "github.com/manageai-inet/agentic-assets/chunk"
)

type GraphRAGServiceCompileTimeOptions struct {
	// (Engine Config) Max concurrent for processing (e.g. chunking, extraction, and embedding)
	MaxConcurrent int `json:"max_concurrent"`
	// (Engine Config) LLM for entities and relations extraction, should support tool calling
	LLM llm.LLM
	// (Engine Config) Chunk Extractor
	ChunkingExtractor chunk.ChunkExtractor
	// (Engine Config) Graph Extractor
	GraphExtractor graph.GraphExtractor
}

type GraphRAGServiceRuntimeOptions struct {
	// (Indexing Options) Specific Domain for Graph Extraction (e.g. "Finance", "Healthcare", "Manufacturing", "Enterprise")
	Domain string `json:"domain,omitempty"`
	// (Indexing Options) Entity Types for Graph Extraction (e.g. "Person", "Organization", "Location", "Product", "Event", "Other")
	EntityTypes []string `json:"entity_types,omitempty"`

	// (Retrieval Options) Max walks for PageRank
	MaxWalks int `json:"max_walks,omitempty"`
	// (Retrieval Options) Damping factor for PageRank (also known as alpha)
	DampingFactor float32 `json:"damping_factor,omitempty"`
	// (Retrieval Options) Relative weight for entities extracted from query by type, order is [named_entities, generic_entities, relations]
	// they will be divided by number of entities in each type
	// Default is (0.35, 0.15, 0.5)
	QueryWeights [3]float32 `json:"query_weights,omitempty"`
	// (Retrieval Options) Top K for PageRank
	TopK *int `json:"top_k,omitempty"`
	// (Retrieval Options) Threshold for PageRank
	Threshold *float32 `json:"threshold,omitempty"`

	// (Retrieval Options) Version of the knowledge base to retrieve
	Version *int `json:"version,omitempty"`
	// (Retrieval Options) Label of the knowledge base to retrieve
	Label *string `json:"label,omitempty"`
}

type GraphRAGServiceOptions struct {
	GraphRAGServiceCompileTimeOptions
	GraphRAGServiceRuntimeOptions
}

func NewGraphRAGServiceOptions() *GraphRAGServiceOptions {
	defaultTopK := utils.DefaultTopK
	defaultThreshold := utils.DefaultThreshold
	return &GraphRAGServiceOptions{
		GraphRAGServiceCompileTimeOptions: GraphRAGServiceCompileTimeOptions{
			MaxConcurrent: utils.DefaultMaxConcurrent,
		},
		GraphRAGServiceRuntimeOptions: GraphRAGServiceRuntimeOptions{
			Domain:        utils.DefaultDomain,
			EntityTypes:   utils.DefaultEntityTypes,
			MaxWalks:      utils.DefaultMaxWalks,
			DampingFactor: utils.DefaultDampingFactor,
			QueryWeights:  utils.DefaultQueryWeights,
			TopK:          &defaultTopK,
			Threshold:     &defaultThreshold,
		},
	}
}

func MergeGraphRAGServiceRuntimeOptions(opts1 GraphRAGServiceRuntimeOptions, opts2 *GraphRAGServiceRuntimeOptions) GraphRAGServiceRuntimeOptions {
	if opts2 == nil {
		return opts1
	}
	if opts1.Domain == "" {
		opts1.Domain = opts2.Domain
	}
	if len(opts1.EntityTypes) == 0 {
		opts1.EntityTypes = opts2.EntityTypes
	}
	if opts1.MaxWalks == 0 {
		opts1.MaxWalks = opts2.MaxWalks
	}
	queryWeightsSum := 0.0
	for _, w := range opts1.QueryWeights {
		queryWeightsSum += float64(w)
	}
	if queryWeightsSum == 0 {
		opts1.QueryWeights = opts2.QueryWeights
	} else if queryWeightsSum != 1 {
		for i := range opts1.QueryWeights {
			opts1.QueryWeights[i] /= float32(queryWeightsSum)
		}
	}
	if opts1.DampingFactor == 0 {
		opts1.DampingFactor = opts2.DampingFactor
	}
	if opts1.TopK == nil {
		opts1.TopK = opts2.TopK
	}
	if opts1.Threshold == nil {
		opts1.Threshold = opts2.Threshold
	}
	if opts1.Version == nil {
		opts1.Version = opts2.Version
	}
	if opts1.Label == nil {
		opts1.Label = opts2.Label
	}
	return opts1
}

func NewGraphRAGServiceRuntimeOptionsFromConfig(config *map[string]any, defaultOpts *GraphRAGServiceRuntimeOptions) *GraphRAGServiceRuntimeOptions {
	opts := GraphRAGServiceRuntimeOptions{}
	if config == nil {
		opts = MergeGraphRAGServiceRuntimeOptions(opts, defaultOpts)
		return &opts
	}
	jsonBytes, err := json.Marshal(config)
	if err != nil {
		opts = MergeGraphRAGServiceRuntimeOptions(opts, defaultOpts)
		return &opts
	}
	if err := json.Unmarshal(jsonBytes, &opts); err != nil {
		opts = MergeGraphRAGServiceRuntimeOptions(opts, defaultOpts)
		return &opts
	}
	opts = MergeGraphRAGServiceRuntimeOptions(opts, defaultOpts)
	return &opts
}

func (opts *GraphRAGServiceOptions) Apply(options ...Option) {
	for _, opt := range options {
		opt(opts)
	}
}

type Option func(*GraphRAGServiceOptions)

func WithMaxConcurrent(maxConcurrent int) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.MaxConcurrent = maxConcurrent
	}
}

func WithChunkingExtractor(chunkingExtractor chunk.ChunkExtractor) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.ChunkingExtractor = chunkingExtractor
	}
}

func WithLLM(llm llm.LLM) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.LLM = llm
		opts.GraphExtractor.SetLLM(llm)
	}
}

func WithGraphExtractor(graphExtractor graph.GraphExtractor) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.GraphExtractor = graphExtractor
		if opts.LLM != nil {
			opts.GraphExtractor.SetLLM(opts.LLM)
		}
	}
}

func WithDomain(domain string, entityTypes []string) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.Domain = domain
		if len(entityTypes) > 0 {
			opts.EntityTypes = entityTypes
		}
	}
}

func WithTopK(topK int) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.TopK = &topK
	}
}

func WithThreshold(threshold float32) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.Threshold = &threshold
	}
}

func WithMaxWalks(maxWalks int) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.MaxWalks = maxWalks
	}
}

// Alias of WithMaxWalks()
func WithMaxIteration(maxIteration int) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.MaxWalks = maxIteration
	}
}

func WithDampingFactor(dampingFactor float32) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.DampingFactor = dampingFactor
	}
}

// Alias of WithDampingFactor()
func WithAlpha(alpha float32) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.DampingFactor = alpha
	}
}

func WithQueryWeights(queryWeights [3]float32) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.QueryWeights = queryWeights
	}
}

// For setting weight of named entities in query (queryWeights[0])
func WithNamedEntityWeight(weight float32) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.QueryWeights[0] = weight
	}
}

// For setting weight of generic entities in query (queryWeights[1])
func WithGenericEntityWeight(weight float32) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.QueryWeights[1] = weight
	}
}

// For setting weight of query (queryWeights[2])
func WithQueryWeight(weight float32) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.QueryWeights[2] = weight
	}
}

func WithVersion(version *int) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.Version = version
	}
}

func WithLabel(label *string) Option {
	return func(opts *GraphRAGServiceOptions) {
		opts.Label = label
	}
}
