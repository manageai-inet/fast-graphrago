package models

import (
	"github.com/manageai-inet/fast-graphrago/utils"

	"github.com/james-bowman/sparse"
	"github.com/openai/openai-go"
)

type GraphEntity struct {
	Name       string `json:"name"        jsonschema:"The name of the entity"`
	Type       string `json:"type"        jsonschema:"The type of the entity"`
	Desc       string `json:"desc"        jsonschema:"The description for the entity context"`
	ChunkIndex int    `json:"chunk_index" jsonschema:"Zero-based index of the source chunk in the input batch"`
}

type RelativeEntity struct {
	Name string `json:"name" jsonschema:"The name of the entity"`
	Type string `json:"type" jsonschema:"The type of the entity"`
}

type GraphRelation struct {
	Source     RelativeEntity `json:"source"      jsonschema:"The source of the relation"`
	Target     RelativeEntity `json:"target"      jsonschema:"The target of the relation"`
	Desc       string         `json:"desc"        jsonschema:"The description for the relation context"`
	ChunkIndex int            `json:"chunk_index" jsonschema:"Zero-based index of the source chunk in the input batch"`
}

type ExtractedGraph struct {
	Entities  []GraphEntity   `json:"entities" jsonschema:"The list of extracted entities"`
	Relations []GraphRelation `json:"relationships" jsonschema:"The list of extracted relations, each relation must have a source and target entity that exist in the entities list"`
}

func GetGraphExtractionTool() (openai.ChatCompletionToolParam, error) {
	name := "graph_extraction"
	description := "Extract entities and relations from the given text"
	tool, err := utils.StructToTool[ExtractedGraph](name, description)
	if err != nil {
		return openai.ChatCompletionToolParam{}, err
	}
	return tool, nil
}

type RelationGroup struct {
	Source  RelativeEntity `json:"source" jsonschema:"The source of the relation"`
	Target  RelativeEntity `json:"target" jsonschema:"The target of the relation"`
	Desc    string         `json:"desc" jsonschema:"The description for the relation context"`
	Indices []int          `json:"indices" jsonschema:"The list of relation indices within the cluster"`
}

type RelationClusters struct {
	Clusters map[string]RelationGroup `json:"clusters" jsonschema:"map of relation clusters, key is the cluster id and value is the relation"`
}

func GetRelationClusterTool() (openai.ChatCompletionToolParam, error) {
	name := "relation_cluster"
	description := "Cluster relations based on semantic similarity"
	tool, err := utils.StructToTool[RelationClusters](name, description)
	if err != nil {
		return openai.ChatCompletionToolParam{}, err
	}
	return tool, nil
}

type PairGroupResult struct {
	GroupId  int                      `json:"group_id"  jsonschema:"The 0-based index of this group, matching the GROUP N label in the input"`
	Clusters map[string]RelationGroup `json:"clusters"  jsonschema:"Map of relation clusters for this group, key is cluster id"`
}

type BatchRelationClusters struct {
	Groups []PairGroupResult `json:"groups" jsonschema:"Array of results, one per input GROUP; every GROUP N in the input must appear here"`
}

func GetBatchRelationClusterTool() (openai.ChatCompletionToolParam, error) {
	name := "batch_relation_cluster"
	description := "Cluster relations for multiple entity pairs at once; one entry per GROUP in the input"
	tool, err := utils.StructToTool[BatchRelationClusters](name, description)
	if err != nil {
		return openai.ChatCompletionToolParam{}, err
	}
	return tool, nil
}

const NamedEntityType = "NAMED"
const GenericEntityType = "GENERIC"
const QueryEntityType = "QUERY"

type ExtractedQuery struct {
	NamedEntities   []string `json:"named_entities" jsonschema:"The list of extracted named entities"`
	GenericEntities []string `json:"generic_entities" jsonschema:"The list of extracted generic entities"`
}

func GetQueryExtractionTool() (openai.ChatCompletionToolParam, error) {
	name := "query_extraction"
	description := "Extract entities from the given query"
	tool, err := utils.StructToTool[ExtractedQuery](name, description)
	if err != nil {
		return openai.ChatCompletionToolParam{}, err
	}
	return tool, nil
}

type EntityAsset struct {
	Name        string
	Type        string
	Description string
	ChunkIds    []string
}

type RelationAsset struct {
	Source      string
	SourceType  string
	Target      string
	TargetType  string
	Description string
	ChunkIds    []string
}

type GraphAssets struct {
	EntityAssets   []EntityAsset
	RelationAssets []RelationAsset
}

type QueryEntity struct {
	Name      string
	Type      string
	Embedding []float32
}

type TransitionMatrix struct {
	CSR              *sparse.CSR
	EntityIndiceToId []string
	EntityIDToIndex  map[string]int
	Dim              int // it is square matrix size, number of entities
}
