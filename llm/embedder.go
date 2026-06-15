package llm

import (
	"context"
	"fmt"
	"os"
	"strconv"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

type OpenAIEmbedder struct {
	client openai.Client
	model string
	dim int
}

func NewOpenAIEmbedder(model string, dim int, opts ...option.RequestOption) *OpenAIEmbedder {
	client := openai.NewClient(opts...)
	return &OpenAIEmbedder{
		model: model,
		client: client,
		dim: dim,
	}
}

func NewOpenAIEmbedderFromEnv() *OpenAIEmbedder {
	opts := openai.DefaultClientOptions()
	model := os.Getenv("OPENAI_EMBEDDING_MODEL_ID")
	if model == "" {
		panic("OPENAI_EMBEDDING_MODEL_ID is not set")
	}
	dim := os.Getenv("OPENAI_EMBEDDING_DIM")
	if dim == "" {
		panic("OPENAI_EMBEDDING_DIM is not set")
	}
	dimInt, err := strconv.Atoi(dim)
	if err != nil {
		panic("OPENAI_EMBEDDING_DIM is not a valid integer")
	}
	return NewOpenAIEmbedder(model, dimInt, opts...)
}

func (m *OpenAIEmbedder) GetEmbeddingModel() string {
	return m.model
}

func (m *OpenAIEmbedder) GetEmbeddingDim() int {
	return m.dim
}

func (m *OpenAIEmbedder) Embed(ctx context.Context, content string) ([]float32, error) {
	embeddingResponse, err := m.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{OfString: openai.String(content)}, Model: m.model,
	})
	if err != nil {
		return nil, err
	}
	vector := embeddingResponse.Data[0].Embedding
	// cast to []float32
	embedding := make([]float32, len(vector))
	for i, v := range vector {
		embedding[i] = float32(v)
	}
	return embedding, nil
}

func (m *OpenAIEmbedder) EmbedBatch(ctx context.Context, contents []string) ([][]float32, error) {
	embeddingResponse, err := m.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: contents}, Model: m.model,
	})
	if err != nil {
		return nil, err
	}
	// OpenAI does not guarantee response order matches input order.
	// Use data.Index (the input index) instead of the response position.
	embeddings := make([][]float32, len(contents))
	for _, data := range embeddingResponse.Data {
		idx := int(data.Index)
		if idx < 0 || idx >= len(embeddings) {
			return nil, fmt.Errorf("openai returned embedding with out-of-range index %d (batch size %d)", idx, len(contents))
		}
		vector := data.Embedding
		embedding := make([]float32, len(vector))
		for j, v := range vector {
			embedding[j] = float32(v)
		}
		embeddings[idx] = embedding
	}
	// verify all slots were filled
	for i, e := range embeddings {
		if e == nil {
			return nil, fmt.Errorf("openai did not return embedding for input index %d (batch size %d)", i, len(contents))
		}
	}
	return embeddings, nil
}