package llm

import (
	"context"
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
	embeddings := make([][]float32, len(embeddingResponse.Data))
	for i, data := range embeddingResponse.Data {
		vector := data.Embedding
		embedding := make([]float32, len(vector))
		for j, v := range vector {
			embedding[j] = float32(v)
		}
		embeddings[i] = embedding
	}
	return embeddings, nil
}