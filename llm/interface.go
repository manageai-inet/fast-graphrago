package llm

import (
	"context"
	"github.com/openai/openai-go"
)

// LLM defines the interface for interacting with Chat Completion Models.
type LLM interface {
	// LLM implementation must support tool calling, 
	// this will be call at Graph Extraction Step.
	Generate(ctx context.Context, messages []openai.ChatCompletionMessage, tools []openai.ChatCompletionToolParam, toolChoice *openai.ChatCompletionToolChoiceOptionUnionParam) (*openai.ChatCompletionMessage, error)
}