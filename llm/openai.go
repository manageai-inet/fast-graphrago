package llm

import (
	"context"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

type OpenAIChatModel struct {
	client openai.Client
	model string
	tools []openai.ChatCompletionToolParam
}

func NewOpenAIChatModel(model string, opts ...option.RequestOption) *OpenAIChatModel {
	client := openai.NewClient(opts...)
	return &OpenAIChatModel{
		model: model,
		client: client,
	}
}

func NewOpenAIChatModelFromEnv() *OpenAIChatModel {
	opts := openai.DefaultClientOptions()
	model := os.Getenv("OPENAI_MODEL_ID")
	if model == "" {
		panic("OPENAI_MODEL_ID is not set")
	}
	return NewOpenAIChatModel(model, opts...)
}

func (m *OpenAIChatModel) Generate(ctx context.Context, messages []openai.ChatCompletionMessage, tools []openai.ChatCompletionToolParam, toolChoice *openai.ChatCompletionToolChoiceOptionUnionParam) (*openai.ChatCompletionMessage, error) {
	inputMessages := []openai.ChatCompletionMessageParamUnion{}
	for _, message := range messages {
		inputMessages = append(inputMessages, message.ToParam())
	}
	chatCompletion, err := m.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Messages: inputMessages,
		Tools: tools,
		Model: openai.ChatModelGPT4o,
		ToolChoice: *toolChoice,
	})
	if err != nil {
		return nil, err
	}
	return &chatCompletion.Choices[0].Message, nil
}