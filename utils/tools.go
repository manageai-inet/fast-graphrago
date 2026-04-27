package utils

import (
	"encoding/json"
    "fmt"

    "github.com/google/jsonschema-go/jsonschema"
    "github.com/openai/openai-go"
)

func StructToTool[T any](name, description string) (openai.ChatCompletionToolParam, error) {
    // 1. Infer the schema from type T
    // This automatically handles nested structs, slices, and maps
    schema, err := jsonschema.For[T](nil)
    if err != nil {
        return openai.ChatCompletionToolParam{}, fmt.Errorf("failed to infer schema from type for tool %s: %w", name, err)
    }

    var params openai.FunctionParameters
    b, err := json.Marshal(schema)
    if err != nil {
        return openai.ChatCompletionToolParam{}, fmt.Errorf("failed to marshal schema for tool %s: %w", name, err)
    }
    err = json.Unmarshal(b, &params)
    if err != nil {
        return openai.ChatCompletionToolParam{}, fmt.Errorf("failed to unmarshal schema for tool %s: %w", name, err)
    }

    // 2. Wrap into the SDK's Tool format
    return openai.ChatCompletionToolParam{
        Function: openai.FunctionDefinitionParam{
            Name:        name,
            Description: openai.String(description),
            Parameters:  params,
        },
    }, nil
}

// HandleToolCall takes a JSON string and returns a pointer to the specific struct type
func HandleToolCall[T any](jsonArgs string) (*T, error) {
    var args T

    // Unmarshal the LLM's JSON into the generic type T
    err := json.Unmarshal([]byte(jsonArgs), &args)
    if err != nil {
        return nil, fmt.Errorf("failed to parse tool arguments into %T: %w", args, err)
    }

    return &args, nil
}