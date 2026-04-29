# Fast-GraphRAG Go Implementation

[![GitHub release](https://img.shields.io/github/v/release/manageai-inet/fast-graphrago?label=version)](https://github.com/manageai-inet/fast-graphrago/releases)
[![Go Reference](https://pkg.go.dev/badge/github.com/manageai-inet/fast-graphrago.svg)](https://pkg.go.dev/github.com/manageai-inet/fast-graphrago)

A Go implementation of **FastGraphRAG**, a scalable framework for knowledge graph construction and retrieval-augmented generation (RAG).

## Overview

`fast-graphrago` is a Go library that enables you to build and query knowledge graphs efficiently. It combines LLM-powered entity extraction with vector-based retrieval to provide contextual, knowledge-graph-enhanced RAG capabilities.

### Key Features

- **Graph Construction**: Automatically extract entities and relationships from documents using LLMs
- **Scalable Indexing**: Efficient indexing of knowledge sources with configurable extraction options
- **Hybrid Retrieval**: Combine vector search with graph-based context propagation for better retrieval
- **LLM-Agnostic**: Support for multiple LLM providers through a clean interface
- **Knowledge Management**: Organize and manage knowledge bases with labels and metadata

## Architecture

The library is organized into several core modules:

- **`rag/`**: Core RAG service for indexing and retrieval
- **`graph/`**: Knowledge graph extraction and entity management
- **`llm/`**: LLM provider interfaces and implementations
- **`models/`**: Data models for graphs and extracted entities
- **`prompts/`**: Prompt templates for LLM interactions
- **`utils/`**: Utility functions and constants

## Installation

```bash
go get github.com/manageai-inet/fast-graphrago
```

### Requirements

- Go 1.25 or higher
- LLM API key (for OpenAI or other supported providers)
- Vector storage backend (via `agentic-assets`)
- Asset storage backend (via `agentic-assets`)

## Quick Start

### Basic Usage

```go
import (
    "context"
    "github.com/manageai-inet/fast-graphrago"
)

// Initialize the indexer with an LLM and storage backends
indexer, err := fastgraphrag.New(llm, vectorStore, assetStore)
if err != nil {
    // handle error
}

// Index documents
ctx := context.Background()
result, err := indexer.Index(ctx, "knowledge-base-1", sources, nil, nil, nil)
if err != nil {
    // handle error
}

// Retrieve contextual results
retrieved, _, err := indexer.Rag.Retrieve(ctx, "your query", []string{"knowledge-base-1"}, nil, nil)
if err != nil {
    // handle error
}
```

### Example API Server

For a complete working example, see the [`example/`](./example/) directory, which provides:

- **REST API Server**: Production-ready API with Fiber framework
- **Document Processing**: Support for PDFs, images, and text files
- **Elasticsearch Integration**: Vector and asset storage
- **Interactive Documentation**: Swagger UI at `/docs`
- **Docker Deployment**: Complete containerized setup

To run the example:

```bash
cd example
cp example.env .env  # Configure your settings
docker-compose up -d  # Start Elasticsearch
go run main.go        # Start the API server
```

Visit `http://localhost:3000/docs` for API documentation.

### Configuring Extraction Options

```go
import "github.com/manageai-inet/fast-graphrago/graph"

options := graph.NewGraphExtractionOptions()
options.Domain = "medical"
options.EntityTypes = []string{"disease", "medication", "symptom"}
```

## API Reference

### FastGraphIndexer

Main entry point for the library.

```go
// Create a new indexer
indexer, err := fastgraphrag.New(
    llm,           // LLM implementation
    vectorStore,   // Vector storage backend
    assetStore,    // Asset storage backend
    opts...        // Additional options
)

// Index knowledge sources
result, err := indexer.Index(
    ctx,
    kbId,      // Knowledge base ID
    sources,   // []KnowledgeSource
    labels,    // Optional labels
    metadata,  // Optional metadata
    config,    // Optional configuration
)
```

### GraphRAGService

Core service interface:

```go
// Index sources into the knowledge graph
Index(ctx, kbId, sources, labels, metadata, config) (contextualAssets, vectorAssets, error)

// Retrieve contextual results for a query
Retrieve(ctx, query, kbIds, seedAssets, config) (retrievedAssets, propagatedAssets, error)
```

### GraphExtractor

Handles entity and relationship extraction:

```go
// Extract graph structure from chunks
ExtractGraph(ctx, chunks, options) (GraphAssets, error)

// Extract entities from natural language query
ExtractEntitiesFromQuery(ctx, query) (ExtractedQuery, error)
```

## Configuration

The library supports flexible configuration through:

- **LLM Options** (via `rag.Option`): Customize extraction behavior
- **Extraction Options** (via `graph.GraphExtractionOptions`): Define domain and entity types
- **Storage Configuration**: Configure vector and asset storage backends

## Dependencies

Key dependencies include:

- `github.com/openai/openai-go`: OpenAI API client
- `github.com/google/jsonschema-go`: JSON schema validation
- `github.com/manageai-inet/agentic-assets`: Asset and vector storage abstractions
- `gonum.org/v1/gonum`: Numerical computing for graph operations
- `github.com/tidwall/sjson`: JSON manipulation utilities

## Acknowledgments

This project is a Go implementation inspired by and built upon the excellent work of the original [FastGraphRAG](https://github.com/circlemind-ai/fast-graphrag) project by CircleMind AI. We deeply appreciate their pioneering work on scalable knowledge graph construction and retrieval-augmented generation. This implementation aims to bring the benefits of FastGraphRAG to the Go ecosystem.

## License

See [LICENSE](LICENSE) file for details.
