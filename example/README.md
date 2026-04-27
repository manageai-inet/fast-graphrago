# FastGraphRAG Example API Server

This directory contains a complete example implementation of a REST API server that demonstrates how to use the `fast-graphrago` library for knowledge graph construction and retrieval-augmented generation (RAG).

## Overview

The example provides a production-ready API server built with [Fiber](https://gofiber.io/) that exposes FastGraphRAG functionality through REST endpoints. It includes:

- **Document Indexing**: Process and index various document types (PDF, images, text files)
- **Knowledge Retrieval**: Query indexed knowledge bases with graph-enhanced RAG
- **Multi-format Support**: Handle PDFs, images, and text documents with OCR support
- **Elasticsearch Integration**: Store vectors and assets in Elasticsearch
- **Swagger Documentation**: Interactive API documentation at `/docs`

## Features

### Document Processing
- **PDF Support**: Extract text from PDF files using OCR layout analysis
- **Image Support**: Process images with OCR for text extraction
- **Text Files**: Support for various text formats (txt, md, csv, json, yaml, xml, html)
- **HTTP Loading**: Load documents from URLs or provide content directly

### API Endpoints

#### Indexing
```http
POST /api/v1/fastgraph/index
```
Index documents into a knowledge base with configurable entity extraction.

#### Retrieval
```http
POST /api/v1/fastgraph/retrieve
```
Query knowledge bases and retrieve contextually relevant information.

#### Asset Management
```http
GET /api/v1/fastgraph/assets/{kb_id}
GET /api/v1/fastgraph/assets/{kb_id}/{asset_type}
GET /api/v1/fastgraph/assets/refs/{kb_id}/{asset_type}
GET /api/v1/fastgraph/versions/{kb_id}
```
Manage and retrieve indexed assets and knowledge base versions.

## Quick Start

### Prerequisites

- Go 1.25 or higher
- Elasticsearch 8.x
- LLM API access (ManageAI or OpenAI)
- OCR service (optional, for PDF/image processing)

### 1. Environment Setup

Copy the example environment file and configure your settings:

```bash
cp example.env .env
```

Edit `.env` with your configuration:

```env
# Elasticsearch
APP_ELASTIC_HOST=http://localhost:9200

# LLM Configuration (Choose one)
# For ManageAI
MAI_BASE_URL=https://manageai-llm-api-base-url
MAI_MODEL_ID=your-model-id
MAI_API_KEY=your-api-key
MAI_EMBEDDING_MODEL=your-embedding-model
MAI_EMBEDDING_DIM=384

# For OpenAI
OPENAI_MODEL_ID=gpt-4
OPENAI_API_KEY=your-openai-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIM=1536

# OCR Service (Optional)
OCR_SERVICE_API_PATH=https://your-ocr-service.com
OCR_SERVICE_API_KEY=your-ocr-key
```

### 2. Start Dependencies

Use Docker Compose to start Elasticsearch:

```bash
docker-compose up -d elasticsearch elasticsearch-ui
```

### 3. Run the Server

```bash
go run main.go
```

The server will start on `http://localhost:3000`.

### 4. Access API Documentation

Visit `http://localhost:3000/docs` for interactive Swagger documentation.

## Docker Deployment

Build and run with Docker Compose:

```bash
docker-compose up --build
```

This will start:
- Elasticsearch on port 9200
- Kibana on port 5601
- FastGraphRAG API server on port 3000

## Usage Examples

### Index a Document

```bash
curl -X POST http://localhost:3000/api/v1/fastgraph/index \
  -H "Content-Type: application/json" \
  -d '{
    "file": {
      "file_name": "document.pdf",
      "file_type": "pdf",
      "file_url": "https://example.com/document.pdf"
    },
    "kb_id": "my-knowledge-base",
    "domain": "medical",
    "entity_types": ["disease", "medication", "symptom"]
  }'
```

### Retrieve Knowledge

```bash
curl -X POST http://localhost:3000/api/v1/fastgraph/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the symptoms of diabetes?",
    "kb_ids": ["my-knowledge-base"],
    "top_k": 5
  }'
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_LOG_LEVEL` | Logging level (DEBUG, INFO, WARN, ERROR) | INFO |
| `APP_CHUNK_SIZE` | Text chunk size for processing | 500 |
| `APP_CHUNK_OVERLAP` | Overlap between text chunks | 50 |
| `APP_ELASTIC_HOST` | Elasticsearch URL | http://localhost:9200 |
| `APP_ASSETS_STORAGE_INDEX` | Index for storing assets | fast-graphrago-assets |
| `APP_VECTOR_STORAGE_INDEX` | Index for storing vectors | fast-graphrago-vectors |
| `APP_MAX_CONCURRENT` | Max concurrent operations | 1 |

### LLM Configuration

Choose either ManageAI or OpenAI:

**ManageAI:**
- `MAI_BASE_URL`: API endpoint
- `MAI_MODEL_ID`: Chat model ID
- `MAI_API_KEY`: API key
- `MAI_EMBEDDING_MODEL`: Embedding model name
- `MAI_EMBEDDING_DIM`: Embedding dimensions

**OpenAI:**
- `OPENAI_MODEL_ID`: Chat model (e.g., gpt-4)
- `OPENAI_API_KEY`: API key
- `OPENAI_EMBEDDING_MODEL`: Embedding model
- `OPENAI_EMBEDDING_DIM`: Embedding dimensions

### OCR Configuration (Optional)

For PDF and image processing:
- `OCR_SERVICE_API_PATH`: OCR service endpoint
- `OCR_SERVICE_API_KEY`: OCR API key
- `OCR_SERVICE_PROJECT_ID`: Project ID
- `OCR_SERVICE_LOCATION`: Service location

## Architecture

The example demonstrates a clean architecture:

- **`main.go`**: Server initialization and routing
- **`config.go`**: Configuration loading and service setup
- **`service.go`**: Business logic layer
- **`handler.go`**: HTTP handlers and API endpoints
- **`models/`**: Request/response data structures
- **`loaders/`**: Custom document loaders for different formats

## Development

### Project Structure

```
example/
├── main.go           # Server entry point
├── config.go         # Configuration and service initialization
├── service.go        # Business logic
├── handler.go        # HTTP handlers
├── models/           # API models
│   ├── requests.go
│   └── responses.go
├── loaders/          # Document loaders
│   ├── image.go
│   └── pdf.go
├── docs/             # Swagger documentation
├── Dockerfile        # Container build
├── docker-compose.yaml
├── example.env       # Environment template
└── README.md         # This file
```

### Adding New Features

1. Define request/response models in `models/`
2. Implement business logic in `service.go`
3. Add HTTP handlers in `handler.go`
4. Register routes in `RegisterRoutes()`

## Troubleshooting

### Common Issues

1. **Elasticsearch Connection Failed**
   - Ensure Elasticsearch is running on the configured host
   - Check network connectivity and authentication settings

2. **LLM API Errors**
   - Verify API keys and endpoints in `.env`
   - Check rate limits and account permissions

3. **OCR Service Issues**
   - Confirm OCR service credentials
   - Check supported file formats

4. **Memory Issues**
   - Adjust `APP_MAX_CONCURRENT` for your system
   - Monitor Elasticsearch resource usage

### Logs

Check application logs for detailed error information. Set `APP_LOG_LEVEL=DEBUG` for verbose logging.

## Contributing

This example serves as a reference implementation. Contributions that improve documentation, add features, or fix issues are welcome.

## License

See root [LICENSE](../LICENSE) file.</content>
<parameter name="filePath">/Users/thebook/Documents/go-packages/fast-graphrago/example/README.md