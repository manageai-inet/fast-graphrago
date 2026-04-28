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
- Elasticsearch 8.x (with security enabled by default in Docker Compose setup)
- LLM API access (ManageAI or OpenAI)
- OCR service (optional, for PDF/image processing)
- Docker and Docker Compose (for containerized deployment)

### 1. Environment Setup

Copy the example environment file and configure your settings:

```bash
cp example.env .env
```

Edit `.env` with your configuration:

```env
# Elasticsearch
APP_ELASTIC_HOST=http://localhost:9200
APP_ELASTIC_USER=elastic
APP_ELASTIC_PASSWORD=changeme
ELASTIC_PASSWORD=changeme

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

Use Docker Compose to start Elasticsearch and Kibana. The setup includes:
- Elasticsearch with security enabled
- Automatic Kibana service token generation
- Persistence volumes for data

```bash
docker compose up -d elasticsearch elasticsearch-ui
```

The service will wait for Elasticsearch to be healthy and generate the Kibana service token automatically.

If you previously started the stack with different security settings, clean up and recreate:

```bash
docker compose down -v
docker compose up -d elasticsearch elasticsearch-ui
```

### 3. Verify Elasticsearch Connection

Verify Elasticsearch is accessible with your configured credentials:

```bash
curl -u elastic:changeme http://localhost:9200
```

You should see the Elasticsearch cluster info.

### 4. Run the Server

```bash
go run main.go
```

The server will start on `http://localhost:3000`.

### 5. Access API Documentation

Visit `http://localhost:3000/docs` for interactive Swagger documentation.

### 6. Access Kibana (Optional)

Kibana is available at `http://localhost:5601` for viewing Elasticsearch indices and monitoring. Use credentials:
- Username: `elastic`
- Password: `changeme` (or your configured `ELASTIC_PASSWORD`)

## Docker Deployment

Build and run the complete stack with Docker Compose:

```bash
docker compose up --build
```

This will start:
- **Elasticsearch 8.17.1** on port 9200 (with security enabled)
- **Kibana 8.17.1** on port 5601 (with auto-generated service token)
- **FastGraphRAG API server** on port 3000

### Container Services

1. **elasticsearch**: The search and analytics engine with security enabled
2. **kibana-token-init**: Generates Kibana service token (runs once)
3. **elasticsearch-ui**: Kibana for cluster monitoring and index management
4. **fastgraphrago-example**: The REST API server (built from Dockerfile)

### Stopping and Cleaning Up

To stop the services:

```bash
docker compose down
```

To remove all data volumes (clean slate):

```bash
docker compose down -v
```

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
| `APP_ELASTIC_USER` | Elasticsearch username | elastic |
| `APP_ELASTIC_PASSWORD` | Elasticsearch password | changeme |
| `ELASTIC_PASSWORD` | Docker Compose Elasticsearch password (must match `APP_ELASTIC_PASSWORD`) | changeme |
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
   - Ensure Elasticsearch is running: `docker compose ps`
   - Check network connectivity and authentication settings
   - Verify credentials in `.env` match docker-compose configuration
   - Try connecting manually: `curl -u elastic:changeme http://localhost:9200`

2. **Authentication Error (401 Unauthorized)**
   - Ensure `APP_ELASTIC_USER` and `APP_ELASTIC_PASSWORD` match Elasticsearch credentials
   - Verify `ELASTIC_PASSWORD` in `.env` (used by docker-compose)
   - Make sure credentials haven't changed between restarts

3. **LLM API Errors**
   - Verify API keys and endpoints in `.env`
   - Check rate limits and account permissions

4. **OCR Service Issues**
   - Confirm OCR service credentials
   - Check supported file formats

5. **Memory Issues**
   - Adjust `APP_MAX_CONCURRENT` for your system
   - Monitor Elasticsearch resource usage

### Kibana Service Token Issues

- If Kibana fails to connect, check that `kibana-token-init` service completed successfully
- View logs: `docker compose logs kibana-token-init`
- The token file is mounted in a volume at `/token/kibana_service_token`

### Container Logs

View logs from any service:

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f elasticsearch
docker compose logs -f fastgraphrago-example
docker compose logs -f elasticsearch-ui
```

For the running application, set `APP_LOG_LEVEL=DEBUG` for verbose logging.

## Contributing

This example serves as a reference implementation. Contributions that improve documentation, add features, or fix issues are welcome.

## License

See root [LICENSE](../LICENSE) file.</content>
<parameter name="filePath">/Users/thebook/Documents/go-packages/fast-graphrago/example/README.md
