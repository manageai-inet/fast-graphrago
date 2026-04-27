// This is an example of how to use the FastGraphRAG library in a Go application.
// It demonstrates how to set up the graph extraction and RAG services, and how to call them with sample data.
package main

import (
	docs "fastgraph-example/docs"

	"github.com/joho/godotenv"
	"github.com/yokeTH/gofiber-scalar/scalar/v3"

	am "github.com/manageai-inet/agentic-assets"
	fiber "github.com/gofiber/fiber/v3"
)

// @title           FastGraphRAG API
// @version         v1.0.0
// @description     This is a simple implementation example of FastGraphRAG API server.
// @contact.name   	ManageAI Team
// @contact.email  	manageai@inet.co.th

// @license.name  MIT

// @host      localhost:3000
// @BasePath  /api/v1

// @externalDocs.description  OpenAPI
// @externalDocs.url          https://swagger.io/resources/open-api/
func main() {
	// Load envkjnlknklironment variables from .env file
	err := godotenv.Load()
	if err != nil {
		panic("Error loading .env file")
	}
	cfg, err := LoadConfig("app")
	if err != nil {
		panic(err.Error())
	}
	logger := am.GetLogger(cfg)
	
	// Initialize FastGraphRAG Indexer from configuration
	logger.Info("Initializing FastGraphRAG Indexer from configuration")
	indexer, err := InitializeFastGraphIndexerFromConfig(cfg)
	if err != nil {
		panic(err.Error())
	}
	extractors, err := InitializeKnowledgeExtractorsFromConfig(cfg)
	if err != nil {
		panic(err.Error())
	}
	
	// Initialize FastGraphService with the indexer and extractors
	logger.Info("Initializing FastGraphService with the indexer and extractors")
	service := NewFastGraphService(*extractors, *indexer)
	service.SetLogger(logger)
	// Call the Index method of the service with sample data

	logger.Info("Initializing FastGraphRAGHandler with the service")
	hand := NewFastGraphRAGHandler(*service)

	app := fiber.New(fiber.Config{
		AppName: "ManageAI FastGraphRAG Examples",
	})

	app.Get("/docs/*", scalar.New(scalar.Config{
        FileContentString: docs.SwaggerInfo.ReadDoc(),
		Theme: scalar.ThemeAlternate,
    }))
	hand.RegisterRoutes(app)

	logger.Info("Starting FastGraphRAG API server on port 3000")
	err = app.Listen(":3000")
	if err != nil {
		panic(err.Error())
	}
}