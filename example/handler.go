package main

import (
	"fastgraph-example/models"
	"errors"
	"log/slog"
	"github.com/gofiber/fiber/v3"
	am "github.com/manageai-inet/agentic-assets"
)

type HandlerGroup interface {
	RegisterRoutes(router fiber.Router)
	am.Loggable
}

func WrapHandler(h HandlerGroup, handler func(c fiber.Ctx) error) func(c fiber.Ctx) error {
	return func(c fiber.Ctx) error {
		logger := am.GetLogger(h)
		logger.Info("Accept Request", slog.String("method", c.Route().Method), slog.String("path", c.FullURL()))
		return handler(c)
	}
}

type FastGraphRAGHandler struct {
	RagSrv    FastGraphService
	am.LoggingCapacity
}

func NewFastGraphRAGHandler(
	ragSrv FastGraphService,
) *FastGraphRAGHandler {
	return &FastGraphRAGHandler{
		RagSrv: ragSrv,
		LoggingCapacity: *am.GetDefaultLoggingCapacity(),
	}
}

func (h *FastGraphRAGHandler) RegisterRoutes(router fiber.Router) {
	group := router.Group("/api/v1/fastgraph")
	group.Post("/index", WrapHandler(h, h.Index))
	group.Post("/retrieve", WrapHandler(h, h.Retrieve))

	group.Get("/assets/refs/:kb_id/:asset_type", WrapHandler(h, h.GetAssetsByRefs))
	group.Get("/assets/:kb_id/:asset_type", WrapHandler(h, h.GetAssets))
	group.Get("/assets/:kb_id", WrapHandler(h, h.GetAssetsByKbId))
	group.Get("/versions/:kb_id", WrapHandler(h, h.GetVersions))
	group.Delete("/:kb_id", WrapHandler(h, h.DeleteKbId))
	group.Post("/extract", WrapHandler(h, h.Extract))
}

// FastGraphRAG Text Extraction API V1
//
//	@Summary		Extracting text content from document
//	@Description	Process extracting text content from given document, support pdf, docx, md, markdown, txt
//	@Tags			FastGraph RAG API
//	@Accept			json
//	@Produce		json
//	@Param			data			body		models.TextExtractionRequest	true	"Text Extraction Request"
//	@Success		200				{object}	models.TextExtractionResult
//	@Failure		400				{object}	models.ErrorResponse
//	@Failure		500				{object}	models.ErrorResponse
//	@Router			/fastgraph/extract [post]
func (h *FastGraphRAGHandler) Extract(c fiber.Ctx) error {
	logger := am.GetLogger(h)
	req := new(models.TextExtractionRequest)
	if err := c.Bind().Body(req); err != nil {
		logger.ErrorContext(c, "failed to bind request body", slog.Any("error", err))
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}
	logger.DebugContext(c, "Accept Request", slog.Any("request", req))
	sources, err := h.RagSrv.Extract(c, req.File)
	if err != nil {
		logger.ErrorContext(c, "failed to extract text", slog.Any("error", err))
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}
	result := []models.TextExtractionResult{}
	for _, source := range sources {
		content := ""
		if source.SourceContents != nil {
			content = *source.SourceContents
		}
		result = append(result, models.TextExtractionResult{
			FileName:    source.SourceName,
			FileType:    source.SourceType,
			TextContent: content,
		})
	}
	return c.Status(fiber.StatusOK).JSON(result)
}

// FastGraphRAG Indexing API V1
//
//	@Summary		Indexing a document
//	@Description	Process indexing a given document bound to specific kb_id, support pdf, docx, md, markdown, txt
//	@Tags			FastGraph RAG API
//	@Accept			json
//	@Produce		json
//	@Param			data			body		models.IndexingRequest	true	"Indexing Request"
//	@Success		201				{object}	models.IndexingResult
//	@Failure		400				{object}	models.ErrorResponse
//	@Failure		500				{object}	models.ErrorResponse
//	@Router			/fastgraph/index [post]
func (h *FastGraphRAGHandler) Index(c fiber.Ctx) error {
	logger := am.GetLogger(h)
	req := new(models.IndexingRequest)
	if err := c.Bind().Body(req); err != nil {
		return err
	}
	logger.DebugContext(c, "Accept Request", slog.Any("request", req))
	config := make(map[string]any)
	for k, v := range req.Config {
		config[k] = v
	}
	if req.Domain != nil {
		config["domain"] = *req.Domain
	}
	if req.EntityTypes != nil {
		config["entity_types"] = *req.EntityTypes
	}
	result, err := h.RagSrv.Index(c, req.File, req.KbId, req.Labels, &req.Metadata, &config)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}
	if result.Status {
		r := models.IndexingResult{
			KbId: result.KbId,
			IndexedAt: result.IndexedAt,
			Version: result.Version,
			EmbeddingModel: result.EmbeddingModel,
			EmbeddingDim: result.EmbeddingDim,
			AssetsCount: result.AssetsCount,
		}
		return c.Status(fiber.StatusCreated).JSON(r)
	} else {
		err = errors.New(*result.Error)
	}
	return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
}

// FastGraphRAG Retrieving API V1
//
//	@Summary		Retrieving knowledge from knowledge base
//	@Description	Process retrieving knowledge from knowledge base
//	@Tags			FastGraph RAG API
//	@Accept			json
//	@Produce		json
//	@Param			data		body		models.RetrieveRequest	true	"Retrieve Request"
//	@Success		200			{object}	models.RetrieveResult
//	@Failure		400			{object}	models.ErrorResponse
//	@Failure		500			{object}	models.ErrorResponse
//	@Router			/fastgraph/retrieve [post]
func (h *FastGraphRAGHandler) Retrieve(c fiber.Ctx) error {
	logger := am.GetLogger(h)
	req := new(models.RetrieveRequest)
	if err := c.Bind().Body(req); err != nil {
		return err
	}
	logger.DebugContext(c, "Accept Request", slog.Any("request", req))
	config := make(map[string]any)
	for k, v := range req.Config {
		config[k] = v
	}
	if req.Label != nil {
		config["label"] = *req.Label
	}
	if req.Version != nil {
		config["version"] = *req.Version
	}
	if req.DampingFactor != nil {
		config["damping_factor"] = *req.DampingFactor
	}
	if req.QueryWeights != nil {
		config["query_weights"] = *req.QueryWeights
	}
	if req.TopK != nil {
		config["top_k"] = *req.TopK
	}
	if req.Threshold != nil {
		config["threshold"] = *req.Threshold
	}
	result, err := h.RagSrv.Retrieve(c, req.Query, req.KbIds, &config)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}
	if result.Status {
		r := models.RetrieveResult{
			Query: result.Query,
			KbIds: result.KbIds,
			RetrieveAt: result.RetrieveAt,
			RetrievedCount: len(*result.RetrievedAssets),
			RetrievedAssets: result.RetrievedAssets,
			Config: result.Config,
		}
		return c.Status(fiber.StatusOK).JSON(r)
	} else {
		err = errors.New(*result.Error)
	}
	return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
}

// FastGraphRAG Asset API V1
//
//	@Summary	Get Assets
//	@Description	Process getting assets from knowledge base
//	@Tags	FastGraph RAG API
//	@Accept	json
//	@Produce	json
//	@Param	data	path	models.GetAssetsRequest	true	"Get Assets Request"
//  @Param  query	query	models.GetAssetsQueryParams	false	"Query Parameters"
//	@Success	200	{object}	models.AssetsResponse
//	@Failure	400	{object}	models.ErrorResponse
//	@Failure	500	{object}	models.ErrorResponse
//	@Router	/fastgraph/assets/{kb_id}/{asset_type} [get]
func (h *FastGraphRAGHandler) GetAssets(c fiber.Ctx) error {
	logger := am.GetLogger(h)
	params := models.GetAssetsRequest{}
	if err := c.Bind().All(&params); err != nil {
		logger.ErrorContext(c, "failed to bind parameters", slog.Any("error", err))
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}
	query := models.GetAssetsQueryParams{}
	if err := c.Bind().Query(&query); err != nil {
		logger.ErrorContext(c, "failed to bind query parameters", slog.Any("error", err))
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}
	result, err := h.RagSrv.GetAssets(c, params.KbId, params.AssetType, query.AssetIds, query.Version, query.Label)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}
	return c.Status(fiber.StatusOK).JSON(models.AssetsResponse{
		KbId: params.KbId,
		AssetsCount: len(result),
		Assets: result,
	})
}

// FastGraphRAG Asset By KbId API V1
//
//	@Summary	Get Assets By KbId
//	@Description	Process getting assets by kb_id from knowledge base
//	@Tags	FastGraph RAG API
//	@Accept	json
//	@Produce	json
//	@Param		data	path	models.GetAssetsRequest	true	"Get Assets Request"
//	@Success	200	{object}	models.AssetsResponse
//	@Failure	400	{object}	models.ErrorResponse
//	@Failure	500	{object}	models.ErrorResponse
//	@Router	/fastgraph/assets/{kb_id} [get]
func (h *FastGraphRAGHandler) GetAssetsByKbId(c fiber.Ctx) error {
	logger := am.GetLogger(h)
	params := models.GetAssetsRequest{}
	if err := c.Bind().All(&params); err != nil {
		logger.ErrorContext(c, "failed to bind parameters", slog.Any("error", err))
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}
	result, err := h.RagSrv.GetAssetsByKbId(c, params.KbId)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}
	return c.Status(fiber.StatusOK).JSON(models.AssetsResponse{
		KbId: params.KbId,
		AssetsCount: len(result),
		Assets: result,
	})
}

// FastGraphRAG Asset Versions By KbId API V1
//
//	@Summary	Get Asset Versions
//	@Description	Process getting asset versions from knowledge base
//	@Tags	FastGraph RAG API
//	@Accept	json
//	@Produce	json
//	@Param		data	path	models.GetAssetsRequest	true	"Get Assets Request"
//	@Success	200	{object}	[]int
//	@Failure	400	{object}	models.ErrorResponse
//	@Failure	500	{object}	models.ErrorResponse
//	@Router	/fastgraph/versions/{kb_id} [get]
func (h *FastGraphRAGHandler) GetVersions(c fiber.Ctx) error {
	logger := am.GetLogger(h)
	params := models.GetAssetsRequest{}
	if err := c.Bind().All(&params); err != nil {
		logger.ErrorContext(c, "failed to bind parameters", slog.Any("error", err))
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}
	result, err := h.RagSrv.GetVersions(c, params.KbId)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}
	return c.Status(fiber.StatusOK).JSON(result)
}

// FastGraphRAG Asset By Refs API V1
//
//	@Summary	Get Assets By Refs
//	@Description	Process getting assets by refs from knowledge base
//	@Tags	FastGraph RAG API
//	@Accept	json
//	@Produce	json
//	@Param	data	path	models.GetAssetsRequest	true	"Get Assets Request"
//  @Param  query	query	models.GetAssetsByRefsQueryParams	false	"Query Parameters"
//	@Success	200	{object}	models.AssetsResponse
//	@Failure	400	{object}	models.ErrorResponse
//	@Failure	500	{object}	models.ErrorResponse
//	@Router	/fastgraph/assets/refs/{kb_id}/{asset_type} [get]
func (h *FastGraphRAGHandler) GetAssetsByRefs(c fiber.Ctx) error {
	logger := am.GetLogger(h)
	params := models.GetAssetsRequest{}
	if err := c.Bind().All(&params); err != nil {
		logger.ErrorContext(c, "failed to bind parameters", slog.Any("error", err))
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}
	query := models.GetAssetsByRefsQueryParams{}
	if err := c.Bind().Query(&query); err != nil {
		logger.ErrorContext(c, "failed to bind query parameters", slog.Any("error", err))
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}
	result, err := h.RagSrv.GetAssetsByRefs(c, params.KbId, params.AssetType, &query.RefIds, &query.RefTypes, query.Version, query.Label)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}
	return c.Status(fiber.StatusOK).JSON(models.AssetsResponse{
		KbId: params.KbId,
		AssetsCount: len(result),
		Assets: result,
	})
}

// FastGraphRAG Delete KbId API V1
//
//	@Summary	Delete KbId
//	@Description	Process deleting kb_id from knowledge base
//	@Tags	FastGraph RAG API
//	@Accept	json
//	@Produce	json
//	@Param		kb_id	path	string	true	"KbId"
//	@Success	200	{object}	int
//	@Failure	400	{object}	models.ErrorResponse
//	@Failure	500	{object}	models.ErrorResponse
//	@Router	/fastgraph/{kb_id} [delete]
func (h *FastGraphRAGHandler) DeleteKbId(c fiber.Ctx) error {
	logger := am.GetLogger(h)
	kbId := c.Params("kb_id")
	if kbId == "" {
		logger.ErrorContext(c, "kb_id is required")
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "kb_id is required"})
	}
	total, err := h.RagSrv.DeleteAssetsByKbId(c, kbId)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}
	return c.Status(fiber.StatusOK).JSON(total)
}