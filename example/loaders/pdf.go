// Since the default HttpFileLoader does not check for file extension because of simplicity implementation and flexibility, 
// we create a custom HttpPDFLoader that only accepts PDF files based on the file extension. 
// This ensures that the PDF Converter is only applied to PDF files loaded from HTTP URLs.
// You can implement similar custom loaders for other file types (e.g. Word, Excel) if you want to use different converters for them.

package loaders

import (
	"context"
	"strings"

	am "github.com/manageai-inet/agentic-assets"
	amk "github.com/manageai-inet/agentic-assets/knowledge_extractor"
)

// This is a custom KnowledgeLoader that only loads PDF files from HTTP URLs. It delegates the actual loading to an underlying HttpFileLoader.
type HttpPDFLoader struct {
	httpLoader *amk.HttpFileLoader
	am.LoggingCapacity
}

func NewHttpPDFLoader(auth *amk.HttpAuthentication) *HttpPDFLoader {
	return &HttpPDFLoader{
		httpLoader: amk.NewHttpFileLoader(auth),
		LoggingCapacity: *am.GetDefaultLoggingCapacity(),
	}
}

// `String` method just for logging and debugging purpose, but you need to implement it.
func (p *HttpPDFLoader) String() string {
	return "HttpPDFLoader"
}

func (p *HttpPDFLoader) IsApplicable(sourceName, sourceUrl string, metadata *map[string]any) bool {
	// HttpFileLoader will check prefix (e.g. http://, https://) for you. You don't need to check it again here.
	if !p.httpLoader.IsApplicable(sourceName, sourceUrl, metadata) {
		return false
	}
	// Check file extension is .pdf (case-insensitive) based on sourceName, not sourceUrl, because some URLs may not have file extension but still return PDF content.
	sourceSplited := strings.Split(sourceName, ".")
	sourceExt := strings.ToLower(sourceSplited[len(sourceSplited)-1])
	return sourceExt == "pdf"
}

func (p *HttpPDFLoader) Load(ctx context.Context, sourceName, sourceUrl string, metadata *map[string]any) ([]byte, error) {
	// Just delegate to the underlying HttpFileLoader to load the file content or you can add some custom logic here if needed.
	return p.httpLoader.Load(ctx, sourceName, sourceUrl, metadata)
}

// You can implement similar custom loaders for other file types (e.g. Word, Excel) if you want to use different converters for them.
type LocalPDFLoader struct {
	localLoader *amk.LocalFileLoader
	am.LoggingCapacity
}

func NewLocalPDFLoader(rootPath string) *LocalPDFLoader {
	return &LocalPDFLoader{localLoader: amk.NewLocalFileLoader(rootPath), LoggingCapacity: *am.GetDefaultLoggingCapacity()}
}

func (p *LocalPDFLoader) String() string {
	return "LocalPDFLoader"
}

func (p *LocalPDFLoader) IsApplicable(sourceName, sourceUrl string, metadata *map[string]any) bool {
	if !p.localLoader.IsApplicable(sourceName, sourceUrl, metadata) {
		return false
	}
	// Check file extension is .pdf (case-insensitive) based on sourceName, not sourceUrl, because some URLs may not have file extension but still return PDF content.
	sourceSplited := strings.Split(sourceName, ".")
	sourceExt := strings.ToLower(sourceSplited[len(sourceSplited)-1])
	return sourceExt == "pdf"
}

func (p *LocalPDFLoader) Load(ctx context.Context, sourceName, sourceUrl string, metadata *map[string]any) ([]byte, error) {
	return p.localLoader.Load(ctx, sourceName, sourceUrl, metadata)
}