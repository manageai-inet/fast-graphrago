package loaders

import (
	"context"
	"strings"

	am "github.com/manageai-inet/agentic-assets"
	amk "github.com/manageai-inet/agentic-assets/knowledge_extractor"
)

// This is just example implementation, so it only supports loading image files with .jpg and .png extensions.
var ImageFileExtensions = []string{"jpg", "png"}

// This is a custom KnowledgeLoader that only loads image files from HTTP URLs. It delegates the actual loading to an underlying HttpFileLoader.
type HttpImageLoader struct {
	httpLoader *amk.HttpFileLoader
	am.LoggingCapacity
}

func NewHttpImageLoader(auth *amk.HttpAuthentication) *HttpImageLoader {
	return &HttpImageLoader{
		httpLoader: amk.NewHttpFileLoader(auth),
		LoggingCapacity: *am.GetDefaultLoggingCapacity(),
	}
}

// `String` method just for logging and debugging purpose, but you need to implement it.
func (l *HttpImageLoader) String() string {
	return "HttpImageLoader"
}

func (l *HttpImageLoader) IsApplicable(sourceName, sourceUrl string, metadata *map[string]any) bool {
	// HttpFileLoader will check prefix (e.g. http://, https://) for you. You don't need to check it again here.
	if !l.httpLoader.IsApplicable(sourceName, sourceUrl, metadata) {
		return false
	}
	// Check file extension is in the list of image file extensions (case-insensitive) based on sourceName, not sourceUrl, because some URLs may not have file extension but still return image content.
	sourceSplited := strings.Split(sourceName, ".")
	sourceExt := strings.ToLower(sourceSplited[len(sourceSplited)-1])
	for _, ext := range ImageFileExtensions {
		if sourceExt == ext {
			return true
		}
	}
	return false
}

func (l *HttpImageLoader) Load(ctx context.Context, sourceName, sourceUrl string, metadata *map[string]any) ([]byte, error) {
	// Just delegate to the underlying HttpFileLoader to load the file content or you can add some custom logic here if needed.
	return l.httpLoader.Load(ctx, sourceName, sourceUrl, metadata)
}

// You can implement similar custom loaders for other file types (e.g. Word, Excel) if you want to use different converters for them.

type LocalImageLoader struct {
	localLoader *amk.LocalFileLoader
	am.LoggingCapacity
}

func NewLocalImageLoader(rootPath string) *LocalImageLoader {
	return &LocalImageLoader{localLoader: amk.NewLocalFileLoader(rootPath), LoggingCapacity: *am.GetDefaultLoggingCapacity()}
}

func (l *LocalImageLoader) String() string {
	return "LocalImageLoader"
}

func (l *LocalImageLoader) IsApplicable(sourceName, sourceUrl string, metadata *map[string]any) bool {
	// Check file extension is in the list of image file extensions (case-insensitive) based on sourceName, not sourceUrl, because some URLs may not have file extension but still return image content.
	sourceSplited := strings.Split(sourceName, ".")
	sourceExt := strings.ToLower(sourceSplited[len(sourceSplited)-1])
	for _, ext := range ImageFileExtensions {
		if sourceExt == ext {
			return true
		}
	}
	return false
}

func (l *LocalImageLoader) Load(ctx context.Context, sourceName, sourceUrl string, metadata *map[string]any) ([]byte, error) {
	return l.localLoader.Load(ctx, sourceName, sourceUrl, metadata)
}