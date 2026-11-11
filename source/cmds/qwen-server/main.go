// Command qwen-server runs a minimal OpenAI-compatible inference server on top
// of the Qwen3.8 Vulkan/CPU pipeline.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/inference"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/openaiserver"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	modelPath := flag.String("model", os.Getenv("QWEN_SERVER_MODEL"), "model GGUF path (required)")
	host := flag.String("host", "127.0.0.1", "listen host")
	port := flag.Int("port", 8080, "listen port")
	modelID := flag.String("model-id", "", "model id exposed via the API (default: model file name)")
	apiKey := flag.String("api-key", os.Getenv("QWEN_SERVER_API_KEY"), "optional Bearer API key")
	device := flag.String("device", "auto", "compute device: auto|cpu|vulkan")
	contextLen := flag.Int("context", 0, "context length override (0 = GGUF value)")
	maxTokens := flag.Int("default-max-tokens", 1024, "default max_tokens when the request omits it")
	temperature := flag.Float64("default-temperature", 1.0, "default temperature")
	topP := flag.Float64("default-top-p", 1.0, "default top_p")
	topK := flag.Int("default-top-k", 0, "default top_k (0 = disabled)")
	logLevel := flag.String("log-level", "info", "quiet|info|debug")
	flag.Parse()

	if *modelPath == "" {
		return fmt.Errorf("--model <path> is required")
	}

	logger := log.New(os.Stderr, "qwen-server: ", log.LstdFlags)
	logger.Printf("loading %s", *modelPath)
	engine, err := inference.Load(inference.LoadOptions{
		Path:          *modelPath,
		Device:        *device,
		ContextLength: *contextLen,
		Log:           os.Stderr,
		LogLevel:      *logLevel,
	})
	if err != nil {
		return err
	}
	defer engine.Close()

	id := *modelID
	if id == "" {
		base := filepath.Base(*modelPath)
		id = strings.TrimSuffix(base, filepath.Ext(base))
	}

	handler := openaiserver.New(engine, openaiserver.Options{
		ModelID:            id,
		APIKey:             *apiKey,
		DefaultMaxTokens:   *maxTokens,
		DefaultTemperature: *temperature,
		DefaultTopP:        *topP,
		DefaultTopK:        *topK,
		Logger:             logger,
	})

	addr := net.JoinHostPort(*host, strconv.Itoa(*port))
	srv := &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()

	logger.Printf("listening on http://%s (model=%s, context=%d)", addr, id, engine.ContextLength())
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return err
	}
	return nil
}
