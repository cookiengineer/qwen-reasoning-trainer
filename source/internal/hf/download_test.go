package hf

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

func testServer(t *testing.T, repo, file string, content []byte) (*httptest.Server, *int32) {
	t.Helper()
	var rangeSeen int32
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/"+repo+"/tree/main", func(w http.ResponseWriter, r *http.Request) {
		sum := sha256.Sum256(content)
		resp := []map[string]any{{
			"type": "file",
			"path": file,
			"size": len(content),
			"lfs":  map[string]any{"oid": hex.EncodeToString(sum[:]), "size": len(content)},
		}}
		json.NewEncoder(w).Encode(resp)
	})
	mux.HandleFunc("/"+repo+"/resolve/main/"+file, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") != "" {
			atomic.StoreInt32(&rangeSeen, 1)
		}
		http.ServeContent(w, r, file, time.Time{}, bytes.NewReader(content))
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, &rangeSeen
}

func TestFileInfo(t *testing.T) {
	content := []byte("hello world")
	srv, _ := testServer(t, "owner/repo", "model.gguf", content)
	c := &Client{BaseURL: srv.URL}
	size, sha, err := c.FileInfo(context.Background(), "owner/repo", "model.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if size != int64(len(content)) {
		t.Errorf("size = %d, want %d", size, len(content))
	}
	sum := sha256.Sum256(content)
	if sha != hex.EncodeToString(sum[:]) {
		t.Errorf("sha = %s", sha)
	}
}

func TestDownloadFresh(t *testing.T) {
	content := []byte("the quick brown fox jumps over the lazy dog")
	srv, _ := testServer(t, "owner/repo", "model.gguf", content)
	sum := sha256.Sum256(content)
	c := &Client{BaseURL: srv.URL}
	dest := filepath.Join(t.TempDir(), "model.gguf")
	if err := c.Download(context.Background(), "owner/repo", "model.gguf", dest, hex.EncodeToString(sum[:])); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(content) {
		t.Errorf("content = %q, want %q", got, content)
	}
}

func TestDownloadResume(t *testing.T) {
	content := []byte("0123456789abcdefghijklmnopqrstuvwxyz")
	srv, rangeSeen := testServer(t, "owner/repo", "model.gguf", content)
	sum := sha256.Sum256(content)
	dir := t.TempDir()
	dest := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(dest+".part", content[:10], 0o644); err != nil {
		t.Fatal(err)
	}
	c := &Client{BaseURL: srv.URL}
	if err := c.Download(context.Background(), "owner/repo", "model.gguf", dest, hex.EncodeToString(sum[:])); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(dest)
	if string(got) != string(content) {
		t.Errorf("content = %q, want %q", got, content)
	}
	if atomic.LoadInt32(rangeSeen) != 1 {
		t.Error("expected a Range request during resume")
	}
	if _, err := os.Stat(dest + ".part"); !os.IsNotExist(err) {
		t.Error("part file should be renamed away")
	}
}

func TestDownloadChecksumMismatch(t *testing.T) {
	content := []byte("data that will not match the checksum")
	srv, _ := testServer(t, "owner/repo", "model.gguf", content)
	c := &Client{BaseURL: srv.URL}
	dest := filepath.Join(t.TempDir(), "model.gguf")
	err := c.Download(context.Background(), "owner/repo", "model.gguf", dest, "deadbeef")
	if err == nil {
		t.Fatal("expected checksum mismatch error")
	}
	if _, statErr := os.Stat(dest); !os.IsNotExist(statErr) {
		t.Error("corrupt file should be removed")
	}
}

func TestDownloadAlreadyPresent(t *testing.T) {
	content := []byte("already here")
	srv, rangeSeen := testServer(t, "owner/repo", "model.gguf", content)
	sum := sha256.Sum256(content)
	dest := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(dest, content, 0o644); err != nil {
		t.Fatal(err)
	}
	c := &Client{BaseURL: srv.URL}
	if err := c.Download(context.Background(), "owner/repo", "model.gguf", dest, hex.EncodeToString(sum[:])); err != nil {
		t.Fatal(err)
	}
	if atomic.LoadInt32(rangeSeen) != 0 {
		t.Error("no request should be made when the file already matches")
	}
}
