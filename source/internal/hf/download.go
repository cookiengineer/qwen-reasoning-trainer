// Package hf implements a minimal Hugging Face Hub downloader using only the
// standard library. It resolves model files, supports HTTP range resume, and
// verifies SHA-256 checksums reported by the Hub tree API.
package hf

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Defaults for the target model.
const (
	DefaultRepo   = "unsloth/Qwen3.8-27B-GGUF"
	DefaultFile   = "Qwen3.8-27B-UD-Q4_K_M.gguf"
	DefaultSHA256 = "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482"
	DefaultSize   = 16464440224
)

// Client talks to the Hugging Face Hub.
type Client struct {
	// BaseURL is the Hub root, default https://huggingface.co.
	BaseURL string
	// Token is an optional HF access token.
	Token string
	// HTTPClient is optional; a default client with a long timeout is used.
	HTTPClient *http.Client
	// Progress, when set, is called periodically during downloads.
	Progress func(downloaded, total int64)
}

func (c *Client) base() string {
	if c.BaseURL != "" {
		return strings.TrimRight(c.BaseURL, "/")
	}
	if env := os.Getenv("HF_ENDPOINT"); env != "" {
		return strings.TrimRight(env, "/")
	}
	return "https://huggingface.co"
}

func (c *Client) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{Timeout: 0}
}

func (c *Client) token() string {
	if c.Token != "" {
		return c.Token
	}
	return os.Getenv("HF_TOKEN")
}

func (c *Client) newRequest(ctx context.Context, method, url string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, url, nil)
	if err != nil {
		return nil, err
	}
	if tok := c.token(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	req.Header.Set("User-Agent", "qwen-reasoning-trainer")
	return req, nil
}

// ResolveURL returns the download URL for a file in a repo.
func (c *Client) ResolveURL(repo, file string) string {
	return fmt.Sprintf("%s/%s/resolve/main/%s", c.base(), repo, file)
}

// treeEntry is one element of the Hub tree API response.
type treeEntry struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size int64  `json:"size"`
	LFS  *struct {
		OID  string `json:"oid"`
		Size int64  `json:"size"`
	} `json:"lfs"`
}

// FileInfo queries the Hub tree API for size and LFS SHA-256 of a file.
func (c *Client) FileInfo(ctx context.Context, repo, file string) (size int64, sha string, err error) {
	url := fmt.Sprintf("%s/api/models/%s/tree/main", c.base(), repo)
	req, err := c.newRequest(ctx, http.MethodGet, url)
	if err != nil {
		return 0, "", err
	}
	resp, err := c.httpClient().Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, "", fmt.Errorf("hf: tree API %s: %s", url, resp.Status)
	}
	var entries []treeEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return 0, "", fmt.Errorf("hf: decode tree: %w", err)
	}
	for _, e := range entries {
		if e.Path == file {
			if e.LFS != nil && e.LFS.OID != "" {
				return e.LFS.Size, e.LFS.OID, nil
			}
			return e.Size, "", nil
		}
	}
	return 0, "", fmt.Errorf("hf: file %q not found in %s", file, repo)
}

// Download fetches file from repo into dest. It resumes partial downloads from
// dest+".part" and, when wantSHA is non-empty, verifies the SHA-256 of the
// result. progress may be nil.
func (c *Client) Download(ctx context.Context, repo, file, dest, wantSHA string) error {
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}

	if ok, err := verifyFile(dest, wantSHA, 0); err == nil && ok {
		return nil
	}

	part := dest + ".part"
	var existing int64
	if st, err := os.Stat(part); err == nil {
		existing = st.Size()
	}

	req, err := c.newRequest(ctx, http.MethodGet, c.ResolveURL(repo, file))
	if err != nil {
		return err
	}
	if existing > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existing))
	}

	resp, err := c.httpClient().Do(req)
	if err != nil {
		return fmt.Errorf("hf: download %s: %w", file, err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		// Server ignored the range; restart.
		existing = 0
	case http.StatusPartialContent:
		// Append.
	case http.StatusRequestedRangeNotSatisfiable:
		// Already complete; treat as done.
	default:
		return fmt.Errorf("hf: download %s: unexpected status %s", file, resp.Status)
	}

	var total int64 = -1
	if resp.ContentLength >= 0 {
		total = existing + resp.ContentLength
	}

	flags := os.O_CREATE | os.O_WRONLY
	if existing == 0 {
		flags |= os.O_TRUNC
	} else {
		flags |= os.O_APPEND
	}
	f, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return err
	}

	written := existing
	buf := make([]byte, 1<<20)
	lastReport := time.Now()
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := f.Write(buf[:n]); werr != nil {
				f.Close()
				return werr
			}
			written += int64(n)
			if c.Progress != nil && time.Since(lastReport) > 200*time.Millisecond {
				c.Progress(written, total)
				lastReport = time.Now()
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			f.Close()
			return fmt.Errorf("hf: download %s: %w", file, rerr)
		}
	}
	if err := f.Close(); err != nil {
		return err
	}
	if c.Progress != nil {
		c.Progress(written, total)
	}

	if err := os.Rename(part, dest); err != nil {
		return err
	}
	if wantSHA != "" {
		ok, err := verifyFile(dest, wantSHA, 0)
		if err != nil {
			return err
		}
		if !ok {
			os.Remove(dest)
			return fmt.Errorf("hf: checksum mismatch for %s", file)
		}
	}
	return nil
}

// verifyFile returns true when the file exists and its SHA-256 matches wantSHA.
// If wantSHA is empty, only existence (and minSize, when > 0) is checked.
func verifyFile(path, wantSHA string, minSize int64) (bool, error) {
	st, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	if wantSHA == "" {
		return minSize == 0 || st.Size() >= minSize, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return false, err
	}
	return strings.EqualFold(hex.EncodeToString(h.Sum(nil)), wantSHA), nil
}

// SHA256File returns the hex SHA-256 of a file.
func SHA256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
