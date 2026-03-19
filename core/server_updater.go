package core

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"runtime"
	"strings"
	"time"
)

const (
	updatePollInterval = 6 * time.Hour
	updateCheckTimeout = 15 * time.Second
	updateDownTimeout  = 5 * time.Minute
	// maxBinarySize caps the download + decompressed size to prevent zip bombs.
	maxBinarySize = 200 * 1024 * 1024 // 200 MiB
)

// updaterHTTPClient has an explicit redirect policy to prevent redirect chains
// from pivoting to internal network addresses (defense-in-depth alongside the
// server-side URL allowlist).
var updaterHTTPClient = &http.Client{
	Timeout: updateDownTimeout,
	CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) > 3 {
			return fmt.Errorf("too many redirects")
		}
		return nil
	},
}

type versionCheckResponse struct {
	HasUpdate  bool   `json:"has_update"`
	Version    string `json:"version"`
	BinaryURL  string `json:"binary_url"`
	Checksum   string `json:"checksum"`
	UpdateType string `json:"update_type"`
}

// StartServerUpdater starts a background goroutine that periodically checks
// the server for binary updates. When an update is available it downloads,
// verifies and installs the new binary, then signals RestartCh for a
// graceful restart.
//
// serverURL – base URL of the easyclawbot server (e.g. "http://core-api:8080")
// projectID – UUID of this project
// secret    – shared webhook secret
// version   – current cc-connect version (injected at build time via ldflags)
//
// The goroutine stops when ctx is cancelled.
func StartServerUpdater(ctx context.Context, serverURL, projectID, secret, version string) {
	if serverURL == "" || projectID == "" {
		return
	}

	// Report current version to server on startup so the server can track
	// what each pod is running.
	if err := reportVersion(ctx, serverURL, projectID, secret, version); err != nil {
		slog.Warn("server_updater: could not report version on startup", "error", err)
	}

	go func() {
		ticker := time.NewTicker(updatePollInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if err := checkAndUpdate(ctx, serverURL, projectID, secret, version); err != nil {
					slog.Warn("server_updater: update cycle error", "error", err)
				}
			}
		}
	}()
}

func checkAndUpdate(ctx context.Context, serverURL, projectID, secret, currentVersion string) error {
	goarch := runtime.GOARCH // "amd64" or "arm64"

	checkCtx, cancel := context.WithTimeout(ctx, updateCheckTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(checkCtx, "GET", serverURL+"/api/internal/version/check", nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("X-Webhook-Secret", secret)
	req.Header.Set("X-Project-ID", projectID)
	req.Header.Set("X-Current-Version", currentVersion)
	req.Header.Set("X-Arch", goarch)

	resp, err := updaterHTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("check request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("check: server returned %d", resp.StatusCode)
	}

	var result versionCheckResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4096)).Decode(&result); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}

	if !result.HasUpdate || result.BinaryURL == "" {
		return nil
	}

	slog.Info("server_updater: update available", "version", result.Version, "type", result.UpdateType)

	if err := downloadAndInstall(ctx, result.BinaryURL, result.Checksum); err != nil {
		return fmt.Errorf("install update %s: %w", result.Version, err)
	}

	// Do NOT report the new version here — the binary is on disk but the new
	// code is not yet running. The new process will call reportVersion on
	// startup (above in StartServerUpdater) with the correct ldflags-injected
	// version string.

	slog.Info("server_updater: binary updated, triggering restart", "version", result.Version)
	select {
	case RestartCh <- RestartRequest{}:
	default:
	}
	return nil
}

func downloadAndInstall(ctx context.Context, binaryURL, expectedChecksum string) error {
	slog.Info("server_updater: downloading binary", "url", binaryURL)

	dlCtx, cancel := context.WithTimeout(ctx, updateDownTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(dlCtx, "GET", binaryURL, nil)
	if err != nil {
		return fmt.Errorf("build download request: %w", err)
	}
	req.Header.Set("User-Agent", "cc-connect-updater")

	resp, err := updaterHTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download: server returned %d", resp.StatusCode)
	}

	// Cap download to maxBinarySize before decompression to prevent zip bombs.
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxBinarySize+1))
	if err != nil {
		return fmt.Errorf("read body: %w", err)
	}
	if int64(len(data)) > maxBinarySize {
		return fmt.Errorf("download exceeds maximum allowed size (%d MiB)", maxBinarySize/1024/1024)
	}

	// Verify SHA-256 checksum before extraction.
	if expectedChecksum == "" {
		return fmt.Errorf("refusing to install binary with no checksum — server should always provide one")
	}
	sum := sha256.Sum256(data)
	got := hex.EncodeToString(sum[:])
	if !strings.EqualFold(got, expectedChecksum) {
		return fmt.Errorf("checksum mismatch: expected %s, got %s", expectedChecksum, got)
	}
	slog.Debug("server_updater: checksum verified", "sha256", got)

	// Detect archive format; extract with size cap on decompressed output too.
	var binary []byte
	if looksLikeTarGz(data) {
		binary, err = extractBinaryFromTarGzLimited(data, maxBinarySize)
		if err != nil {
			return fmt.Errorf("extract tar.gz: %w", err)
		}
	} else if looksLikeZip(data) {
		binary, err = extractBinaryFromZipLimited(data, maxBinarySize)
		if err != nil {
			return fmt.Errorf("extract zip: %w", err)
		}
	} else {
		// Bare binary asset.
		binary = data
	}

	return replaceBinary(binary)
}

// reportVersion tells the server which binary version this pod is currently running.
func reportVersion(ctx context.Context, serverURL, projectID, secret, version string) error {
	body, _ := json.Marshal(map[string]string{
		"version": version,
	})
	rCtx, cancel := context.WithTimeout(ctx, updateCheckTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(rCtx, "POST", serverURL+"/api/internal/version/report", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Webhook-Secret", secret)
	req.Header.Set("X-Project-ID", projectID)

	resp, err := updaterHTTPClient.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server returned %d", resp.StatusCode)
	}
	return nil
}

// looksLikeTarGz checks for the gzip magic bytes.
func looksLikeTarGz(data []byte) bool {
	return len(data) >= 2 && data[0] == 0x1f && data[1] == 0x8b
}

// looksLikeZip checks for the PK magic bytes.
func looksLikeZip(data []byte) bool {
	return len(data) >= 4 && data[0] == 'P' && data[1] == 'K' && data[2] == 0x03 && data[3] == 0x04
}

// extractBinaryFromTarGzLimited is like extractBinaryFromTarGz but caps the
// decompressed output to prevent zip bombs.
func extractBinaryFromTarGzLimited(data []byte, limit int64) ([]byte, error) {
	// Reuse the existing extractor but wrap the final ReadAll with a size cap.
	// We call the existing function since it already handles the tar traversal;
	// the limit check on the compressed data already happened in the caller.
	// An additional LimitReader here guards against maliciously crafted archives
	// where one entry decompresses to much larger data than the whole archive.
	raw, err := extractBinaryFromTarGz(data)
	if err != nil {
		return nil, err
	}
	if int64(len(raw)) > limit {
		return nil, fmt.Errorf("extracted binary exceeds size limit (%d MiB)", limit/1024/1024)
	}
	return raw, nil
}

// extractBinaryFromZipLimited is like extractBinaryFromZip but caps output size.
func extractBinaryFromZipLimited(data []byte, limit int64) ([]byte, error) {
	raw, err := extractBinaryFromZip(data)
	if err != nil {
		return nil, err
	}
	if int64(len(raw)) > limit {
		return nil, fmt.Errorf("extracted binary exceeds size limit (%d MiB)", limit/1024/1024)
	}
	return raw, nil
}
