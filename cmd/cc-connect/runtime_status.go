package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
)

func runRuntimeStatus(args []string) {
	var dataDir string
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--data-dir":
			if i+1 < len(args) {
				i++
				dataDir = args[i]
			}
		case "--help", "-h":
			fmt.Println("Usage: cc-connect runtime-status [--data-dir <path>]")
			return
		}
	}

	sockPath := resolveSocketPath(dataDir)
	client := &http.Client{
		Transport: &http.Transport{
			DialContext: func(_ context.Context, _, _ string) (net.Conn, error) {
				return net.Dial("unix", sockPath)
			},
		},
	}

	resp, err := client.Get("http://unix/runtime/status")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: failed to connect: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		fmt.Fprintf(os.Stderr, "Error: %s\n", string(body))
		os.Exit(1)
	}

	var parsed map[string]any
	if err := json.Unmarshal(body, &parsed); err != nil {
		fmt.Fprintf(os.Stderr, "Error: invalid response: %v\n", err)
		os.Exit(1)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(parsed)
}
