package core

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const netStatsRoot = "/sys/class/net"

type NetworkUsage struct {
	BytesIn  int64
	BytesOut int64
}

type podNetworkSnapshot struct {
	rxBytes int64
	txBytes int64
}

func capturePodNetworkSnapshot() (*podNetworkSnapshot, error) {
	return capturePodNetworkSnapshotFrom(netStatsRoot)
}

func capturePodNetworkSnapshotFrom(root string) (*podNetworkSnapshot, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("read interfaces: %w", err)
	}

	var snap podNetworkSnapshot
	found := false
	for _, entry := range entries {
		name := entry.Name()
		if name == "lo" || strings.HasPrefix(name, "ifb") {
			continue
		}

		rx, err := readInt64File(filepath.Join(root, name, "statistics", "rx_bytes"))
		if err != nil {
			continue
		}
		tx, err := readInt64File(filepath.Join(root, name, "statistics", "tx_bytes"))
		if err != nil {
			continue
		}

		snap.rxBytes += rx
		snap.txBytes += tx
		found = true
	}

	if !found {
		return nil, fmt.Errorf("no pod network interface counters found")
	}
	return &snap, nil
}

func readInt64File(path string) (int64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	value, err := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
	if err != nil {
		return 0, fmt.Errorf("parse %s: %w", path, err)
	}
	return value, nil
}

func measureTurnNetwork(start *podNetworkSnapshot) NetworkUsage {
	if start == nil {
		return NetworkUsage{}
	}

	end, err := capturePodNetworkSnapshot()
	if err != nil || end == nil {
		return NetworkUsage{}
	}

	usage := NetworkUsage{
		BytesIn:  end.rxBytes - start.rxBytes,
		BytesOut: end.txBytes - start.txBytes,
	}
	if usage.BytesIn < 0 {
		usage.BytesIn = 0
	}
	if usage.BytesOut < 0 {
		usage.BytesOut = 0
	}
	return usage
}
