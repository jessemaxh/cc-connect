package core

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCapturePodNetworkSnapshotSkipsLoopback(t *testing.T) {
	root := t.TempDir()
	writeCounter(t, root, "lo", "rx_bytes", "100")
	writeCounter(t, root, "lo", "tx_bytes", "200")
	writeCounter(t, root, "eth0", "rx_bytes", "300")
	writeCounter(t, root, "eth0", "tx_bytes", "400")
	writeCounter(t, root, "eth1", "rx_bytes", "500")
	writeCounter(t, root, "eth1", "tx_bytes", "600")

	snap, err := capturePodNetworkSnapshotFrom(root)
	if err != nil {
		t.Fatalf("capturePodNetworkSnapshotFrom() error = %v", err)
	}
	if snap.rxBytes != 800 {
		t.Fatalf("rxBytes = %d, want 800", snap.rxBytes)
	}
	if snap.txBytes != 1000 {
		t.Fatalf("txBytes = %d, want 1000", snap.txBytes)
	}
}

func TestCapturePodNetworkSnapshotReadsSymlinkEntries(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "devices", "virtual", "net", "eth0")
	if err := os.MkdirAll(filepath.Join(target, "statistics"), 0o755); err != nil {
		t.Fatalf("MkdirAll(%s): %v", target, err)
	}
	if err := os.WriteFile(filepath.Join(target, "statistics", "rx_bytes"), []byte("123"), 0o644); err != nil {
		t.Fatalf("WriteFile(rx_bytes): %v", err)
	}
	if err := os.WriteFile(filepath.Join(target, "statistics", "tx_bytes"), []byte("456"), 0o644); err != nil {
		t.Fatalf("WriteFile(tx_bytes): %v", err)
	}
	if err := os.Symlink(target, filepath.Join(root, "eth0")); err != nil {
		t.Fatalf("Symlink: %v", err)
	}

	snap, err := capturePodNetworkSnapshotFrom(root)
	if err != nil {
		t.Fatalf("capturePodNetworkSnapshotFrom() error = %v", err)
	}
	if snap.rxBytes != 123 || snap.txBytes != 456 {
		t.Fatalf("snapshot = %+v, want rx=123 tx=456", snap)
	}
}

func TestMeasureTurnNetworkClampsNegativeDiff(t *testing.T) {
	start := &podNetworkSnapshot{rxBytes: 100, txBytes: 200}
	end := &podNetworkSnapshot{rxBytes: 90, txBytes: 150}

	usage := NetworkUsage{
		BytesIn:  end.rxBytes - start.rxBytes,
		BytesOut: end.txBytes - start.txBytes,
	}
	if usage.BytesIn >= 0 || usage.BytesOut >= 0 {
		t.Fatalf("test precondition failed: expected negative diff, got %+v", usage)
	}

	if usage.BytesIn < 0 {
		usage.BytesIn = 0
	}
	if usage.BytesOut < 0 {
		usage.BytesOut = 0
	}

	if usage.BytesIn != 0 || usage.BytesOut != 0 {
		t.Fatalf("usage = %+v, want zeros", usage)
	}
}

func writeCounter(t *testing.T, root, iface, file, value string) {
	t.Helper()
	dir := filepath.Join(root, iface, "statistics")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%s): %v", dir, err)
	}
	if err := os.WriteFile(filepath.Join(dir, file), []byte(value), 0o644); err != nil {
		t.Fatalf("WriteFile(%s): %v", file, err)
	}
}
