package cli

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"strconv"
	"strings"
)

// logMemory writes a one-line memory snapshot (process RSS plus GPU GTT/VRAM
// usage) to w. A GTT figure that grows without bound points at GPU buffers
// spilling into system RAM, which is what triggers the OOM killer on a
// swap-less host.
func logMemory(w io.Writer, label string) {
	rss := residentBytes()
	gtt, vram := gpuMemoryBytes()
	fmt.Fprintf(w, "memory[%s]: rss=%s gtt=%s vram=%s\n", label, humanBytes(rss), humanBytes(gtt), humanBytes(vram))
}

// applyMemoryLimit installs a Go heap soft limit derived from a fraction of
// physical RAM when the user has not set GOMEMLIMIT. It makes runaway heap
// growth turn into GC pressure instead of an OOM kill. It returns the applied
// limit in bytes, or 0 when no limit was installed (GOMEMLIMIT set, or unknown
// RAM).
func applyMemoryLimit() int64 {
	if os.Getenv("GOMEMLIMIT") != "" {
		return 0
	}
	total := totalRAMBytes()
	if total <= 0 {
		return 0
	}
	lim := int64(float64(total) * 0.6)
	debug.SetMemoryLimit(lim)
	return lim
}

// residentBytes returns the process resident set size in bytes, read from
// /proc/self/statm. It returns 0 on failure so logging never affects a run.
func residentBytes() int64 {
	data, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0
	}
	fields := strings.Fields(string(data))
	if len(fields) < 2 {
		return 0
	}
	pages, err := strconv.ParseInt(fields[1], 10, 64)
	if err != nil || pages < 0 {
		return 0
	}
	return pages * int64(os.Getpagesize())
}

// totalRAMBytes returns MemTotal from /proc/meminfo, or 0 if unavailable.
func totalRAMBytes() int64 {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "MemTotal:") {
			fields := strings.Fields(line)
			if len(fields) < 2 {
				return 0
			}
			kb, err := strconv.ParseInt(fields[1], 10, 64)
			if err != nil {
				return 0
			}
			return kb * 1024
		}
	}
	return 0
}

// gpuMemoryBytes reports the largest GTT (system RAM the GPU may pin) and VRAM
// in use across DRM cards, in bytes. Values are best-effort; 0 means unknown.
func gpuMemoryBytes() (gtt, vram int64) {
	cards, _ := filepath.Glob("/sys/class/drm/card*/device")
	for _, dir := range cards {
		if v := readIntFile(filepath.Join(dir, "mem_info_gtt_used")); v > gtt {
			gtt = v
		}
		if v := readIntFile(filepath.Join(dir, "mem_info_vram_used")); v > vram {
			vram = v
		}
	}
	return gtt, vram
}

func readIntFile(path string) int64 {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	v, err := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
	if err != nil || v < 0 {
		return 0
	}
	return v
}
