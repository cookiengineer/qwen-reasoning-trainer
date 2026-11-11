package inference

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// KV cache types supported by the server. Q4_K is the default; Q8_0 is the
// internal higher-quality alternative. There is intentionally no f16/f32 KV
// cache mode and no CLI flag for this choice.
var (
	DefaultKVCacheType     = quant.TypeQ4_K
	AlternativeKVCacheType = quant.TypeQ8_0
)

// ParseKVCacheType parses the supported KV cache types. Q4_K is the default;
// Q8_0 is the alternative.
func ParseKVCacheType(s string) (quant.Type, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "q4_k", "q4k":
		return DefaultKVCacheType, nil
	case "q8_0", "q80":
		return AlternativeKVCacheType, nil
	default:
		return 0, fmt.Errorf("invalid KV cache type %q (want q4_k or q8_0)", s)
	}
}

// ParseContextLength parses a context length such as "128k", "512K", or
// "524288". A k/K suffix means 1024 tokens.
func ParseContextLength(s string) (int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, fmt.Errorf("context length is empty")
	}
	mult := int64(1)
	last := s[len(s)-1]
	if last == 'k' || last == 'K' {
		mult = 1024
		s = strings.TrimSpace(s[:len(s)-1])
	}
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid context length %q", s)
	}
	if n <= 0 {
		return 0, fmt.Errorf("context length must be positive")
	}
	n *= mult
	if n > math.MaxInt32 {
		return 0, fmt.Errorf("context length %d is too large", n)
	}
	return int(n), nil
}

// EstimateKVBytes returns the K+V cache size for context tokens. Only
// full-attention layers own a KV cache; recurrent layers use fixed SSM state.
func EstimateKVBytes(cfg *modelcfg.Config, context int, typ quant.Type) uint64 {
	if cfg == nil || context <= 0 {
		return 0
	}
	headDim := cfg.KeyLength
	if headDim == 0 && cfg.HeadCount > 0 && cfg.EmbeddingLength > 0 {
		headDim = cfg.EmbeddingLength / cfg.HeadCount
	}
	kvHeads := cfg.HeadCountKV
	if kvHeads == 0 {
		kvHeads = cfg.HeadCount
	}
	if headDim <= 0 || kvHeads <= 0 {
		return 0
	}
	perToken := kvHeadBytes(headDim*kvHeads, typ)
	var layers uint64
	for il := 0; il < cfg.TrunkLayers(); il++ {
		if !cfg.IsRecurrent(il) {
			layers++
		}
	}
	return layers * uint64(context) * 2 * perToken
}

func kvHeadBytes(elements int, typ quant.Type) uint64 {
	if elements <= 0 {
		return 0
	}
	block := typ.BlockSize()
	size := typ.TypeSize()
	if block <= 0 || size <= 0 {
		// Unknown type: fall back to a float32 estimate.
		return uint64(elements) * 4
	}
	if elements%block == 0 {
		return uint64(elements/block) * uint64(size)
	}
	return uint64(math.Ceil(float64(elements)/float64(block))) * uint64(size)
}

// AutoPrefillChunk chooses a prompt chunk size from the context length and the
// backend's device memory. The goal is to keep scratch/activation memory and
// command-buffer size bounded without making short-context prefill needlessly
// slow. It is not exposed as a CLI flag.
func AutoPrefillChunk(context int, caps compute.Capabilities) int {
	if context <= 0 {
		context = 32768
	}
	chunk := 1024
	switch {
	case context <= 32*1024:
		chunk = 2048
	case context <= 128*1024:
		chunk = 1024
	case context <= 256*1024:
		chunk = 512
	default:
		chunk = 256
	}
	if caps.MemoryBytes > 0 {
		switch {
		case caps.MemoryBytes < 16<<30:
			chunk /= 4
		case caps.MemoryBytes < 32<<30:
			chunk /= 2
		case caps.MemoryBytes >= 64<<30 && chunk < 2048:
			chunk *= 2
		}
	}
	if chunk < 64 {
		chunk = 64
	}
	if chunk > context {
		chunk = context
	}
	return chunk
}

// ResolveRopeScaling turns the server's auto/none/linear/yarn setting into a
// concrete mode and factor. Auto selects YaRN for contexts above native.
func ResolveRopeScaling(context, native int, mode string, factor float64) (string, float64, error) {
	if mode == "" {
		mode = "auto"
	}
	if native <= 0 {
		native = context
	}
	switch mode {
	case "auto":
		if context > native {
			mode = "yarn"
			if factor <= 0 {
				factor = float64(context) / float64(native)
			}
		} else {
			mode = "none"
			factor = 1
		}
	case "none":
		factor = 1
	case "linear", "yarn":
		if factor <= 0 {
			if context > native {
				factor = float64(context) / float64(native)
			} else {
				factor = 1
			}
		}
	default:
		return "", 0, fmt.Errorf("unknown rope scaling %q (want auto, none, linear, or yarn)", mode)
	}
	return mode, factor, nil
}

// FormatTokens renders a token count compactly (e.g. 512k).
func FormatTokens(n int) string {
	if n > 0 && n%1024 == 0 {
		return fmt.Sprintf("%dk", n/1024)
	}
	return strconv.Itoa(n)
}

// FormatBytes renders a byte count in GiB/MiB.
func FormatBytes(n uint64) string {
	const unit = 1024.0
	f := float64(n)
	switch {
	case f >= unit*unit*unit:
		return fmt.Sprintf("%.2f GiB", f/(unit*unit*unit))
	case f >= unit*unit:
		return fmt.Sprintf("%.2f MiB", f/(unit*unit))
	case f >= unit:
		return fmt.Sprintf("%.2f KiB", f/unit)
	default:
		return fmt.Sprintf("%d B", n)
	}
}
