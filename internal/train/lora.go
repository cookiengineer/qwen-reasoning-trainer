package train

import (
	"math"
	"runtime"
	"sync"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// LoRAConfig describes which projections receive adapters and their geometry.
type LoRAConfig struct {
	Rank    int
	Alpha   float32
	Targets []string
}

// DefaultLoRA is the agreed configuration: rank 16, alpha 32, all attention and
// MLP projections. For linear-attention layers the equivalent projections
// (fused qkv, gate, ssm_out) are used.
func DefaultLoRA() LoRAConfig {
	return LoRAConfig{
		Rank:  16,
		Alpha: 32,
		Targets: []string{
			"attn_q", "attn_k", "attn_v", "attn_output",
			"attn_qkv", "attn_gate", "ssm_out",
			"ffn_gate", "ffn_up", "ffn_down",
		},
	}
}

// Scale returns alpha/rank, the factor applied to the low-rank update.
func (c LoRAConfig) Scale() float32 { return c.Alpha / float32(c.Rank) }

// LoRA holds the two low-rank factors for one base weight. A is [K,rank], B is
// [rank,N] in GGML layout (Dims[0] is the reduction dimension). B is
// initialized to zero so the adapter starts as a no-op.
type LoRA struct {
	A     *compute.Tensor
	B     *compute.Tensor
	Scale float32
}

// NewLoRA allocates adapters for a base weight [K,N].
func NewLoRA(seed uint64, K, N, rank int, alpha float32) *LoRA {
	a := compute.NewF32(K, rank)
	s := seed
	for i := range a.F32 {
		s = s*6364136223846793005 + 1442695040888963407
		u := float64((s>>11)&((1<<53)-1)) / float64(1<<53)
		a.F32[i] = float32((u*2 - 1) / math.Sqrt(float64(K)))
	}
	b := compute.NewF32(rank, N)
	return &LoRA{A: a, B: b, Scale: alpha / float32(rank)}
}

// Params returns the trainable adapter tensors.
func (l *LoRA) Params() []*compute.Tensor { return []*compute.Tensor{l.A, l.B} }

// Update returns the low-rank contribution scale * B (A x) for activation x
// [K,M], giving [N,M]. It is used by the host graph; the device path folds it
// into the matmul.
func (l *LoRA) Update(x *compute.Tensor) (*compute.Tensor, error) {
	u, err := compute.MatMul(l.A, x) // [rank,M]
	if err != nil {
		return nil, err
	}
	v, err := compute.MatMul(l.B, u) // [N,M]
	if err != nil {
		return nil, err
	}
	return compute.Scale(v, l.Scale), nil
}

// MergeInto adds the adapter into a dequantized float32 weight [K,N] in place:
// W <- W + scale * B A.
//
// It is expressed as r rank-1 updates so the inner loop over k is contiguous in
// both W and A (the naive n/k/j loop strides through A by K and is dominated by
// cache misses). Large weights are split across goroutines by output row.
func (l *LoRA) MergeInto(w *compute.Tensor) {
	K := w.Ne(0)
	N := w.Ne(1)
	r := l.A.Ne(1)
	workers := runtime.GOMAXPROCS(0)
	if workers > N {
		workers = N
	}
	if workers <= 1 || int64(K)*int64(N) < 1<<20 {
		l.mergeRows(w.F32, 0, N, K, r)
		return
	}
	block := (N + workers - 1) / workers
	var wg sync.WaitGroup
	for n0 := 0; n0 < N; n0 += block {
		n1 := n0 + block
		if n1 > N {
			n1 = N
		}
		wg.Add(1)
		go func(n0, n1 int) {
			defer wg.Done()
			l.mergeRows(w.F32, n0, n1, K, r)
		}(n0, n1)
	}
	wg.Wait()
}

// mergeRows applies the adapter to output rows [n0,n1) of a [K,N] GGML weight.
func (l *LoRA) mergeRows(w []float32, n0, n1, K, r int) {
	scale := l.Scale
	for n := n0; n < n1; n++ {
		wrow := w[n*K : (n+1)*K]
		for j := 0; j < r; j++ {
			bn := scale * l.B.F32[n*r+j]
			if bn == 0 {
				continue
			}
			arow := l.A.F32[j*K : (j+1)*K]
			for k := range wrow {
				wrow[k] += bn * arow[k]
			}
		}
	}
}

// MergeToQuant dequantizes a base weight, applies the adapter, and requantizes
// to the base's original type (falling back to Q4_K when the type has no
// encoder), mirroring the abliteration output policy. It returns the output
// type and raw bytes.
func (l *LoRA) MergeToQuant(baseType quant.Type, baseRaw []byte, dims []int) (quant.Type, []byte, error) {
	n := 1
	for _, d := range dims {
		n *= d
	}
	f32, err := quant.DequantParallel(baseType, baseRaw, int64(n))
	if err != nil {
		return 0, nil, err
	}
	wt := &compute.Tensor{Dims: dims, Type: quant.TypeF32, F32: f32}
	l.MergeInto(wt)
	outType := baseType
	if !quant.CanQuantize(outType) {
		outType = quant.TypeQ4_K
	}
	raw, err := quant.QuantizeParallel(outType, wt.F32)
	if err != nil {
		return 0, nil, err
	}
	return outType, raw, nil
}
