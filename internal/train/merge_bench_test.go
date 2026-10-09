package train

import (
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// BenchmarkMergeToQuantFFNDown measures the host merge of one representative
// large weight (ffn_down, rank 16, Q5_K base): dequantize + LoRA merge +
// requantize. Run with -benchtime=1x.
func BenchmarkMergeToQuantFFNDown(b *testing.B) {
	const K, N = 17408, 5120
	f32 := make([]float32, K*N)
	s := uint64(1)
	for i := range f32 {
		s = s*6364136223846793005 + 1442695040888963407
		f32[i] = float32(int64(s>>40))/float32(1<<23) - 1
	}
	raw, err := quant.Quantize(quant.TypeQ5_K, f32)
	if err != nil {
		b.Fatal(err)
	}
	l := NewLoRA(1, K, N, 16, 32)
	// Non-zero B so the merge actually adds something.
	for i := range l.B.F32 {
		l.B.F32[i] = 0.01
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := l.MergeToQuant(quant.TypeQ5_K, raw, []int{K, N}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDequantQ5K(b *testing.B) {
	const K, N = 17408, 5120
	f32 := make([]float32, K*N)
	for i := range f32 {
		f32[i] = float32(int64(i%2000))/1000 - 1
	}
	raw, err := quant.Quantize(quant.TypeQ5_K, f32)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := quant.Dequant(quant.TypeQ5_K, raw, int64(K*N)); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQuantizeQ5K(b *testing.B) {
	const K, N = 17408, 5120
	f32 := make([]float32, K*N)
	for i := range f32 {
		f32[i] = float32(int64(i%2000))/1000 - 1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := quant.Quantize(quant.TypeQ5_K, f32); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMergeIntoOnly(b *testing.B) {
	const K, N = 17408, 5120
	f32 := make([]float32, K*N)
	l := NewLoRA(1, K, N, 16, 32)
	for i := range l.B.F32 {
		l.B.F32[i] = 0.01
	}
	wt := &compute.Tensor{Dims: []int{K, N}, F32: f32}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.MergeInto(wt)
	}
}
