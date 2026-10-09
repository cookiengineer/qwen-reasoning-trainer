package vulkan

import (
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// benchGDNBackward times the atomic GatedDeltaNet backward at realistic model
// dimensions. It is capped by the kernel's scratch limit (h*sv*(T+1)*sv must
// stay under 128M elements), so T is bounded around 160 for h=48, sv=128.
func benchGDNBackward(b *testing.B, sv, h, nTok int) {
	v, err := New()
	if err != nil {
		b.Skip(err)
	}
	defer v.Close()
	if !v.Capabilities().Float32Atomics {
		b.Skip("vulkan device lacks float32 atomics")
	}
	up := func(t *compute.Tensor) compute.Buffer {
		buf, err := v.Upload(t)
		if err != nil {
			b.Fatal(err)
		}
		return buf
	}
	q := autograd.Random(1, 1.0, sv, h, nTok)
	k := autograd.Random(2, 1.0, sv, h, nTok)
	vv := autograd.Random(3, 1.0, sv, h, nTok)
	g := autograd.Random(4, 0.3, sv, h, nTok)
	beta := autograd.Random(5, 0.5, 1, h, nTok)
	state := autograd.Random(6, 0.5, sv, sv, h)
	dOut := autograd.Random(7, 1.0, sv, h, nTok)
	dNew := autograd.Random(8, 1.0, sv, sv, h)

	qb, kb, vb := up(q), up(k), up(vv)
	gb, bb, sb := up(g), up(beta), up(state)
	dob, dnb := up(dOut), up(dNew)
	if err := v.Sync(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _, _, _, _, err := v.GatedDeltaNetBackward(qb, kb, vb, gb, bb, sb, dob, dnb)
		if err != nil {
			b.Fatal(err)
		}
		if err := v.Sync(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGDNBackward128(b *testing.B)   { benchGDNBackward(b, 128, 48, 128) }
func BenchmarkGDNBackward160(b *testing.B)   { benchGDNBackward(b, 128, 48, 160) }
func BenchmarkGDNBackwardSmall(b *testing.B) { benchGDNBackward(b, 64, 16, 64) }

func benchGDNChunkedBackward(b *testing.B, sv, h, nTok, chunk int) {
	v, err := New()
	if err != nil {
		b.Skip(err)
	}
	defer v.Close()
	if !v.Capabilities().Float32Atomics {
		b.Skip("vulkan device lacks float32 atomics")
	}
	up := func(t *compute.Tensor) compute.Buffer {
		buf, err := v.Upload(t)
		if err != nil {
			b.Fatal(err)
		}
		return buf
	}
	q := autograd.Random(1, 1.0, sv, h, nTok)
	k := autograd.Random(2, 1.0, sv, h, nTok)
	vv := autograd.Random(3, 1.0, sv, h, nTok)
	g := autograd.Random(4, 0.3, sv, h, nTok)
	beta := autograd.Random(5, 0.5, 1, h, nTok)
	state := autograd.Random(6, 0.5, sv, sv, h)
	dOut := autograd.Random(7, 1.0, sv, h, nTok)
	dNew := autograd.Random(8, 1.0, sv, sv, h)
	qb, kb, vb := up(q), up(k), up(vv)
	gb, bb, sb := up(g), up(beta), up(state)
	dob, dnb := up(dOut), up(dNew)
	if err := v.Sync(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _, _, _, _, err := v.GatedDeltaNetChunkedBackward(qb, kb, vb, gb, bb, sb, dob, dnb, chunk)
		if err != nil {
			b.Fatal(err)
		}
		if err := v.Sync(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGDNChunked128(b *testing.B)  { benchGDNChunkedBackward(b, 128, 48, 128, 64) }
func BenchmarkGDNChunked512(b *testing.B)  { benchGDNChunkedBackward(b, 128, 48, 512, 64) }
func BenchmarkGDNChunked1024(b *testing.B) { benchGDNChunkedBackward(b, 128, 48, 1024, 64) }

func BenchmarkGDNChunked512C16(b *testing.B)  { benchGDNChunkedBackward(b, 128, 48, 512, 16) }
func BenchmarkGDNChunked512C32(b *testing.B)  { benchGDNChunkedBackward(b, 128, 48, 512, 32) }
func BenchmarkGDNChunked512C48(b *testing.B)  { benchGDNChunkedBackward(b, 128, 48, 512, 48) }
func BenchmarkGDNChunked1024C32(b *testing.B) { benchGDNChunkedBackward(b, 128, 48, 1024, 32) }
