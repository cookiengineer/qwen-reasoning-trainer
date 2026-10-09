package vulkan_test

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
)

func TestVulkanGDNChunkedForwardAux(t *testing.T) {
	_, v := newBackends(t)
	vb, ok := v.(*vulkan.Backend)
	if !ok {
		t.Fatal("vulkan backend type")
	}
	sv, h, nTok, chunk := 4, 3, 19, 4
	q := autograd.Random(1, 1.0, sv, h, nTok)
	k := autograd.Random(2, 1.0, sv, h, nTok)
	vv := autograd.Random(3, 1.0, sv, h, nTok)
	g := autograd.Random(4, 0.3, 1, h, nTok)
	beta := autograd.Random(5, 0.5, 1, h, nTok)
	state := autograd.Random(6, 0.5, sv, sv, h)

	work, L, err := vb.GatedDeltaNetChunkedForwardAux(
		up(t, v, q.Dims, q.F32), up(t, v, k.Dims, k.F32), up(t, v, vv.Dims, vv.F32),
		up(t, v, g.Dims, g.F32), up(t, v, beta.Dims, beta.F32), up(t, v, state.Dims, state.F32), chunk)
	if err != nil {
		t.Fatal(err)
	}
	gotF := down(t, v, work)
	got := make([]float64, len(gotF))
	for i := range gotF {
		got[i] = float64(gotF[i])
	}

	// Pure-Go reference.
	NC := L.NC
	at := func(arr []float32, idx int) float64 { return float64(arr[idx]) }
	idx3 := func(t0, hh, i int) int { return (t0*h+hh)*sv + i }
	// Gc
	for hh := 0; hh < h; hh++ {
		for c := 0; c < NC; c++ {
			t0 := c * chunk
			clen := chunk
			if t0+clen > nTok {
				clen = nTok - t0
			}
			for i := 0; i < sv; i++ {
				cum := 0.0
				for r := 0; r < clen; r++ {
					cum += at(g.F32, (t0+r)*h+hh)
					gi := L.Gc + ((hh*NC+c)*chunk)*sv + r*sv + i
					if d := math.Abs(got[gi] - cum); d > 1e-5 {
						t.Fatalf("Gc[%d,%d,%d,%d] dev=%g want=%g", hh, c, r, i, got[gi], cum)
					}
				}
			}
		}
	}
	// A / Bt / delta / boundaries
	bnd := func(arr []float64, hh, c, j, i int) float64 {
		return arr[L.Bnd+((hh*(NC+1)+c)*sv+j)*sv+i]
	}
	delta := func(dev bool, hh, c, r, j int) float64 {
		if dev {
			return got[L.Delta+((hh*NC+c)*chunk+r)*sv+j]
		}
		return 0
	}
	for hh := 0; hh < h; hh++ {
		// boundary 0
		for j := 0; j < sv; j++ {
			for i := 0; i < sv; i++ {
				if d := math.Abs(bnd(got, hh, 0, j, i) - at(state.F32, hh*sv*sv+j*sv+i)); d > 1e-5 {
					t.Fatalf("bnd0[%d,%d,%d] %g vs %g", hh, j, i, bnd(got, hh, 0, j, i), at(state.F32, hh*sv*sv+j*sv+i))
				}
			}
		}
		for c := 0; c < NC; c++ {
			t0 := c * chunk
			clen := chunk
			if t0+clen > nTok {
				clen = nTok - t0
			}
			gcBase := L.Gc + ((hh*NC+c)*chunk)*sv
			// A/Bt
			for r := 0; r < clen; r++ {
				for p := 0; p < r; p++ {
					a := 0.0
					for i := 0; i < sv; i++ {
						cexp := math.Exp(got[gcBase+r*sv+i] - got[gcBase+p*sv+i])
						a += cexp * at(k.F32, idx3(t0+p, hh, i)) * at(k.F32, idx3(t0+r, hh, i))
					}
					ai := L.A + (hh*NC+c)*chunk*chunk + p*chunk + r
					if d := math.Abs(got[ai] - a); d > 1e-5 {
						t.Fatalf("A[%d,%d,%d,%d] dev=%g want=%g", hh, c, p, r, got[ai], a)
					}
				}
			}
			// delta via recurrence
			for r := 0; r < clen; r++ {
				b := at(beta.F32, (t0+r)*h+hh)
				for j := 0; j < sv; j++ {
					s := 0.0
					for i := 0; i < sv; i++ {
						s += bnd(got, hh, c, j, i) * math.Exp(got[gcBase+r*sv+i]) * at(k.F32, idx3(t0+r, hh, i))
					}
					rhs := b * (at(vv.F32, idx3(t0+r, hh, j)) - s)
					for p := 0; p < r; p++ {
						ai := L.A + (hh*NC+c)*chunk*chunk + p*chunk + r
						rhs -= b * got[ai] * delta(true, hh, c, p, j)
					}
					if d := math.Abs(delta(true, hh, c, r, j) - rhs); d > 1e-4 {
						t.Fatalf("delta[%d,%d,%d,%d] dev=%g want=%g", hh, c, r, j, delta(true, hh, c, r, j), rhs)
					}
				}
			}
		}
	}
}
