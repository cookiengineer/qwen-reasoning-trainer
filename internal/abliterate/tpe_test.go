package abliterate

import (
	"reflect"
	"testing"
)

// TestTPESamplerConverges checks that TPE drives a smooth 2-D objective much
// lower than uniform random search on the same budget.
func TestTPESamplerConverges(t *testing.T) {
	objective := func(u []float64) float64 {
		a := u[0] - 0.7
		b := u[1] - 0.3
		return a*a + b*b
	}

	tpe := NewTPESampler(2, 1)
	tpeBest := 1.0
	for i := 0; i < 200; i++ {
		u := tpe.Suggest()
		f := objective(u)
		tpe.Observe(u, f)
		if f < tpeBest {
			tpeBest = f
		}
	}

	rnd := NewRandomSampler(2, 1)
	rndBest := 1.0
	for i := 0; i < 200; i++ {
		u := rnd.Suggest()
		f := objective(u)
		rnd.Observe(u, f)
		if f < rndBest {
			rndBest = f
		}
	}

	if tpeBest > 1e-4 {
		t.Fatalf("TPE best %v too high", tpeBest)
	}
	if tpeBest >= rndBest {
		t.Fatalf("TPE best %v did not beat random %v", tpeBest, rndBest)
	}
}

func TestTPEReproducible(t *testing.T) {
	run := func() []float64 {
		s := NewTPESampler(3, 99)
		var last []float64
		for i := 0; i < 60; i++ {
			u := s.Suggest()
			f := (u[0]-0.2)*(u[0]-0.2) + (u[1]-0.8)*(u[1]-0.8) + (u[2]-0.5)*(u[2]-0.5)
			s.Observe(u, f)
			last = u
		}
		return last
	}
	if !reflect.DeepEqual(run(), run()) {
		t.Fatal("TPE not reproducible for a fixed seed")
	}
}
