package abliterate

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestSearchSpaceDecode(t *testing.T) {
	s := NewSearchSpace(64)
	if s.Dims() != 10 {
		t.Fatalf("dims = %d, want 10", s.Dims())
	}
	names := s.Names()
	wantNames := []string{
		"direction_scope", "direction_index",
		"attn.max_weight", "attn.max_weight_position", "attn.min_weight", "attn.min_weight_distance",
		"mlp.max_weight", "mlp.max_weight_position", "mlp.min_weight", "mlp.min_weight_distance",
	}
	if !reflect.DeepEqual(names, wantNames) {
		t.Fatalf("names = %v", names)
	}

	lo := s.Decode(make([]float64, s.Dims()))
	hi := s.Decode(fill(s.Dims(), 1))
	last := float64(63)

	if lo.DirectionGlobal != true || hi.DirectionGlobal != false {
		t.Fatalf("scope: lo global=%v hi global=%v", lo.DirectionGlobal, hi.DirectionGlobal)
	}
	if math.Abs(lo.DirectionIndex-0.4*last) > 1e-9 || math.Abs(hi.DirectionIndex-0.9*last) > 1e-9 {
		t.Fatalf("direction index: %v %v", lo.DirectionIndex, hi.DirectionIndex)
	}
	// MLP lower bound is clamped to zero.
	if lo.MLP.MaxWeight != 0 || hi.MLP.MaxWeight != 1.5 {
		t.Fatalf("mlp max weight: %v %v", lo.MLP.MaxWeight, hi.MLP.MaxWeight)
	}
	if lo.Attn.MaxWeight != 0.8 || hi.Attn.MaxWeight != 1.5 {
		t.Fatalf("attn max weight: %v %v", lo.Attn.MaxWeight, hi.Attn.MaxWeight)
	}
	// min_weight is derived as fraction * max_weight.
	u := make([]float64, s.Dims())
	u[4] = 1.0 // attn min_weight fraction
	tp := s.Decode(u)
	if tp.Attn.MaxWeight != 0.8 {
		t.Fatalf("attn max weight = %v", tp.Attn.MaxWeight)
	}
	if math.Abs(tp.Attn.MinWeight-0.8) > 1e-9 {
		t.Fatalf("attn min weight = %v", tp.Attn.MinWeight)
	}
}

func TestTrialParamsParams(t *testing.T) {
	tp := TrialParams{
		DirectionGlobal: true,
		DirectionIndex:  30,
		Attn:            WeightDistribution{MaxWeight: 1, MaxWeightPosition: 40, MinWeight: 0.2, MinWeightDistance: 10},
		MLP:             WeightDistribution{MaxWeight: 0},
	}
	p := tp.Params()
	if p.DirectionIndex == nil || *p.DirectionIndex != 30 {
		t.Fatalf("direction index not set: %v", p.DirectionIndex)
	}
	if p.Attn.MaxWeightPosition != 40 || p.MLP.MaxWeight != 0 {
		t.Fatalf("params = %+v", p)
	}

	perLayer := TrialParams{DirectionGlobal: false}
	if perLayer.Params().DirectionIndex != nil {
		t.Fatal("per-layer direction should leave DirectionIndex nil")
	}
}

func TestSamplerReproducible(t *testing.T) {
	for _, name := range []string{"random", "halton", "tpe"} {
		a := NewSampler(name, 5, 12345)
		b := NewSampler(name, 5, 12345)
		for i := 0; i < 20; i++ {
			ua, ub := a.Suggest(), b.Suggest()
			if !reflect.DeepEqual(ua, ub) {
				t.Fatalf("%s not reproducible at %d: %v vs %v", name, i, ua, ub)
			}
		}
	}
}

func TestSearchFindsMinimum(t *testing.T) {
	s := NewSearchSpace(64)
	// The direction index and max weights are monotone in the unit coords, so
	// a quadratic in the decoded params is a valid smooth objective with a
	// known optimum.
	score := func(tp TrialParams) (float64, error) {
		a := tp.Attn.MinWeightFraction - 0.4
		m := tp.MLP.MinWeightFraction - 0.6
		return a*a + m*m, nil
	}
	res, err := s.Search(NewSampler("halton", s.Dims(), 7), SearchOptions{Trials: 512}, score)
	if err != nil {
		t.Fatal(err)
	}
	if res.BestScore > 1e-3 {
		t.Fatalf("best score %v too high (best unit %v)", res.BestScore, res.BestUnit)
	}

	rnd, err := s.Search(NewRandomSampler(s.Dims(), 99), SearchOptions{Trials: 2048}, score)
	if err != nil {
		t.Fatal(err)
	}
	if rnd.BestScore > 1e-2 {
		t.Fatalf("random best score %v too high", rnd.BestScore)
	}
}

func TestSearchResume(t *testing.T) {
	s := NewSearchSpace(16)
	score := func(tp TrialParams) (float64, error) {
		a := tp.Attn.MinWeightFraction - 0.5
		return a * a, nil
	}
	// A full run and a split run with the same seed must produce the same
	// sequence of trials (resume is exact).
	full, err := s.Search(NewRandomSampler(s.Dims(), 42), SearchOptions{Trials: 40}, score)
	if err != nil {
		t.Fatal(err)
	}
	first, err := s.Search(NewRandomSampler(s.Dims(), 42), SearchOptions{Trials: 20}, score)
	if err != nil {
		t.Fatal(err)
	}
	second, err := s.Search(NewRandomSampler(s.Dims(), 42), SearchOptions{Trials: 40, Prior: first.Results}, score)
	if err != nil {
		t.Fatal(err)
	}
	for i := range full.Results {
		if full.Results[i].Unit[0] != second.Results[i].Unit[0] {
			t.Fatalf("trial %d unit differs after resume", i)
		}
		if full.Results[i].Score != second.Results[i].Score {
			t.Fatalf("trial %d score differs after resume", i)
		}
	}
}

func TestSearchAbortsOnError(t *testing.T) {
	s := NewSearchSpace(8)
	sentinel := errors.New("sentinel")
	boom := func(TrialParams) (float64, error) { return 0, sentinel }
	if _, err := s.Search(NewRandomSampler(s.Dims(), 1), SearchOptions{Trials: 10}, boom); !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want sentinel", err)
	}
}

func TestHaltonBounds(t *testing.T) {
	hal := NewHaltonSampler(4, 0)
	for i := 0; i < 100; i++ {
		for _, v := range hal.Suggest() {
			if v < 0 || v >= 1 {
				t.Fatalf("halton value out of range: %v", v)
			}
		}
	}
}

func fill(n int, v float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
}
