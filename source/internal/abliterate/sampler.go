package abliterate

import (
	"math/rand"
)

// Sampler proposes points in the unit cube and learns from their scores.
// Suggest returns the next point; Observe records the score returned by the
// objective for the most recent Suggest. Resume fast-forwards the sampler past
// already-evaluated trials so a persisted study can be continued; stateless
// samplers advance their sequence, stateful samplers replay the points.
// Implementations must be deterministic given their seed so resume is exact.
type Sampler interface {
	Suggest() []float64
	Observe(u []float64, score float64)
	Resume(prior []TrialResult)
	Dims() int
	Name() string
}

// RandomSampler draws uniform points in [0,1)^d. It is the baseline sampler and
// the default; a TPE sampler can be substituted behind the same interface.
type RandomSampler struct {
	rng *rand.Rand
	d   int
}

// NewRandomSampler creates a seeded uniform sampler over d dimensions.
func NewRandomSampler(d int, seed int64) *RandomSampler {
	return &RandomSampler{rng: rand.New(rand.NewSource(seed)), d: d}
}

// Suggest implements Sampler.
func (s *RandomSampler) Suggest() []float64 {
	u := make([]float64, s.d)
	for i := range u {
		u[i] = s.rng.Float64()
	}
	return u
}

// Observe implements Sampler (no-op for random search).
func (s *RandomSampler) Observe(u []float64, score float64) {}

// Resume advances the RNG by one draw per prior trial so the sequence
// continues exactly where a persisted study stopped.
func (s *RandomSampler) Resume(prior []TrialResult) {
	for range prior {
		s.Suggest()
	}
}

// Dims implements Sampler.
func (s *RandomSampler) Dims() int { return s.d }

// Name implements Sampler.
func (s *RandomSampler) Name() string { return "random" }

// HaltonSampler is a deterministic low-discrepancy sampler (Halton sequence).
// It uses the first d primes as bases and a scrambling seed that advances by a
// random offset so different runs explore different points.
type HaltonSampler struct {
	d     int
	index uint64
	bases []int
}

// NewHaltonSampler creates a Halton sampler over d dimensions, starting at the
// given index offset.
func NewHaltonSampler(d int, offset uint64) *HaltonSampler {
	return &HaltonSampler{d: d, index: offset, bases: firstPrimes(d)}
}

// Suggest implements Sampler.
func (s *HaltonSampler) Suggest() []float64 {
	u := make([]float64, s.d)
	for i := 0; i < s.d; i++ {
		u[i] = radicalInverse(s.index+1, uint64(s.bases[i]))
	}
	s.index++
	return u
}

// Observe implements Sampler (no-op).
func (s *HaltonSampler) Observe(u []float64, score float64) {}

// Resume advances the Halton index past the prior trials.
func (s *HaltonSampler) Resume(prior []TrialResult) { s.index += uint64(len(prior)) }

// Dims implements Sampler.
func (s *HaltonSampler) Dims() int { return s.d }

// Name implements Sampler.
func (s *HaltonSampler) Name() string { return "halton" }

// radicalInverse returns the radical inverse of n in the given base.
func radicalInverse(n, base uint64) float64 {
	var inv float64
	f := 1.0 / float64(base)
	for n > 0 {
		inv += float64(n%base) * f
		n /= base
		f /= float64(base)
	}
	return inv
}

func firstPrimes(n int) []int {
	primes := make([]int, 0, n)
	for c := 2; len(primes) < n; c++ {
		prime := true
		for _, p := range primes {
			if p*p > c {
				break
			}
			if c%p == 0 {
				prime = false
				break
			}
		}
		if prime {
			primes = append(primes, c)
		}
	}
	return primes
}

// NewSampler constructs a sampler by name ("random", "halton", or "tpe"),
// falling back to random for an empty/unknown name.
func NewSampler(name string, d int, seed int64) Sampler {
	switch name {
	case "halton":
		off := uint64(0)
		if seed > 0 {
			off = uint64(seed)
		}
		return NewHaltonSampler(d, off)
	case "tpe":
		return NewTPESampler(d, seed)
	default:
		return NewRandomSampler(d, seed)
	}
}
