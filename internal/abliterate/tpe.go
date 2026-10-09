package abliterate

import (
	"math"
	"math/rand"
)

// TPESampler is a compact, dependency-free Tree-structured Parzen Estimator
// (TPE) over the unit cube, following the single-objective independent-dimension
// formulation (Bergstra et al.). It models the density of the best-so-far points
// ("good") and the rest ("bad") with Gaussian-kernel Parzen estimators and picks
// the candidate that maximizes p(x|good) / p(x|bad).
//
// It is the optimizer counterpart to RandomSampler and is selected with
// --sampler tpe. The first Startup trials are uniformly random, because TPE has
// nothing to model before that.
type TPESampler struct {
	d       int
	gamma   float64
	nEI     int
	startup int
	rng     *rand.Rand

	x [][]float64
	y []float64
}

// NewTPESampler creates a TPE sampler over d dimensions.
func NewTPESampler(d int, seed int64) *TPESampler {
	startup := 10
	if s := 2 * d; s > startup {
		startup = s
	}
	return &TPESampler{
		d:       d,
		gamma:   0.25,
		nEI:     64,
		startup: startup,
		rng:     rand.New(rand.NewSource(seed)),
	}
}

// Suggest implements Sampler.
func (s *TPESampler) Suggest() []float64 {
	if len(s.y) < s.startup {
		return s.randomPoint()
	}

	// Sort history by score (ascending: lower is better).
	order := make([]int, len(s.y))
	for i := range order {
		order[i] = i
	}
	sortIntsByScore(order, s.y)
	nGood := int(float64(len(order)) * s.gamma)
	if nGood < 1 {
		nGood = 1
	}
	if nGood >= len(order) {
		return s.randomPoint()
	}
	good := make([][]float64, nGood)
	for i := 0; i < nGood; i++ {
		good[i] = s.x[order[i]]
	}
	bad := make([][]float64, len(order)-nGood)
	for i := nGood; i < len(order); i++ {
		bad[i-nGood] = s.x[order[i]]
	}

	hGood := bandwidth(good, s.d)
	hBad := bandwidth(bad, s.d)

	best := s.randomPoint()
	bestL := math.Inf(-1)
	for i := 0; i < s.nEI; i++ {
		cand := sampleKDE(good, hGood, s.rng)
		l := logDensity(cand, good, hGood) - logDensity(cand, bad, hBad)
		if l > bestL {
			bestL = l
			best = cand
		}
	}
	return best
}

// Observe implements Sampler.
func (s *TPESampler) Observe(u []float64, score float64) {
	s.x = append(s.x, clampUnit(u))
	s.y = append(s.y, score)
}

// Resume replays prior trials into the model.
func (s *TPESampler) Resume(prior []TrialResult) {
	for _, r := range prior {
		s.Observe(r.Unit, r.Score)
	}
}

// Dims implements Sampler.
func (s *TPESampler) Dims() int { return s.d }

// Name implements Sampler.
func (s *TPESampler) Name() string { return "tpe" }

func (s *TPESampler) randomPoint() []float64 {
	u := make([]float64, s.d)
	for i := range u {
		u[i] = s.rng.Float64()
	}
	return u
}

// bandwidth returns a per-dimension Silverman bandwidth for the Parzen
// estimator, computed over the given points.
func bandwidth(points [][]float64, d int) []float64 {
	h := make([]float64, d)
	n := float64(len(points))
	scale := math.Pow(n, -1.0/(4.0+float64(d)))
	if len(points) < 2 {
		for j := range h {
			h[j] = 0.3
		}
		return h
	}
	mean := make([]float64, d)
	for _, p := range points {
		for j := 0; j < d; j++ {
			mean[j] += p[j]
		}
	}
	for j := range mean {
		mean[j] /= n
	}
	for j := 0; j < d; j++ {
		var v float64
		for _, p := range points {
			diff := p[j] - mean[j]
			v += diff * diff
		}
		std := math.Sqrt(v / n)
		// Floor the spread so the estimator can bridge gaps between the
		// current good region and a better one; without it the bandwidth
		// collapses onto the first good cluster and the search stalls.
		if std < 0.05 {
			std = 0.05
		}
		h[j] = std * scale
		if h[j] < 0.02 {
			h[j] = 0.02
		}
	}
	return h
}

// sampleKDE draws a sample from the Parzen density: a random kernel center plus
// Gaussian noise scaled by the bandwidth.
func sampleKDE(points [][]float64, h []float64, rng *rand.Rand) []float64 {
	center := points[rng.Intn(len(points))]
	out := make([]float64, len(h))
	for j := range out {
		out[j] = center[j] + rng.NormFloat64()*h[j]
	}
	return clampUnit(out)
}

// logDensity returns log p(x) under an axis-aligned Gaussian-kernel Parzen
// estimator, including the normalization constants so it is comparable across
// different point sets.
func logDensity(x []float64, points [][]float64, h []float64) float64 {
	d := len(x)
	logs := make([]float64, len(points))
	for i, p := range points {
		var q float64
		for j := 0; j < d; j++ {
			z := (x[j] - p[j]) / h[j]
			q += z * z
		}
		logs[i] = -0.5 * q
	}
	m := math.Inf(-1)
	for _, l := range logs {
		if l > m {
			m = l
		}
	}
	var sum float64
	for _, l := range logs {
		sum += math.Exp(l - m)
	}
	// log( (1/(n * prod h)) * (2pi)^(-d/2) * sum_i exp(-0.5 z^2) )
	const log2pi = 1.8378770664093453
	norm := -math.Log(float64(len(points))) - float64(d)/2*log2pi
	var logh float64
	for _, hj := range h {
		logh += math.Log(hj)
	}
	return m + math.Log(sum) + norm - logh
}

func clampUnit(u []float64) []float64 {
	out := make([]float64, len(u))
	for i, v := range u {
		if v < 0 {
			v = 0
		} else if v > 1 {
			v = 1
		}
		out[i] = v
	}
	return out
}

func sortIntsByScore(order []int, scores []float64) {
	// Insertion sort is fine: histories are small (hundreds of trials).
	for i := 1; i < len(order); i++ {
		v := order[i]
		j := i - 1
		for j >= 0 && scores[order[j]] > scores[v] {
			order[j+1] = order[j]
			j--
		}
		order[j+1] = v
	}
}
