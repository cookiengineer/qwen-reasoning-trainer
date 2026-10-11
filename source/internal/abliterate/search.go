// Package abliterate implements directional ablation of refusal directions in
// Qwen3.8 weights, following the technique used by Heretic
// (references/heretic/src/heretic/modifiers/abliteration.py):
//
//  1. Collect per-layer residual-stream means over "good" and "bad" prompts.
//  2. Compute the difference-of-means direction per layer (optionally
//     orthogonalized against the good direction).
//  3. Orthogonalize the attention out-projection and MLP down-projection
//     matrices against that direction, with an optional magnitude-preserving
//     (MPOA) row renormalization.
//
// This file adds the automated parameter search: a small, dependency-free
// optimizer over the same parameter space Heretic explores (direction scope and
// index, plus a per-component ablation-weight distribution). The search is
// scored on a callback so it can be driven by the real evaluator or a synthetic
// objective in tests.
package abliterate

import (
	"fmt"
	"math"
)

// WeightDistribution is the per-component ablation-weight kernel: the weight
// peaks at MaxWeight at layer MaxWeightPosition and decays linearly to MinWeight
// at distance MinWeightDistance, beyond which it is zero.
type WeightDistribution struct {
	MaxWeight         float64
	MaxWeightPosition float64
	// MinWeightFraction is the sampled fraction in [0,1]; MinWeight is the
	// derived value MinWeightFraction * MaxWeight (matching Heretic).
	MinWeightFraction float64
	MinWeight         float64
	MinWeightDistance float64
}

// TrialParams is one point in the abliteration search space. It maps directly
// onto Params.
type TrialParams struct {
	// DirectionGlobal selects a single interpolated direction index for all
	// layers; when false each layer uses its own direction.
	DirectionGlobal bool
	DirectionIndex  float64
	Attn            WeightDistribution
	MLP             WeightDistribution
}

// Params converts a TrialParams into the ablation Params used by Ablate.
func (tp TrialParams) Params() Params {
	p := Params{RowNormalization: "full"}
	if tp.DirectionGlobal {
		di := tp.DirectionIndex
		p.DirectionIndex = &di
	}
	p.Attn = ComponentParams{
		MaxWeight:         tp.Attn.MaxWeight,
		MaxWeightPosition: tp.Attn.MaxWeightPosition,
		MinWeight:         tp.Attn.MinWeight,
		MinWeightDistance: tp.Attn.MinWeightDistance,
	}
	p.MLP = ComponentParams{
		MaxWeight:         tp.MLP.MaxWeight,
		MaxWeightPosition: tp.MLP.MaxWeightPosition,
		MinWeight:         tp.MLP.MinWeight,
		MinWeightDistance: tp.MLP.MinWeightDistance,
	}
	return p
}

// searchDim is one continuous dimension of the search space. Every dimension is
// sampled in the unit interval and linearly mapped to [lo, hi]; the scope
// dimension is the exception (a threshold at 0.5).
type searchDim struct {
	name   string
	lo, hi float64
}

// SearchSpace is the rectangular search space. All dimensions are sampled
// unconditionally, even when unused, so that samplers that assume a fixed
// dimensionality (TPE) work correctly (this mirrors Heretic's note).
type SearchSpace struct {
	trunk int
	dims  []searchDim
}

// NewSearchSpace builds the search space for a model with trunk layers.
func NewSearchSpace(trunk int) *SearchSpace {
	if trunk < 2 {
		trunk = 2
	}
	last := float64(trunk - 1)
	dims := []searchDim{
		{"direction_scope", 0, 1},
		{"direction_index", 0.4 * last, 0.9 * last},
	}
	// Attn (attention/linear out-projection) gets a positive lower bound; the
	// MLP lower bound is negative and clamped to 0 so it can be disabled
	// entirely. Both mirror Heretic's suggest_parameters.
	components := []struct {
		name  string
		lower float64
	}{
		{"attn", 0.8},
		{"mlp", -0.25},
	}
	for _, c := range components {
		dims = append(dims,
			searchDim{c.name + ".max_weight", c.lower, 1.5},
			searchDim{c.name + ".max_weight_position", 0.6 * last, last},
			searchDim{c.name + ".min_weight", 0, 1},
			searchDim{c.name + ".min_weight_distance", 1, math.Max(0.6*last, 1)},
		)
	}
	return &SearchSpace{trunk: trunk, dims: dims}
}

// Dims returns the number of search dimensions.
func (s *SearchSpace) Dims() int { return len(s.dims) }

// Names returns the dimension names in order.
func (s *SearchSpace) Names() []string {
	out := make([]string, len(s.dims))
	for i, d := range s.dims {
		out[i] = d.name
	}
	return out
}

// Decode maps a unit-cube point (each coordinate in [0,1]) to TrialParams.
// Coordinates outside [0,1] are clamped.
func (s *SearchSpace) Decode(u []float64) TrialParams {
	if len(u) != len(s.dims) {
		panic(fmt.Sprintf("abliterate: unit vector length %d, want %d", len(u), len(s.dims)))
	}
	get := func(i int) float64 {
		v := u[i]
		if v < 0 {
			v = 0
		} else if v > 1 {
			v = 1
		}
		d := s.dims[i]
		return d.lo + v*(d.hi-d.lo)
	}
	tp := TrialParams{
		DirectionGlobal: get(0) < 0.5,
		DirectionIndex:  get(1),
	}
	tp.Attn = WeightDistribution{
		MaxWeight:         math.Max(0, get(2)),
		MaxWeightPosition: get(3),
		MinWeightFraction: get(4),
		MinWeightDistance: get(5),
	}
	tp.MLP = WeightDistribution{
		MaxWeight:         math.Max(0, get(6)),
		MaxWeightPosition: get(7),
		MinWeightFraction: get(8),
		MinWeightDistance: get(9),
	}
	tp.Attn.MinWeight = tp.Attn.MinWeightFraction * tp.Attn.MaxWeight
	tp.MLP.MinWeight = tp.MLP.MinWeightFraction * tp.MLP.MaxWeight
	return tp
}

// TrialResult records one evaluated trial.
type TrialResult struct {
	Index  int
	Unit   []float64
	Params TrialParams
	Score  float64
}

// SearchResult is the outcome of a search.
type SearchResult struct {
	Best      TrialParams
	BestUnit  []float64
	BestScore float64
	Results   []TrialResult
}

// EvalFunc scores a parameter set. Lower is better. It must not keep a
// reference to the params beyond the call.
type EvalFunc func(TrialParams) (float64, error)

// SearchOptions configures a search run.
type SearchOptions struct {
	// Trials is the total number of trials (including any resumed prior ones).
	Trials int
	// Prior holds results from a previous run to resume from. The sampler is
	// fast-forwarded past them.
	Prior []TrialResult
	// OnTrial is called after each new trial (optional).
	OnTrial func(TrialResult)
}

// Search runs Trials total trials using the sampler, calling eval for each new
// trial and OnTrial (optional) after each. Prior results (if any) are replayed
// into the sampler and seeded into the result. Errors from eval abort the
// search and are returned with the partial result.
func (s *SearchSpace) Search(sampler Sampler, opt SearchOptions, eval EvalFunc) (*SearchResult, error) {
	if sampler == nil {
		return nil, fmt.Errorf("abliterate: nil sampler")
	}
	if sampler.Dims() != s.Dims() {
		return nil, fmt.Errorf("abliterate: sampler dims %d != space dims %d", sampler.Dims(), s.Dims())
	}
	res := &SearchResult{BestScore: math.Inf(1)}
	for _, r := range opt.Prior {
		if r.Score < res.BestScore {
			res.BestScore = r.Score
			res.Best = r.Params
			res.BestUnit = append([]float64(nil), r.Unit...)
		}
	}
	res.Results = append(res.Results, opt.Prior...)
	sampler.Resume(opt.Prior)

	newTrials := opt.Trials - len(opt.Prior)
	for i := 0; i < newTrials; i++ {
		u := sampler.Suggest()
		tp := s.Decode(u)
		score, err := eval(tp)
		if err != nil {
			return res, err
		}
		sampler.Observe(u, score)
		tr := TrialResult{Index: len(res.Results), Unit: append([]float64(nil), u...), Params: tp, Score: score}
		res.Results = append(res.Results, tr)
		if score < res.BestScore {
			res.BestScore = score
			res.Best = tp
			res.BestUnit = append([]float64(nil), u...)
		}
		if opt.OnTrial != nil {
			opt.OnTrial(tr)
		}
	}
	return res, nil
}
