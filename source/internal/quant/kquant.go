package quant

import "math"

const groupMaxEps float32 = 1e-15

// This file holds faithful float32 ports of llama.cpp's K-quant quantization
// helpers (ggml-quants.c, revision 08246a28f6000100433d297c4e037c02e9d2d464).
// They exist so the Go encoders can be byte-identical to the reference
// quantizers; every arithmetic operation is float32 in the same order as the C
// code. See e2e-tests/verify-quants for the byte-exact cross-check.

// nearestInt is a direct port of ggml-quants.c nearest_int: round to nearest,
// ties to even, via the float32 magic-number trick.
func nearestInt(fval float32) int {
	val := fval + 12582912.0
	i := int32(math.Float32bits(val))
	return int(i&0x007fffff) - 0x00400000
}

func absf(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func sqrtf(x float32) float32 { return float32(math.Sqrt(float64(x))) }

// minInt returns the smaller of two ints.
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// makeQXQuants is a port of make_qx_quants (used by Q6_K). qw may be nil.
func makeQXQuants(n, nmax int, x []float32, L []int8, rmseType int, qw []float32) float32 {
	var maxV, amax float32
	for i := 0; i < n; i++ {
		ax := absf(x[i])
		if ax > amax {
			amax = ax
			maxV = x[i]
		}
	}
	if amax < groupMaxEps {
		for i := 0; i < n; i++ {
			L[i] = 0
		}
		return 0
	}
	iscale := -float32(nmax) / maxV
	if rmseType == 0 {
		for i := 0; i < n; i++ {
			l := nearestInt(iscale * x[i])
			L[i] = int8(nmax + clampInt(l, -nmax, nmax-1))
		}
		return 1 / iscale
	}
	returnEarly := false
	if rmseType < 0 {
		rmseType = -rmseType
		returnEarly = true
	}
	weight := func(i int) float32 {
		switch {
		case qw != nil:
			return qw[i]
		case rmseType == 1:
			return x[i] * x[i]
		case rmseType == 2:
			return 1
		case rmseType == 3:
			return absf(x[i])
		default:
			return sqrtf(absf(x[i]))
		}
	}
	var sumlx, suml2 float32
	for i := 0; i < n; i++ {
		l := nearestInt(iscale * x[i])
		l = clampInt(l, -nmax, nmax-1)
		L[i] = int8(l + nmax)
		w := weight(i)
		sumlx += w * x[i] * float32(l)
		suml2 += w * float32(l) * float32(l)
	}
	scale := float32(0)
	if suml2 != 0 {
		scale = sumlx / suml2
	}
	if returnEarly {
		if suml2 > 0 {
			return 0.5 * (scale + 1/iscale)
		}
		return 1 / iscale
	}
	best := scale * sumlx
	for is := -9; is <= 9; is++ {
		if is == 0 {
			continue
		}
		iscale = -(float32(nmax) + 0.1*float32(is)) / maxV
		sumlx, suml2 = 0, 0
		for i := 0; i < n; i++ {
			l := nearestInt(iscale * x[i])
			l = clampInt(l, -nmax, nmax-1)
			w := weight(i)
			sumlx += w * x[i] * float32(l)
			suml2 += w * float32(l) * float32(l)
		}
		if suml2 > 0 && sumlx*sumlx > best*suml2 {
			for i := 0; i < n; i++ {
				l := nearestInt(iscale * x[i])
				L[i] = int8(nmax + clampInt(l, -nmax, nmax-1))
			}
			scale = sumlx / suml2
			best = scale * sumlx
		}
	}
	return scale
}

// makeQ3Quants is a port of make_q3_quants (used by Q3_K).
func makeQ3Quants(n, nmax int, x []float32, L []int8, doRMSE bool) float32 {
	var maxV, amax float32
	for i := 0; i < n; i++ {
		ax := absf(x[i])
		if ax > amax {
			amax = ax
			maxV = x[i]
		}
	}
	if amax < groupMaxEps {
		for i := 0; i < n; i++ {
			L[i] = 0
		}
		return 0
	}
	iscale := -float32(nmax) / maxV
	if doRMSE {
		var sumlx, suml2 float32
		for i := 0; i < n; i++ {
			l := nearestInt(iscale * x[i])
			l = clampInt(l, -nmax, nmax-1)
			L[i] = int8(l)
			w := x[i] * x[i]
			sumlx += w * x[i] * float32(l)
			suml2 += w * float32(l) * float32(l)
		}
		for itry := 0; itry < 5; itry++ {
			nChanged := 0
			for i := 0; i < n; i++ {
				w := x[i] * x[i]
				slx := sumlx - w*x[i]*float32(L[i])
				if slx > 0 {
					sl2 := suml2 - w*float32(L[i])*float32(L[i])
					newL := nearestInt(x[i] * sl2 / slx)
					newL = clampInt(newL, -nmax, nmax-1)
					if newL != int(L[i]) {
						slx += w * x[i] * float32(newL)
						sl2 += w * float32(newL) * float32(newL)
						if sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2 {
							L[i] = int8(newL)
							sumlx = slx
							suml2 = sl2
							nChanged++
						}
					}
				}
			}
			if nChanged == 0 {
				break
			}
		}
		for i := 0; i < n; i++ {
			L[i] += int8(nmax)
		}
		if suml2 > 0 {
			return sumlx / suml2
		}
		return 0
	}
	for i := 0; i < n; i++ {
		l := nearestInt(iscale * x[i])
		l = clampInt(l, -nmax, nmax-1)
		L[i] = int8(l + nmax)
	}
	return 1 / iscale
}

// makeQKX2Quants is a port of make_qkx2_quants (used by Q4_K). It fills L and
// sets *theMin; the returned scale is positive.
func makeQKX2Quants(n, nmax int, x, weights []float32, L []uint8, theMin *float32, Laux []uint8, rmin, rdelta float32, nstep int, useMAD bool) float32 {
	minV := x[0]
	maxV := x[0]
	sumW := weights[0]
	sumX := sumW * x[0]
	for i := 1; i < n; i++ {
		if x[i] < minV {
			minV = x[i]
		}
		if x[i] > maxV {
			maxV = x[i]
		}
		w := weights[i]
		sumW += w
		sumX += w * x[i]
	}
	if minV > 0 {
		minV = 0
	}
	if maxV == minV {
		for i := 0; i < n; i++ {
			L[i] = 0
		}
		*theMin = -minV
		return 0
	}
	iscale := float32(nmax) / (maxV - minV)
	scale := 1 / iscale
	var bestError float32
	for i := 0; i < n; i++ {
		l := nearestInt(iscale * (x[i] - minV))
		l = clampInt(l, 0, nmax)
		L[i] = uint8(l)
		diff := scale*float32(L[i]) + minV - x[i]
		if useMAD {
			diff = absf(diff)
		} else {
			diff = diff * diff
		}
		bestError += weights[i] * diff
	}
	if nstep < 1 {
		*theMin = -minV
		return scale
	}
	for is := 0; is <= nstep; is++ {
		iscale = (rmin + rdelta*float32(is) + float32(nmax)) / (maxV - minV)
		var sumL, sumL2, sumXL float32
		for i := 0; i < n; i++ {
			l := nearestInt(iscale * (x[i] - minV))
			l = clampInt(l, 0, nmax)
			Laux[i] = uint8(l)
			w := weights[i]
			sumL += w * float32(l)
			sumL2 += w * float32(l) * float32(l)
			sumXL += w * float32(l) * x[i]
		}
		D := sumW*sumL2 - sumL*sumL
		if D > 0 {
			thisScale := (sumW*sumXL - sumX*sumL) / D
			thisMin := (sumL2*sumX - sumL*sumXL) / D
			if thisMin > 0 {
				thisMin = 0
				thisScale = sumXL / sumL2
			}
			var curError float32
			for i := 0; i < n; i++ {
				diff := thisScale*float32(Laux[i]) + thisMin - x[i]
				if useMAD {
					diff = absf(diff)
				} else {
					diff = diff * diff
				}
				curError += weights[i] * diff
			}
			if curError < bestError {
				for i := 0; i < n; i++ {
					L[i] = Laux[i]
				}
				bestError = curError
				scale = thisScale
				minV = thisMin
			}
		}
	}
	*theMin = -minV
	return scale
}

// makeQKX1Quants is a port of make_qkx1_quants (used by Q5_K).
func makeQKX1Quants(n, nmax int, x []float32, L []uint8, theMin *float32, ntry int, alpha float32) float32 {
	minV := x[0]
	maxV := x[0]
	for i := 1; i < n; i++ {
		if x[i] < minV {
			minV = x[i]
		}
		if x[i] > maxV {
			maxV = x[i]
		}
	}
	if maxV == minV {
		for i := 0; i < n; i++ {
			L[i] = 0
		}
		*theMin = 0
		return 0
	}
	if minV > 0 {
		minV = 0
	}
	iscale := float32(nmax) / (maxV - minV)
	scale := 1 / iscale
	for itry := 0; itry < ntry; itry++ {
		var sumlx float32
		var suml2 int
		didChange := false
		for i := 0; i < n; i++ {
			l := nearestInt(iscale * (x[i] - minV))
			l = clampInt(l, 0, nmax)
			if l != int(L[i]) {
				L[i] = uint8(l)
				didChange = true
			}
			sumlx += (x[i] - minV) * float32(l)
			suml2 += l * l
		}
		scale = sumlx / float32(suml2)
		var sum float32
		for i := 0; i < n; i++ {
			sum += x[i] - scale*float32(L[i])
		}
		minV = alpha*minV + (1-alpha)*sum/float32(n)
		if minV > 0 {
			minV = 0
		}
		iscale = 1 / scale
		if !didChange {
			break
		}
	}
	*theMin = -minV
	return scale
}

// getScaleMinK4 lives in dequant.go (shared by the Q4_K/Q5_K codecs).
