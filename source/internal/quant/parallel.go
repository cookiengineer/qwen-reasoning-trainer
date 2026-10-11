package quant

import (
	"runtime"
	"sync"
)

// parallelThreshold is the minimum element count before Dequant/Quantize split
// the work across goroutines. Below it the goroutine overhead dominates.
const parallelThreshold = 1 << 19

// ranges splits [0,n) into up to workers block-aligned sub-ranges. n must be a
// multiple of bs.
func ranges(n int64, bs, workers int) [][2]int64 {
	if workers < 1 {
		workers = 1
	}
	per := n / int64(workers)
	per -= per % int64(bs)
	if per < int64(bs) {
		return [][2]int64{{0, n}}
	}
	var out [][2]int64
	for i := int64(0); i < n; {
		j := i + per
		if j > n {
			j = n
		}
		out = append(out, [2]int64{i, j})
		i = j
	}
	return out
}

// DequantParallel decodes n elements of type t from src into a new float32 slice,
// splitting the block-aligned work across GOMAXPROCS. It matches Dequant.
func DequantParallel(t Type, src []byte, n int64) ([]float32, error) {
	dst := make([]float32, n)
	bs := t.BlockSize()
	ts := t.TypeSize()
	workers := runtime.GOMAXPROCS(0)
	if bs <= 1 || ts <= 0 || n < parallelThreshold || workers <= 1 {
		if err := DequantTo(t, src, dst); err != nil {
			return nil, err
		}
		return dst, nil
	}
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstErr error
	for _, r := range ranges(n, bs, workers) {
		i, j := r[0], r[1]
		wg.Add(1)
		go func() {
			defer wg.Done()
			byteOff := i / int64(bs) * int64(ts)
			if err := DequantTo(t, src[byteOff:], dst[i:j]); err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	return dst, nil
}

// QuantizeParallel encodes src into a new byte slice, splitting the
// block-aligned work across GOMAXPROCS. It matches Quantize.
func QuantizeParallel(t Type, src []float32) ([]byte, error) {
	size, err := t.Size(int64(len(src)))
	if err != nil {
		return nil, err
	}
	dst := make([]byte, size)
	bs := t.BlockSize()
	ts := t.TypeSize()
	workers := runtime.GOMAXPROCS(0)
	n := int64(len(src))
	if bs <= 1 || ts <= 0 || n < parallelThreshold || workers <= 1 {
		if err := QuantizeTo(t, src, dst); err != nil {
			return nil, err
		}
		return dst, nil
	}
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstErr error
	for _, r := range ranges(n, bs, workers) {
		i, j := r[0], r[1]
		wg.Add(1)
		go func() {
			defer wg.Done()
			byteOff := i / int64(bs) * int64(ts)
			byteEnd := j / int64(bs) * int64(ts)
			if err := QuantizeTo(t, src[i:j], dst[byteOff:byteEnd]); err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	return dst, nil
}
