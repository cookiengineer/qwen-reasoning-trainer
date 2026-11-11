package qwen38

import (
	"fmt"
	"math"
	"sync"
	"testing"
)

func TestConcurrentForwardMatchesSerial(t *testing.T) {
	cfg := tinyConfig()
	m := NewModel(NewRandom(cfg, 99))
	tokens := []int32{1, 5, 2, 7, 3, 9, 4}

	want, err := m.Forward(tokens, ForwardOptions{})
	if err != nil {
		t.Fatal(err)
	}

	const workers = 8
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := m.Forward(tokens, ForwardOptions{State: NewState(cfg)})
			if err != nil {
				errs <- err
				return
			}
			for j := range want.Logits.F32 {
				diff := math.Abs(float64(got.Logits.F32[j] - want.Logits.F32[j]))
				if diff > 1e-5 {
					errs <- fmt.Errorf("logit %d: concurrent %v != serial %v",
						j, got.Logits.F32[j], want.Logits.F32[j])
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
}
