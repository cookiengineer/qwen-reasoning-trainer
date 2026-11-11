package inference

import (
	"math/rand"
	"strings"
	"testing"
)

func TestStopFilterStopsAndHoldsPartialSequences(t *testing.T) {
	var out strings.Builder
	f := newStopFilter([]string{"STOP"}, func(s string) error {
		out.WriteString(s)
		return nil
	})
	if err := f.Write("abcST"); err != nil {
		t.Fatal(err)
	}
	if got := out.String(); got != "ab" {
		t.Fatalf("after partial stop = %q, want %q", got, "ab")
	}
	if err := f.Write("OPtail"); err != nil {
		t.Fatal(err)
	}
	if !f.Stopped() {
		t.Fatal("filter did not stop")
	}
	if got := out.String(); got != "abc" {
		t.Fatalf("out = %q, want %q", got, "abc")
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if got := out.String(); got != "abc" {
		t.Fatalf("close emitted text after stop: %q", got)
	}
}

func TestStopFilterFlushesWithoutStops(t *testing.T) {
	var out strings.Builder
	f := newStopFilter(nil, func(s string) error {
		out.WriteString(s)
		return nil
	})
	if err := f.Write("hello "); err != nil {
		t.Fatal(err)
	}
	if err := f.Write("world"); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if got := out.String(); got != "hello world" {
		t.Fatalf("out = %q", got)
	}
}

func TestSampleTokenGreedy(t *testing.T) {
	logits := []float32{0.1, 0.9, 0.3}
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 20; i++ {
		if got := sampleToken(logits, rng, 0, 0, 1); got != 1 {
			t.Fatalf("greedy = %d, want 1", got)
		}
	}
}

func TestSampleTokenTopKAndSeed(t *testing.T) {
	logits := []float32{1, 2, 3, 4}
	a := rand.New(rand.NewSource(7))
	b := rand.New(rand.NewSource(7))
	for i := 0; i < 50; i++ {
		got := sampleToken(logits, a, 1, 1, 1)
		if got != 3 {
			t.Fatalf("topK=1 sampled %d, want 3", got)
		}
		if got != sampleToken(logits, b, 1, 1, 1) {
			t.Fatal("seeded sampling is not deterministic")
		}
	}
}

func TestSampleTokenTopP(t *testing.T) {
	logits := []float32{0, 0, 10}
	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 50; i++ {
		if got := sampleToken(logits, rng, 1, 0, 0.5); got != 2 {
			t.Fatalf("topP sampled %d, want 2", got)
		}
	}
}
