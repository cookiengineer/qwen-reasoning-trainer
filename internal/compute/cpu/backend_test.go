package cpu_test

import (
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/cpu"
)

func mustUpload(t *testing.T, b compute.Backend, vals []float32, dims ...int) compute.Buffer {
	t.Helper()
	buf, err := b.Upload(&compute.Tensor{Dims: dims, Type: 0, F32: vals})
	if err != nil {
		t.Fatal(err)
	}
	return buf
}

func mustDownload(t *testing.T, b compute.Backend, buf compute.Buffer) []float32 {
	t.Helper()
	tens, err := b.Download(buf)
	if err != nil {
		t.Fatal(err)
	}
	return tens.F32
}

func TestBackendOps(t *testing.T) {
	b := cpu.New()
	defer b.Close()

	if b.Capabilities().Name != "cpu" {
		t.Fatal("unexpected backend name")
	}

	// MatMul: [2,2] x [2,2] from compute reference expectations.
	a := mustUpload(t, b, []float32{1, 2, 3, 4}, 2, 2)
	m := mustUpload(t, b, []float32{5, 6, 7, 8}, 2, 2)
	prod, err := b.MatMul(a, m)
	if err != nil {
		t.Fatal(err)
	}
	got := mustDownload(t, b, prod)
	want := []float32{17, 39, 23, 53}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("matmul[%d] = %v, want %v", i, got[i], w)
		}
	}

	// Unary silu.
	u := mustUpload(t, b, []float32{0, 1}, 2)
	su, err := b.Unary(compute.UnarySilu, u)
	if err != nil {
		t.Fatal(err)
	}
	if s := mustDownload(t, b, su); s[0] != 0 || s[1] < 0.73 || s[1] > 0.74 {
		t.Errorf("silu = %v", s)
	}

	// Binary mul.
	x := mustUpload(t, b, []float32{1, 2}, 2)
	y := mustUpload(t, b, []float32{3, 4}, 2)
	xy, err := b.Binary(compute.BinaryMul, x, y)
	if err != nil {
		t.Fatal(err)
	}
	if p := mustDownload(t, b, xy); p[0] != 3 || p[1] != 8 {
		t.Errorf("mul = %v", p)
	}

	// RMSNorm.
	rn, err := b.RMSNorm(mustUpload(t, b, []float32{3, 4}, 2), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	inv := float32(1 / 3.5355339)
	if r := mustDownload(t, b, rn); r[0] < 0.84 || r[0] > 0.85 {
		t.Errorf("rmsnorm = %v (inv %v)", r, inv)
	}

	// GetRows.
	table := mustUpload(t, b, []float32{1, 2, 3, 4, 5, 6}, 2, 3)
	rows, err := b.GetRows(table, []int32{2, 0})
	if err != nil {
		t.Fatal(err)
	}
	if r := mustDownload(t, b, rows); r[0] != 5 || r[3] != 2 {
		t.Errorf("getrows = %v", r)
	}

	// Softmax sums to 1.
	sm, err := b.Softmax(mustUpload(t, b, []float32{1, 2, 3}, 3))
	if err != nil {
		t.Fatal(err)
	}
	var sum float32
	for _, v := range mustDownload(t, b, sm) {
		sum += v
	}
	if sum < 0.999 || sum > 1.001 {
		t.Errorf("softmax sum = %v", sum)
	}

	// GatedDeltaNet smoke.
	q := mustUpload(t, b, []float32{1, 0}, 2, 1, 1)
	k := mustUpload(t, b, []float32{1, 0}, 2, 1, 1)
	v := mustUpload(t, b, []float32{1, 1}, 2, 1, 1)
	g := mustUpload(t, b, []float32{0}, 1, 1, 1)
	beta := mustUpload(t, b, []float32{1}, 1, 1, 1)
	state := mustUpload(t, b, []float32{0, 0, 0, 0}, 2, 2, 1)
	out, ns, err := b.GatedDeltaNet(q, k, v, g, beta, state)
	if err != nil {
		t.Fatal(err)
	}
	if o := mustDownload(t, b, out); o[0] < 0.70 || o[0] > 0.71 {
		t.Errorf("gdn out = %v", o)
	}
	if s := mustDownload(t, b, ns); s[0] != 1 || s[1] != 0 {
		t.Errorf("gdn state = %v", s)
	}
}
