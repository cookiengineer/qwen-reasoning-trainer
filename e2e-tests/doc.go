// Package e2e holds the opt-in, external cross-checks for this project. They
// shell out to reference implementations (NumPy, Jinja2, Python `regex`) and are
// skipped unless their environment variable is set, so `go test ./...` stays
// pure Go and fast.
//
//	make verify-ref    # NumPy forward pass (QWEN38_VERIFY_REF=1)
//	make verify-data   # chat template + pre-tokenizer (QWEN38_VERIFY_DATA=1)
//
// The reference scripts live next to the tests in this directory.
package e2e
