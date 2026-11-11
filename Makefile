# Qwen3.8-27B Reasoning Trainer
#
# The Go module lives under source/; all Go tooling runs there.
#
# Common targets:
#   make            build the CLI (compiles shaders first)
#   make server     build the OpenAI-compatible inference server
#   make build-all  build both binaries
#   make shaders    compile GLSL compute shaders to embedded SPIR-V
#   make test       run the full test suite
#   make check      fmt-check + vet + test
#   make verify-ref run the NumPy forward-pass reference check
#   make verify-data run the template + pre-tokenizer reference checks
#   make e2e-venv   create source/e2e-tests/.venv with the reference dependencies
#   make clean      remove build artifacts and compiled shaders

GO        ?= go
GLSLC     ?= glslc
SPIRV_VAL ?= spirv-val
# Interpreter used to create the e2e venv (override for a specific python3.x).
SYSTEM_PYTHON ?= python3

# Go module root. `go -C $(SRC)` runs the toolchain there.
SRC     := source
BIN_DIR := build
BIN     := $(BIN_DIR)/qwen-trainer
# Command path relative to $(SRC).
CMD     := ./cmds/qwen-trainer
BIN_SERVER := $(BIN_DIR)/qwen-server
CMD_SERVER := ./cmds/qwen-server

SHADER_DIR := $(SRC)/shaders
SPIRV_DIR  := $(SRC)/internal/compute/vulkan/spirv
SHADER_SRC := $(wildcard $(SHADER_DIR)/*.comp)
SHADER_SPV := $(patsubst $(SHADER_DIR)/%.comp,$(SPIRV_DIR)/%.spv,$(SHADER_SRC))
# Shared GLSL includes (decode helpers, GEMM bodies); any change rebuilds all.
SHADER_INC := $(wildcard $(SHADER_DIR)/*.glsl)

GLSL_FLAGS  := --target-env=vulkan1.1 -O -I $(SHADER_DIR)
SPIRV_FLAGS := --target-env vulkan1.1

# Local Python environment for the opt-in external references. Each check keeps
# its own requirements.txt under source/e2e-tests/<name>/; both are installed
# into one shared venv (redundancy is fine).
E2E_DIR       := $(SRC)/e2e-tests
E2E_REF_REQS  := $(E2E_DIR)/verify-ref/requirements.txt
E2E_DATA_REQS := $(E2E_DIR)/verify-data/requirements.txt
E2E_VENV      := $(E2E_DIR)/.venv
# Absolute so it works as an env var regardless of the caller's CWD.
E2E_PYTHON ?= $(CURDIR)/$(E2E_VENV)/bin/python
E2E_STAMP  := $(E2E_VENV)/.deps-ok

# Opt-in real-model tests. run-tests.sh does not sync the 27B GGUF; on the
# target host ./models is a symlink to the model directory. Override REAL_MODEL
# for a different checkpoint and PPL_TEXT for a different evaluation corpus.
REAL_MODEL    ?= $(CURDIR)/models/Qwen3.8-27B-UD-Q4_K_M.gguf
PPL_TEXT      ?= $(CURDIR)/datasets/wikitext2-test.txt
REAL_ENV      := QWEN38_REAL_DEVICE=1 QWEN38_MODEL="$(REAL_MODEL)"
REAL_CONTEXTS ?= 128k 256k 384k 512k
NEEDLE_CONTEXTS ?= 128k 256k 384k 512k

.PHONY: all build build-all server shaders test vet fmt fmt-check tidy clean check \
        verify-ref verify-data verify-quants e2e-venv inspect gpu-info \
        test-real test-real-parity test-real-bench test-real-quality test-real-needle \
        test-real-decode

all: build

# Build the CLI. The .spv files are embedded via go:embed, so shaders must be
# present before compiling.
build: shaders
	$(GO) -C $(SRC) build -o $(CURDIR)/$(BIN) $(CMD)

# Build the OpenAI-compatible inference server.
server: shaders
	$(GO) -C $(SRC) build -o $(CURDIR)/$(BIN_SERVER) $(CMD_SERVER)

build-all: build server

# Compile and validate each GLSL shader to SPIR-V.
$(SPIRV_DIR)/%.spv: $(SHADER_DIR)/%.comp $(SHADER_INC)
	@mkdir -p $(SPIRV_DIR)
	$(GLSLC) $(GLSL_FLAGS) -o $@ $<
	$(SPIRV_VAL) $(SPIRV_FLAGS) $@
	@echo "built $@"

shaders: $(SHADER_SPV)

test: shaders
	$(GO) -C $(SRC) test ./...

# Real-model device-vs-host parity and memory estimate checks (R1).
test-real: test-real-parity

test-real-parity: shaders
	$(REAL_ENV) $(GO) -C $(SRC) test ./internal/inference \
		-run 'TestRealDevice' -count=1 -v -timeout 2h

# Long-context prefill/decode benchmarks (R2). Each preset runs in its own
# process so resident host weights from the previous preset are reclaimed.
test-real-bench: shaders
	@for context in $(REAL_CONTEXTS); do \
		echo "=== real benchmark $$context ==="; \
		$(REAL_ENV) QWEN38_REAL_BENCH=1 QWEN38_BENCH_CONTEXT=$$context \
			$(GO) -C $(SRC) test ./internal/inference \
			-run 'TestRealLongContextBench' -count=1 -v -timeout 6h || exit $$?; \
	done

# Isolated steady-state decode benchmark (weights + attention + GDN).
test-real-decode: shaders
	$(REAL_ENV) QWEN38_REAL_DECODE=1 \
		$(GO) -C $(SRC) test ./internal/inference \
		-run 'TestRealDecodeBench' -count=1 -v -timeout 2h

# Q4_K vs Q8_0 KV quality on WikiText-2 (R3).
test-real-quality: shaders
	$(REAL_ENV) QWEN38_REAL_QUALITY=1 QWEN38_PPL_TEXT="$(PPL_TEXT)" \
		$(GO) -C $(SRC) test ./internal/inference \
		-run 'TestRealKVQuality' -count=1 -v -timeout 6h

# Optional long-context needle/retrieval checks, including YaRN 1.5/2.0 (R3).
test-real-needle: shaders
	@for context in $(NEEDLE_CONTEXTS); do \
		echo "=== real needle $$context ==="; \
		$(REAL_ENV) QWEN38_REAL_NEEDLE=1 QWEN38_NEEDLE_CONTEXT=$$context \
			QWEN38_PPL_TEXT="$(PPL_TEXT)" \
			$(GO) -C $(SRC) test ./internal/inference \
			-run 'TestRealNeedle' -count=1 -v -timeout 12h || exit $$?; \
	done

vet:
	$(GO) -C $(SRC) vet ./...

fmt:
	gofmt -w $(SRC)

fmt-check:
	@out="$$(gofmt -l $(SRC) | grep -v '^$(SRC)/e2e-tests/\.venv/')"; \
	if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi

tidy:
	$(GO) -C $(SRC) mod tidy

check: fmt-check vet test

# External cross-checks live in ./source/e2e-tests/<name> as standalone Go
# programs, one per check, run with `go run`. They use a project-local venv
# (E2E_VENV) with pinned deps, created on demand. Override the interpreter with
# `make verify-data E2E_PYTHON=...` (then create it yourself: e.g. a venv with
# numpy+jinja2+regex).
e2e-venv: $(E2E_STAMP)

$(E2E_STAMP): $(E2E_REF_REQS) $(E2E_DATA_REQS)
	@echo "setting up $(E2E_VENV) ..."
	@test -d $(E2E_VENV) || $(SYSTEM_PYTHON) -m venv $(E2E_VENV)
	@$(E2E_VENV)/bin/python -m pip install --quiet --disable-pip-version-check --upgrade pip
	@$(E2E_VENV)/bin/python -m pip install --quiet --disable-pip-version-check -r $(E2E_REF_REQS) -r $(E2E_DATA_REQS)
	@touch $(E2E_STAMP)

verify-ref: $(E2E_STAMP)
	QWEN38_REF_PYTHON=$(E2E_PYTHON) $(GO) -C $(SRC) run ./e2e-tests/verify-ref

verify-data: $(E2E_STAMP)
	QWEN38_REF_PYTHON=$(E2E_PYTHON) $(GO) -C $(SRC) run ./e2e-tests/verify-data

# Diff our block decoders and encoders against a vendored copy of llama.cpp's
# ggml-quants.c (needs a C compiler; self-contained).
verify-quants:
	$(GO) -C $(SRC) run ./e2e-tests/verify-quants

inspect: build
	$(BIN) inspect

gpu-info: build
	$(BIN) gpu-info

clean:
	rm -rf $(BIN_DIR)
	rm -f $(SHADER_SPV)
