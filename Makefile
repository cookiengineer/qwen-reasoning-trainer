# Qwen3.8-27B Reasoning Trainer
#
# Common targets:
#   make            build the CLI (compiles shaders first)
#   make shaders    compile GLSL compute shaders to embedded SPIR-V
#   make test       run the full test suite
#   make check      fmt-check + vet + test
#   make verify-ref run the NumPy forward-pass reference check
#   make verify-data run the template + pre-tokenizer reference checks
#   make e2e-venv   create e2e-tests/.venv with the reference dependencies
#   make clean      remove build artifacts and compiled shaders

GO        ?= go
GLSLC     ?= glslc
SPIRV_VAL ?= spirv-val
# Interpreter used to create the e2e venv (override for a specific python3.x).
SYSTEM_PYTHON ?= python3

BIN_DIR := build
BIN     := $(BIN_DIR)/qwen-trainer
CMD     := ./cmd/qwen-trainer

SHADER_DIR := shaders
SPIRV_DIR  := internal/compute/vulkan/spirv
SHADER_SRC := $(wildcard $(SHADER_DIR)/*.comp)
SHADER_SPV := $(patsubst $(SHADER_DIR)/%.comp,$(SPIRV_DIR)/%.spv,$(SHADER_SRC))
# Shared GLSL includes (decode helpers, GEMM bodies); any change rebuilds all.
SHADER_INC := $(wildcard $(SHADER_DIR)/*.glsl)

GLSL_FLAGS  := --target-env=vulkan1.1 -O -I $(SHADER_DIR)
SPIRV_FLAGS := --target-env vulkan1.1

# Local Python environment for the opt-in external references.
E2E_DIR    := e2e-tests
E2E_REQS   := $(E2E_DIR)/requirements.txt
E2E_VENV   := $(E2E_DIR)/.venv
# Absolute so it works as an env var; `go test` runs with CWD set to the package.
E2E_PYTHON ?= $(CURDIR)/$(E2E_VENV)/bin/python
E2E_STAMP  := $(E2E_VENV)/.deps-ok

.PHONY: all build shaders test vet fmt fmt-check tidy clean check \
        verify-ref verify-data verify-quants e2e-venv inspect gpu-info

all: build

# Build the CLI. The .spv files are embedded via go:embed, so shaders must be
# present before compiling.
build: shaders
	$(GO) build -o $(BIN) $(CMD)

# Compile and validate each GLSL shader to SPIR-V.
$(SPIRV_DIR)/%.spv: $(SHADER_DIR)/%.comp $(SHADER_INC)
	@mkdir -p $(SPIRV_DIR)
	$(GLSLC) $(GLSL_FLAGS) -o $@ $<
	$(SPIRV_VAL) $(SPIRV_FLAGS) $@
	@echo "built $@"

shaders: $(SHADER_SPV)

test: shaders
	$(GO) test ./...

vet:
	$(GO) vet ./...

fmt:
	gofmt -w cmd internal e2e-tests

fmt-check:
	@out="$$(gofmt -l cmd internal e2e-tests)"; \
	if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi

tidy:
	$(GO) mod tidy

check: fmt-check vet test

# External cross-checks live in ./e2e-tests/<name> as standalone Go programs, one
# per check, run with `go run`. They use a project-local venv (E2E_VENV) with
# pinned deps, created on demand. Override the interpreter with
# `make verify-data E2E_PYTHON=...` (then create it yourself: e.g. a venv with
# numpy+jinja2+regex).
e2e-venv: $(E2E_STAMP)

$(E2E_STAMP): $(E2E_REQS)
	@echo "setting up $(E2E_VENV) ..."
	@test -d $(E2E_VENV) || $(SYSTEM_PYTHON) -m venv $(E2E_VENV)
	@$(E2E_VENV)/bin/python -m pip install --quiet --disable-pip-version-check --upgrade pip
	@$(E2E_VENV)/bin/python -m pip install --quiet --disable-pip-version-check -r $(E2E_REQS)
	@touch $(E2E_STAMP)

verify-ref: $(E2E_STAMP)
	QWEN38_REF_PYTHON=$(E2E_PYTHON) $(GO) run ./e2e-tests/verify-ref

verify-data: $(E2E_STAMP)
	QWEN38_REF_PYTHON=$(E2E_PYTHON) $(GO) run ./e2e-tests/verify-data

# Diff our block decoders and encoders against a vendored copy of llama.cpp's
# ggml-quants.c (needs a C compiler; self-contained).
verify-quants:
	$(GO) run ./e2e-tests/verify-quants

inspect: build
	$(BIN) inspect

gpu-info: build
	$(BIN) gpu-info

clean:
	rm -rf $(BIN_DIR)
	rm -f $(SHADER_SPV)
