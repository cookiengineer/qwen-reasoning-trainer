# Qwen3.8-27B Reasoning Trainer
#
# Common targets:
#   make            build the CLI (compiles shaders first)
#   make shaders    compile GLSL compute shaders to embedded SPIR-V
#   make test       run the full test suite
#   make check      fmt-check + vet + test
#   make verify-ref run the NumPy forward-pass reference check
#   make clean      remove build artifacts and compiled shaders

GO        ?= go
GLSLC     ?= glslc
SPIRV_VAL ?= spirv-val

BIN_DIR := build
BIN     := $(BIN_DIR)/qwen-trainer
CMD     := ./cmd/qwen-trainer

SHADER_DIR := shaders
SPIRV_DIR  := internal/compute/vulkan/spirv
SHADER_SRC := $(wildcard $(SHADER_DIR)/*.comp)
SHADER_SPV := $(patsubst $(SHADER_DIR)/%.comp,$(SPIRV_DIR)/%.spv,$(SHADER_SRC))

GLSL_FLAGS  := --target-env=vulkan1.1 -O
SPIRV_FLAGS := --target-env vulkan1.1

.PHONY: all build shaders test vet fmt fmt-check tidy clean check \
        verify-ref inspect gpu-info

all: build

# Build the CLI. The .spv files are embedded via go:embed, so shaders must be
# present before compiling.
build: shaders
	$(GO) build -o $(BIN) $(CMD)

# Compile and validate each GLSL shader to SPIR-V.
$(SPIRV_DIR)/%.spv: $(SHADER_DIR)/%.comp
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
	gofmt -w cmd internal

fmt-check:
	@out="$$(gofmt -l cmd internal)"; \
	if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi

tidy:
	$(GO) mod tidy

check: fmt-check vet test

# Independent NumPy cross-check of the forward pass (requires python3 + numpy).
verify-ref: shaders
	QWEN38_VERIFY_REF=1 $(GO) test -run TestReferenceNumpy -v ./internal/model/qwen38/...

inspect: build
	$(BIN) inspect

gpu-info: build
	$(BIN) gpu-info

clean:
	rm -rf $(BIN_DIR)
	rm -f $(SHADER_SPV)
