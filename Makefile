# This is not the build system, just a helper to run common development commands.
# Make sure to first initialize the build system with:
#     make dev-install

PYTHON ?= python3
ROOT_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD_DIR := $(shell PYTHONPATH="$(ROOT_DIR)/python" $(PYTHON) -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
TRITON_OPT := $(BUILD_DIR)/bin/triton-opt
PYTEST := $(PYTHON) -m pytest
LLVM_BUILD_PATH ?= "$(ROOT_DIR)/.llvm-project/build"
LLVM_BUILD ?= core
AMD_LLVM_BUILD_PATH ?= "$(ROOT_DIR)/.llvm-project-amd/build"
AMD_CODEGEN_LIBRARY = $(shell PYTHONPATH="$(ROOT_DIR)/python" $(PYTHON) -c 'from build_helpers import amd_codegen_library_name; print(amd_codegen_library_name())')
NUM_PROCS ?= 8
NUM_GPUS ?= 1
WARMUP_PROCS ?= $(NUM_PROCS)

# Incremental builds

.PHONY: all
all:
	ninja -C $(BUILD_DIR)

.PHONY: triton-opt
triton-opt:
	ninja -C $(BUILD_DIR) triton-opt

# Testing

.PHONY: test-lit
test-lit:
	ninja -C $(BUILD_DIR) check-triton-lit-tests

.PHONY: test-cpp
test-cpp:
	ninja -C $(BUILD_DIR) check-triton-unit-tests

.PHONY: test-unit
test-unit: all
	$(PYTHON) -m triton._test_runner suite unit --num-gpus $(NUM_GPUS) --num-procs $(NUM_PROCS)

.PHONY: test-plugins
test-plugins: all
	TRITON_CI_CACHE_PHASE=plugins $(PYTEST) -vvv python/test/unit/plugins

.PHONY: test-gluon
test-gluon: all
	$(PYTHON) -m triton._test_runner suite gluon --num-gpus $(NUM_GPUS) --num-procs $(NUM_PROCS)

.PHONY: test-warmup
test-warmup: all
	$(PYTHON) -m triton._test_runner warmup --warmup-procs $(WARMUP_PROCS)

.PHONY: test-gsan
test-gsan: all
	$(PYTHON) -m triton._test_runner suite gsan --num-gpus $(NUM_GPUS) --num-procs $(NUM_PROCS)

.PHONY: test-regression
test-regression: all
	TRITON_CI_CACHE_PHASE=regression $(PYTEST) -p triton._compile_warmup -n $(NUM_PROCS) python/test/regression

.PHONY: test-microbenchmark
test-microbenchmark: all
	$(PYTHON) python/test/microbenchmark/launch_overhead.py

.PHONY: test-interpret
test-interpret: all
	cd python/test/unit && TRITON_CI_CACHE_PHASE=interpreter TRITON_INTERPRET=1 $(PYTEST) -n 16 -m interpreter cuda language/test_core.py language/test_standard.py \
		language/test_random.py language/test_subprocess.py language/test_line_info.py \
		language/test_tuple.py test_debug.py runtime/test_launch.py runtime/test_autotuner.py::test_kwargs[False] \
		../../tutorials/06-fused-attention.py::test_op --device=cpu

.PHONY: test-proton
test-proton: all
	TRITON_CI_CACHE_PHASE=proton $(PYTEST) -n 8 third_party/proton/test --ignore=third_party/proton/test/test_override.py -k "not test_overhead and not test_hw_trace"
	TRITON_CI_CACHE_PHASE=proton-hw-trace $(PYTEST) third_party/proton/test/test_profile.py::test_hw_trace
	TRITON_CI_CACHE_PHASE=proton-override $(PYTEST) third_party/proton/test/test_override.py
	TRITON_CI_CACHE_PHASE=proton-overhead $(PYTEST) third_party/proton/test/test_instrumentation.py::test_overhead

.PHONY: test-python
test-python: test-unit test-plugins test-regression test-interpret test-proton

.PHONY: test-nogpu
test-nogpu: test-lit test-cpp
	TRITON_CI_CACHE_PHASE=gluon-frontend $(PYTEST) python/test/gluon/test_frontend.py
	TRITON_CI_CACHE_PHASE=triton-frontend $(PYTEST) python/test/unit/language/test_frontend.py

.PHONY: test
test: test-lit test-cpp test-python

# pip install-ing

.PHONY: dev-install-requires
dev-install-requires:
	$(PYTHON) -m pip install -r python/requirements.txt
	$(PYTHON) -m pip install -r python/test-requirements.txt


.PHONY: dev-install-torch
dev-install-torch:
	# install torch but ensure pytorch-triton isn't installed
	$(PYTHON) -m pip install torch
	$(PYTHON) -m pip uninstall triton pytorch-triton -y

.PHONY: dev-install-triton
dev-install-triton:
	$(PYTHON) -m pip install -e . --no-build-isolation -v

.PHONY: dev-install
.NOPARALLEL: dev-install
dev-install: dev-install-requires dev-install-triton

.PHONY: build-llvm
build-llvm:
	LLVM_BUILD_PATH=$(LLVM_BUILD_PATH) scripts/build-llvm-project.sh

.PHONY: build-amd-llvm
build-amd-llvm:
	AMD_LLVM_BUILD_PATH=$(AMD_LLVM_BUILD_PATH) scripts/build-amd-llvm.sh

LLVM_BUILD_TARGETS_core := build-llvm
LLVM_BUILD_TARGETS_amd := build-amd-llvm
LLVM_BUILD_TARGETS_all := build-llvm build-amd-llvm

.PHONY: dev-install-llvm
.NOPARALLEL: dev-install-llvm
dev-install-llvm: $(LLVM_BUILD_TARGETS_$(LLVM_BUILD))
	$(if $(filter $(LLVM_BUILD),core amd all),,$(error LLVM_BUILD must be core, amd, or all))
ifneq ($(filter $(LLVM_BUILD),core all),)
	LLVM_INCLUDE_DIRS=$(LLVM_BUILD_PATH)/include \
		LLVM_LIBRARY_DIR=$(LLVM_BUILD_PATH)/lib \
		LLVM_SYSPATH=$(LLVM_BUILD_PATH) \
	$(if $(filter all,$(LLVM_BUILD)),TRITON_AMD_CODEGEN_PATH="$(ROOT_DIR)/third_party/amd/backend/lib/$(AMD_CODEGEN_LIBRARY)") $(MAKE) dev-install
else
	# Reset cached core LLVM paths so installation uses the prebuilt core llvm.
	LLVM_SYSPATH= \
		LLVM_INCLUDE_DIRS= \
		LLVM_LIBRARY_DIR= \
		TRITON_AMD_CODEGEN_PATH="$(ROOT_DIR)/third_party/amd/backend/lib/$(AMD_CODEGEN_LIBRARY)" \
		TRITON_APPEND_CMAKE_ARGS="$(TRITON_APPEND_CMAKE_ARGS) -U LLVM_DIR -U MLIR_DIR -U LLD_DIR -U LLVM_INCLUDE_DIRS -U LLVM_LIBRARY_DIR" \
	$(MAKE) dev-install
endif

.PHONY: dev-install-amd-llvm
dev-install-amd-llvm:
	$(MAKE) dev-install-llvm LLVM_BUILD=amd

# Updating lit tests

.PHONY: golden-samples
golden-samples: triton-opt
	$(TRITON_OPT) test/TritonGPU/samples/simulated-grouped-gemm.mlir.in -tritongpu-pipeline -canonicalize | \
		$(PYTHON) utils/generate-test-checks.py --source test/TritonGPU/samples/simulated-grouped-gemm.mlir.in --source_delim_regex="\bmodule" \
		-o test/TritonGPU/samples/simulated-grouped-gemm.mlir
	$(TRITON_OPT) test/TritonGPU/samples/descriptor-matmul-pipeline.mlir.in -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline -canonicalize | \
		$(PYTHON) utils/generate-test-checks.py --source test/TritonGPU/samples/descriptor-matmul-pipeline.mlir.in --source_delim_regex="\bmodule" \
		-o test/TritonGPU/samples/descriptor-matmul-pipeline.mlir

# Documentation
#
.PHONY: docs-requirements
docs-requirements:
	$(PYTHON) -m pip install -r docs/requirements.txt -q

.PHONY: docs-only
docs-only:
	cd docs; PATH="$(BUILD_DIR):$(PATH)" $(PYTHON) -m sphinx . _build/html/main

.PHONY: docs
.NOPARALLEL: docs
docs: docs-requirements docs-only
