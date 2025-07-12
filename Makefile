# This is not the build system, just a helper to run common development commands.
# Make sure to first initialize the build system with:
#     make dev-install

PYTHON ?= python
BUILD_DIR := $(shell cd python; $(PYTHON) -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
TRITON_OPT := $(BUILD_DIR)/bin/triton-opt
PYTEST := $(PYTHON) -m pytest
LLVM_BUILD_PATH ?= ".llvm-project/build"
NUM_PROCS ?= 8

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
	cd python/test/unit && $(PYTEST) -s -n $(NUM_PROCS) --ignore=language/test_line_info.py \
		--ignore=language/test_subprocess.py --ignore=test_debug.py
	$(PYTEST) -s -n $(NUM_PROCS) python/test/unit/language/test_subprocess.py
	$(PYTEST) -s -n $(NUM_PROCS) python/test/unit/test_debug.py --forked
	$(PYTEST) -s -n 8 python/triton_kernels/tests/
	TRITON_DISABLE_LINE_INFO=0 $(PYTEST) -s python/test/unit/language/test_line_info.py
	# Run attention separately to avoid out of gpu memory
	$(PYTEST) -vs python/tutorials/06-fused-attention.py
	$(PYTEST) -vs python/tutorials/gluon/01-attention-forward.py
	TRITON_ALWAYS_COMPILE=1 TRITON_DISABLE_LINE_INFO=0 LLVM_PASS_PLUGIN_PATH=python/triton/instrumentation/libGPUInstrumentationTestLib.so \
		$(PYTEST) --capture=tee-sys -rfs -vvv python/test/unit/instrumentation/test_gpuhello.py
	$(PYTEST) -s -n $(NUM_PROCS) python/test/gluon

.PHONY: test-gluon
test-gluon: all
	$(PYTEST) -s -n $(NUM_PROCS) python/test/gluon

.PHONY: test-regression
test-regression: all
	$(PYTEST) -s -n $(NUM_PROCS) python/test/regression

.PHONY: test-interpret
test-interpret: all
	cd python/test/unit && TRITON_INTERPRET=1 $(PYTEST) -s -n 16 -m interpreter cuda language/test_core.py language/test_standard.py \
		language/test_random.py language/test_block_pointer.py language/test_subprocess.py language/test_line_info.py \
		language/test_tuple.py runtime/test_autotuner.py::test_kwargs[False] \
		../../tutorials/06-fused-attention.py::test_op --device=cpu

.PHONY: test-proton
test-proton: all
	$(PYTEST) -s -n 8 third_party/proton/test --ignore=third_party/proton/test/test_override.py
	$(PYTEST) -s third_party/proton/test/test_override.py

.PHONY: test-python
test-python: test-unit test-regression test-interpret test-proton

.PHONY: test-nogpu
test-nogpu: test-lit test-cpp

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

.PHONY: dev-install-llvm
.NOPARALLEL: dev-install-llvm
dev-install-llvm:
	LLVM_BUILD_PATH=$(LLVM_BUILD_PATH) scripts/build-llvm-project.sh
	TRITON_BUILD_WITH_CLANG_LLD=1 TRITON_BUILD_WITH_CCACHE=0 \
		LLVM_INCLUDE_DIRS=$(LLVM_BUILD_PATH)/include \
		LLVM_LIBRARY_DIR=$(LLVM_BUILD_PATH)/lib \
		LLVM_SYSPATH=$(LLVM_BUILD_PATH) \
	$(MAKE) dev-install

# Updating lit tests

.PHONY: golden-samples
golden-samples: triton-opt
	$(TRITON_OPT) test/TritonGPU/samples/simulated-grouped-gemm.mlir.in -tritongpu-pipeline -canonicalize | \
		$(PYTHON) utils/generate-test-checks.py --source test/TritonGPU/samples/simulated-grouped-gemm.mlir.in --source_delim_regex="\bmodule" \
		-o test/TritonGPU/samples/simulated-grouped-gemm.mlir
	$(TRITON_OPT) test/TritonGPU/samples/descriptor-matmul-pipeline.mlir.in -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline -canonicalize | \
		$(PYTHON) utils/generate-test-checks.py --source test/TritonGPU/samples/descriptor-matmul-pipeline.mlir.in --source_delim_regex="\bmodule" \
		-o test/TritonGPU/samples/descriptor-matmul-pipeline.mlir
