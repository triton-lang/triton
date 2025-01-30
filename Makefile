# This is not the build system, just a helper to run common development commands.
# Make sure to first initialize the build system with:
#     make dev-install

PYTHON := python3
BUILD_DIR := $(shell cd python; $(PYTHON) -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
TRITON_OPT := $(BUILD_DIR)/bin/triton-opt
PYTEST := $(PYTHON) -m pytest

.PHONY: all
all:
	ninja -C $(BUILD_DIR)

.PHONY: triton-opt
triton-opt:
	ninja -C $(BUILD_DIR) triton-opt

.PHONY: test-lit
test-lit:
	ninja -C $(BUILD_DIR) check-triton-lit-tests

.PHONY: test-cpp
test-cpp:
	ninja -C $(BUILD_DIR) check-triton-unit-tests

.PHONY: test-python
test-unit: all
	cd python/test/unit
	$(PYTEST) -s -n 8 --ignore=cuda/test_flashattention.py --ignore=language/test_line_info.py --ignore=language/test_subprocess.py --ignore=test_debug.py
	$(PYTEST) -s -n 8 language/test_subprocess.py
	$(PYTEST) -s -n 8 test_debug.py --forked
	TRITON_DISABLE_LINE_INFO=0 $(PYTEST) -s language/test_line_info.py
	# Run cuda/test_flashattention.py separately to avoid out of gpu memory
	$(PYTEST) -s cuda/test_flashattention.py
	TRITON_ALWAYS_COMPILE=1 TRITON_DISABLE_LINE_INFO=0 LLVM_PASS_PLUGIN_PATH=$(BUILD_DIR)/triton/instrumentation/libGPUInstrumentationTestLib.so \
		$(PYTEST) --capture=tee-sys -rfs -vvv instrumentation/test_gpuhello.py

.PHONY: test-regression
test-regression: all
	$(PYTEST) -s -n 8 python/test/regression

.PHONY: test-interpret
test-interpret: all
	cd python/test/unit
	TRITON_INTERPRET=1 $(PYTEST) -s -n 16 -m interpreter language/test_core.py language/test_standard.py \
		language/test_random.py language/test_block_pointer.py language/test_subprocess.py language/test_line_info.py \
		runtime/test_autotuner.py::test_kwargs[False] \
		../../tutorials/06-fused-attention.py::test_op --device=cpu

.PHONY: test-proton
test-proton: all
	$(PYTEST) -s third_party/proton/test

.PHONY: test-python
test-python: test-unit test-regression test-interpret test-proton

.PHONY: test-nogpu
test-nogpu: test-lit test-cpp

.PHONY: test
test: test-lit test-cpp test-python

.PHONY: dev-install
dev-install:
	# build-time dependencies
	$(PYTHON) -m pip install ninja cmake wheel pybind11
	# test dependencies
	$(PYTHON) -m pip install scipy numpy torch pytest lit pandas matplotlib llnl-hatchet
	$(PYTHON) -m pip install -e python --no-build-isolation -v

.PHONY: golden-samples
golden-samples: triton-opt
	$(TRITON_OPT) test/TritonGPU/samples/simulated-grouped-gemm.mlir.in -tritongpu-loop-scheduling -tritongpu-pipeline -canonicalize | \
		$(PYTHON) utils/generate-test-checks.py --source test/TritonGPU/samples/simulated-grouped-gemm.mlir.in --source_delim_regex="\bmodule" \
		-o test/TritonGPU/samples/simulated-grouped-gemm.mlir
	$(TRITON_OPT) test/TritonGPU/samples/descriptor-matmul-pipeline.mlir.in -tritongpu-loop-scheduling -tritongpu-pipeline -canonicalize | \
		$(PYTHON) utils/generate-test-checks.py --source test/TritonGPU/samples/descriptor-matmul-pipeline.mlir.in --source_delim_regex="\bmodule" \
		-o test/TritonGPU/samples/descriptor-matmul-pipeline.mlir
