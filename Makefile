# This is not the build system, just a helper to run common development commands.
# Make sure to first initialize the build system with:
#     make dev-install

PYTHON ?= python
BUILD_DIR := $(shell cd python; $(PYTHON) -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
TRITON_OPT := $(BUILD_DIR)/bin/triton-opt
PYTEST := $(PYTHON) -m pytest

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

.PHONY: test-python
test-unit: all
	cd python/test/unit && $(PYTEST) -vs language/test_core.py -k 'test_bin_op['

.PHONY: test-regression
test-regression: all
	$(PYTEST) -s -n 8 python/test/regression

.PHONY: test-interpret
test-interpret: all
	cd python/test/unit && TRITON_INTERPRET=1 $(PYTEST) -s -n 16 -m interpreter cuda language/test_core.py language/test_standard.py \
		language/test_random.py language/test_block_pointer.py language/test_subprocess.py language/test_line_info.py \
		language/test_tuple.py runtime/test_autotuner.py::test_kwargs[False] \
		../../tutorials/06-fused-attention.py::test_op --device=cpu

.PHONY: test-proton
test-proton: all
	$(PYTEST) -s -n 8 third_party/proton/test

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
	$(PYTHON) -m pip install -e python --no-build-isolation -v

.PHONY: dev-install
.NOPARALLEL: dev-install
dev-install: dev-install-requires dev-install-triton

# Updating lit tests

.PHONY: golden-samples
golden-samples: triton-opt
	$(TRITON_OPT) test/TritonGPU/samples/simulated-grouped-gemm.mlir.in -tritongpu-loop-scheduling -tritongpu-pipeline -canonicalize | \
		$(PYTHON) utils/generate-test-checks.py --source test/TritonGPU/samples/simulated-grouped-gemm.mlir.in --source_delim_regex="\bmodule" \
		-o test/TritonGPU/samples/simulated-grouped-gemm.mlir
	$(TRITON_OPT) test/TritonGPU/samples/descriptor-matmul-pipeline.mlir.in -tritongpu-loop-scheduling -tritongpu-pipeline -canonicalize | \
		$(PYTHON) utils/generate-test-checks.py --source test/TritonGPU/samples/descriptor-matmul-pipeline.mlir.in --source_delim_regex="\bmodule" \
		-o test/TritonGPU/samples/descriptor-matmul-pipeline.mlir
