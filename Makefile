# This is not the build system, just a helper to run common development commands.
# Make sure to first initialize the build system with:
#     make dev-install

PYTHON ?= python3
ROOT_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD_DIR := $(shell PYTHONPATH="$(ROOT_DIR)/python" $(PYTHON) -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
INSTALL_DIR ?= $(dir $(BUILD_DIR))install
TRITON_OPT := $(BUILD_DIR)/bin/triton-opt
PYTEST := $(PYTHON) -m pytest
LLVM_BUILD_PATH ?= "$(ROOT_DIR)/.llvm-project/build"
NUM_PROCS ?= 8
TRITON_KERNELS_PATH := $(ROOT_DIR)/python/triton_kernels$(if $(PYTHONPATH),:$(PYTHONPATH))
FAST_NVIDIA_RUNNER := $(filter nvidia-h100 nvidia-gb200,$(RUNNER_TYPE))

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
ifneq ($(FAST_NVIDIA_RUNNER),)
	@set -e; \
	cd python/test/unit; \
	TRITON_CI_CACHE_PHASE=unit $(PYTEST) -n $(NUM_PROCS) \
		--ignore-glob='plugins/*' --ignore=test_debug.py --ignore=language/test_subprocess.py & \
	unit_main_pid=$$!; \
	TRITON_CI_CACHE_PHASE=unit $(PYTEST) -n $(UNIT_DEBUG_PROCS) \
		test_debug.py language/test_subprocess.py & \
	unit_subprocess_pid=$$!; \
	unit_status=0; \
	if ! wait "$$unit_main_pid"; then unit_status=1; fi; \
	if ! wait "$$unit_subprocess_pid"; then unit_status=1; fi; \
	exit "$$unit_status"
else
	cd python/test/unit && TRITON_CI_CACHE_PHASE=unit $(PYTEST) -n $(NUM_PROCS) --ignore-glob='plugins/*'
endif
	TRITON_CI_CACHE_PHASE=triton-kernels $(PYTEST) -n $(TRITON_KERNEL_PROCS) python/triton_kernels/tests/
	# Run attention separately to avoid out of gpu memory
	TRITON_CI_CACHE_PHASE=attention $(PYTEST) python/tutorials/06-fused-attention.py
	TRITON_ALWAYS_COMPILE=1 TRITON_DISABLE_LINE_INFO=0 LLVM_PASS_PLUGIN_PATH=python/triton/instrumentation/libGPUInstrumentationTestLib.so \
		TRITON_CI_CACHE_PHASE=instrumentation $(PYTEST) --capture=tee-sys -rfs -vvv python/test/unit/instrumentation/test_gpuhello.py

.PHONY: test-plugins
test-plugins: all
	TRITON_CI_CACHE_PHASE=plugins $(PYTEST) -vvv python/test/unit/plugins

.PHONY: test-gluon
ifneq ($(FAST_NVIDIA_RUNNER),)
TRITON_KERNEL_PROCS ?= 3
UNIT_DEBUG_PROCS ?= 4
GLUON_PROCS ?= 8
GLUON_CONSAN_PROCS ?= 4
GLUON_FPSAN_PROCS ?= 8
else
TRITON_KERNEL_PROCS ?= 6
GLUON_PROCS ?= $(NUM_PROCS)
endif
ifeq ($(RUNNER_TYPE),nvidia-gb200)
GLUON_EXAMPLE_PROCS ?= 8
else
GLUON_EXAMPLE_PROCS ?= 2
endif
test-gluon: all
	TRITON_CI_CACHE_PHASE=gluon $(PYTEST) -n $(GLUON_PROCS) \
		python/test/gluon/ python/tutorials/gluon/ \
		$(if $(FAST_NVIDIA_RUNNER),--ignore=python/test/gluon/test_consan.py --ignore=python/test/gluon/test_fpsan.py)
ifneq ($(FAST_NVIDIA_RUNNER),)
	TRITON_CI_CACHE_PHASE=gluon $(PYTEST) -n $(GLUON_CONSAN_PROCS) python/test/gluon/test_consan.py
	TRITON_CI_CACHE_PHASE=gluon $(PYTEST) -n $(GLUON_FPSAN_PROCS) python/test/gluon/test_fpsan.py
endif
	PYTHONPATH="$(TRITON_KERNELS_PATH)" TRITON_CI_CACHE_PHASE=gluon-examples \
		$(PYTEST) -n $(GLUON_EXAMPLE_PROCS) python/examples/gluon/

WARMUP_PROCS ?= $(NUM_PROCS)
ifneq ($(FAST_NVIDIA_RUNNER),)
WARMUP_CAPTURE_PROCS ?= 2
else
WARMUP_CAPTURE_PROCS ?= 4
endif
WARMUP_AUXILIARY_CAPTURE_PROCS ?= 2
WARMUP_UNIT_TESTS := \
	python/test/unit/language/test_matmul.py \
	python/test/unit/language/test_core.py::test_gather
WARMUP_TRITON_KERNEL_TESTS := python/triton_kernels/tests/test_matmul.py::test_op

ifneq ($(filter nvidia-h100 nvidia-gb200,$(RUNNER_TYPE)),)
WARMUP_UNIT_TESTS := \
	python/test/unit/language/test_matmul.py \
	python/test/unit/language/test_warp_specialization.py \
	python/test/unit/language/test_core.py::test_scaled_dot \
	python/test/unit/language/test_core.py::test_dot \
	python/test/unit/language/test_core.py::test_dot3d \
	python/test/unit/language/test_core.py::test_gather \
	python/test/unit/language/test_core.py::test_scan2d \
	python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reduce \
	python/test/unit/language/test_standard.py::test_sort
ifeq ($(RUNNER_TYPE),nvidia-gb200)
WARMUP_TRITON_KERNEL_TESTS += \
	python/triton_kernels/tests/test_reduce.py::test_op \
	python/triton_kernels/tests/test_topk.py::test_topk
endif
endif

# Broad scalar-language capture is slower than compiling under runtime xdist,
# so prewarm only the compile-dense unit tests.
.PHONY: test-warmup
test-warmup: all
	@set -e; \
	warmup_unit_procs=$$(( $(WARMUP_PROCS) / 8 )); \
	warmup_attention_procs=$$(( $(WARMUP_PROCS) / 16 )); \
	warmup_auxiliary_procs=0; \
	if [ "$$warmup_unit_procs" -lt 1 ]; then warmup_unit_procs=1; fi; \
	if [ "$$warmup_attention_procs" -lt 1 ]; then warmup_attention_procs=1; fi; \
	if [ "$(RUNNER_TYPE)" = "nvidia-gb200" ]; then \
		warmup_unit_procs=$$(( $(WARMUP_PROCS) / 8 )); \
		warmup_group_procs=$$(( $(WARMUP_PROCS) * 11 / 28 )); \
		warmup_auxiliary_procs=$$(( $(WARMUP_PROCS) / 14 )); \
		if [ "$$warmup_unit_procs" -lt 1 ]; then warmup_unit_procs=1; fi; \
		if [ "$$warmup_group_procs" -lt 1 ]; then warmup_group_procs=1; fi; \
		if [ "$$warmup_auxiliary_procs" -lt 1 ]; then warmup_auxiliary_procs=1; fi; \
		warmup_triton_kernels_procs=$$(( \
			$(WARMUP_PROCS) - warmup_unit_procs - warmup_group_procs - warmup_auxiliary_procs )); \
	elif [ "$(RUNNER_TYPE)" = "nvidia-h100" ]; then \
		warmup_unit_procs=$$(( $(WARMUP_PROCS) / 8 )); \
		warmup_group_procs=$$(( $(WARMUP_PROCS) * 2 / 7 )); \
		if [ "$$warmup_unit_procs" -lt 1 ]; then warmup_unit_procs=1; fi; \
		if [ "$$warmup_group_procs" -lt 1 ]; then warmup_group_procs=1; fi; \
		warmup_triton_kernels_procs=$$(( \
			$(WARMUP_PROCS) - warmup_unit_procs - warmup_group_procs )); \
	else \
		warmup_group_procs=$$warmup_attention_procs; \
		warmup_triton_kernels_procs=$$(( $(WARMUP_PROCS) - warmup_unit_procs - warmup_group_procs )); \
	fi; \
	if [ "$$warmup_triton_kernels_procs" -lt 1 ]; then warmup_triton_kernels_procs=1; fi; \
	warmup_triton_kernels_worker_procs=$$(( \
		(warmup_triton_kernels_procs + $(WARMUP_CAPTURE_PROCS) - 1) / $(WARMUP_CAPTURE_PROCS) )); \
	TRITON_CI_CACHE_PHASE=warmup-unit $(PYTEST) -s --tb=short \
		--warmup-only --warmup-workers "$$warmup_unit_procs" \
		$(WARMUP_UNIT_TESTS) & \
	warmup_unit_pid=$$!; \
	TRITON_CI_CACHE_PHASE=warmup-triton-kernels $(PYTEST) -s --tb=short \
		-n $(WARMUP_CAPTURE_PROCS) --dist=worksteal \
		--warmup-only --warmup-workers "$$warmup_triton_kernels_worker_procs" \
		$(WARMUP_TRITON_KERNEL_TESTS) & \
	warmup_triton_kernels_pid=$$!; \
	if [ "$(RUNNER_TYPE)" = "nvidia-gb200" ]; then \
		warmup_group_worker_procs=$$(( \
			(warmup_group_procs + $(WARMUP_CAPTURE_PROCS) - 1) / $(WARMUP_CAPTURE_PROCS) )); \
		PYTHONPATH="$(TRITON_KERNELS_PATH)" TRITON_CI_CACHE_PHASE=warmup-gluon \
			$(PYTEST) -s --tb=short -n $(WARMUP_CAPTURE_PROCS) --dist=worksteal \
			--warmup-only --warmup-workers "$$warmup_group_worker_procs" \
			--warmup-phase python/tutorials/06-fused-attention.py=warmup-attention \
			--warmup-phase python/examples/gluon=warmup-gluon-examples \
			python/examples/gluon/01-attention-forward.py::test_op_consan \
			python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op_consan \
			python/tutorials/06-fused-attention.py::test_op \
			python/test/gluon/test_core.py::test_mma_shared_inputs \
			python/examples/gluon/01-attention-forward.py::test_op \
			python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch \
			python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized \
			python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op & \
	elif [ "$(RUNNER_TYPE)" = "nvidia-h100" ]; then \
		warmup_group_worker_procs=$$(( \
			(warmup_group_procs + $(WARMUP_CAPTURE_PROCS) - 1) / $(WARMUP_CAPTURE_PROCS) )); \
		TRITON_CI_CACHE_PHASE=warmup-gluon \
			$(PYTEST) -s --tb=short -n $(WARMUP_CAPTURE_PROCS) --dist=worksteal \
			--warmup-only --warmup-workers "$$warmup_group_worker_procs" \
			--warmup-phase python/tutorials/06-fused-attention.py=warmup-attention \
			--warmup-phase python/test/regression=warmup-regression \
			python/tutorials/06-fused-attention.py::test_op \
			python/test/gluon/test_core.py::test_mma_shared_inputs \
			python/test/gluon/test_lowerings.py::test_convert1d_layouts \
			python/test/gluon/test_lowerings.py::test_convert2d_layouts \
			python/test/gluon/test_lowerings.py::test_reduce_layouts \
			python/test/regression & \
	else \
		TRITON_CI_CACHE_PHASE=warmup-attention $(PYTEST) -s --tb=short \
			--warmup-only --warmup-workers "$$warmup_group_procs" \
			python/tutorials/06-fused-attention.py::test_op & \
	fi; \
	warmup_group_pid=$$!; \
	warmup_auxiliary_pid=; \
	if [ "$$warmup_auxiliary_procs" -gt 0 ]; then \
		warmup_auxiliary_worker_procs=$$(( \
			(warmup_auxiliary_procs + $(WARMUP_AUXILIARY_CAPTURE_PROCS) - 1) / \
			$(WARMUP_AUXILIARY_CAPTURE_PROCS) )); \
		TRITON_CI_CACHE_PHASE=warmup-gluon $(PYTEST) -s --tb=short \
			-n $(WARMUP_AUXILIARY_CAPTURE_PROCS) --dist=worksteal \
			--warmup-only --warmup-workers "$$warmup_auxiliary_worker_procs" \
			--warmup-phase python/test/regression=warmup-regression \
			python/test/gluon/test_lowerings.py::test_convert1d_layouts \
			python/test/gluon/test_lowerings.py::test_convert2d_layouts \
			python/test/gluon/test_lowerings.py::test_reduce_layouts \
			python/test/regression & \
		warmup_auxiliary_pid=$$!; \
	fi; \
	warmup_status=0; \
	if ! wait "$$warmup_unit_pid"; then warmup_status=1; fi; \
	if ! wait "$$warmup_triton_kernels_pid"; then warmup_status=1; fi; \
	if ! wait "$$warmup_group_pid"; then warmup_status=1; fi; \
	if [ -n "$$warmup_auxiliary_pid" ] && ! wait "$$warmup_auxiliary_pid"; then warmup_status=1; fi; \
	exit "$$warmup_status"

.PHONY: test-gsan
test-gsan: all
	TRITON_CI_CACHE_PHASE=gsan TRITON_DISABLE_LINE_INFO=0 $(PYTEST) -n $(NUM_PROCS) python/test/gsan

.PHONY: test-regression
test-regression: all
	TRITON_CI_CACHE_PHASE=regression $(PYTEST) -n $(NUM_PROCS) python/test/regression

.PHONY: test-microbenchmark
test-microbenchmark: all
	$(PYTHON) python/test/microbenchmark/launch_overhead.py

.PHONY: test-interpret
test-interpret: all
	cd python/test/unit && TRITON_CI_CACHE_PHASE=interpreter TRITON_INTERPRET=1 $(PYTEST) -n 16 -m interpreter cuda language/test_core.py language/test_standard.py \
		language/test_random.py language/test_subprocess.py language/test_line_info.py \
		language/test_tuple.py runtime/test_launch.py runtime/test_autotuner.py::test_kwargs[False] \
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

.PHONY: dev-install-llvm
.NOPARALLEL: dev-install-llvm
dev-install-llvm:
	LLVM_BUILD_PATH=$(LLVM_BUILD_PATH) scripts/build-llvm-project.sh
	LLVM_INCLUDE_DIRS=$(LLVM_BUILD_PATH)/include \
		LLVM_LIBRARY_DIR=$(LLVM_BUILD_PATH)/lib \
		LLVM_SYSPATH=$(LLVM_BUILD_PATH) \
	$(MAKE) dev-install

# Package C++ artifacts

.PHONY: install
install:
	cmake --install $(BUILD_DIR) --prefix $(INSTALL_DIR)

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
