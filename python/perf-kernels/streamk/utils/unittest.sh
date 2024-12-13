#! /bin/bash
git config --global --add safe.directory /__w/triton/triton
python3 ./python/perf-kernels/streamk/tune_streamk.py --gemm_size_file ./python/perf-kernels/streamk/utils/streamk_unit_test_sizes.yaml -dtype_a bf16 -dtype_b bf16 -dtype_c bf16 --compare_wo_tuning
python3 ./python/perf-kernels/streamk/tune_streamk.py --gemm_size_file ./python/perf-kernels/streamk/utils/streamk_unit_test_sizes.yaml --compare_wo_tuning
