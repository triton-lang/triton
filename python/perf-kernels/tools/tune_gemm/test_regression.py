import tune_gemm

import os
import yaml
import pytest
import warnings
from copy import deepcopy
import statistics


class TestRegression:

    @classmethod
    def setup_class(self):
        self.slowdown_threshold = 0.98

        self.test_results = []
        self.test_perf_ratios = []
        try:
            with open('gemm-performance-report-reference.yaml', 'r') as ref_file:
                self.reference_data = yaml.safe_load(ref_file)
        except FileNotFoundError:
            warnings.warn("No reference file found. There will be no regression detected!")
            self.reference_data = []

    @classmethod
    def teardown_class(self):
        with open('gemm-performance-report.yaml', 'w') as out_file:
            yaml.safe_dump(self.test_results, out_file)

    @pytest.mark.parametrize('config', [
        # M // BLOCK_M * N // BLOCK_N % 304 == 0
        # 1 workgroup / CU
        {
            'M': 4864, 'N': 4096, 'K': 4096, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N':
            256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2, 'waves_per_eu':
            0, 'matrix_instr_nonkdim': 16, 'kpack': 2
        },
        {
            'M': 4864, 'N': 4096, 'K': 4160, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N':
            256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2, 'waves_per_eu':
            0, 'matrix_instr_nonkdim': 16, 'kpack': 2
        },
        {
            'M': 4864, 'N': 4096, 'K': 4224, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N':
            256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2, 'waves_per_eu':
            0, 'matrix_instr_nonkdim': 16, 'kpack': 2
        },
        {
            'M': 4864, 'N': 4096, 'K': 4288, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N':
            256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2, 'waves_per_eu':
            0, 'matrix_instr_nonkdim': 16, 'kpack': 2
        },
        # 1 workgroup / CU masked loadK
        {
            'M': 4864, 'N': 4096, 'K': 4097, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N':
            256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2, 'waves_per_eu':
            0, 'matrix_instr_nonkdim': 16, 'kpack': 2
        },
        {
            'M': 4864, 'N': 4096, 'K': 4098, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N':
            256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2, 'waves_per_eu':
            0, 'matrix_instr_nonkdim': 16, 'kpack': 2
        },
        {
            'M': 4864, 'N': 4096, 'K': 4100, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N':
            256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2, 'waves_per_eu':
            0, 'matrix_instr_nonkdim': 16, 'kpack': 2
        },
        {
            'M': 4864, 'N': 4096, 'K': 4104, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N':
            256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2, 'waves_per_eu':
            0, 'matrix_instr_nonkdim': 16, 'kpack': 2
        },
        {
            'M': 4864, 'N': 4096, 'K': 4112, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N':
            256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2, 'waves_per_eu':
            0, 'matrix_instr_nonkdim': 16, 'kpack': 2
        },

        # 2 workgroups / CU
        {
            'M': 4864, 'N': 8192, 'K': 4096, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N':
            256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2, 'waves_per_eu':
            0, 'matrix_instr_nonkdim': 16, 'kpack': 2
        },
        {
            'M': 4864, 'N': 8192, 'K': 4160, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N':
            256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2, 'waves_per_eu':
            0, 'matrix_instr_nonkdim': 16, 'kpack': 2
        },
        {
            'M': 4864, 'N': 8192, 'K': 8192, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N':
            256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2, 'waves_per_eu':
            0, 'matrix_instr_nonkdim': 16, 'kpack': 2
        },
        {
            'M': 4864, 'N': 8192, 'K': 8256, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N':
            256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 2, 'waves_per_eu':
            0, 'matrix_instr_nonkdim': 16, 'kpack': 2
        },
    ], ids=lambda val: f"Config: {val}")
    def test_matmul_performance_regression(self, config, record_property):
        config.setdefault('instruction_sched_variant', 'none')

        M, N, K, col_a, col_b, runConfig = tune_gemm.process_item(deepcopy(config))

        rotating_buffer_size = config.setdefault('rotating_buffer_size', 0)
        icache_flush = config.setdefault('icache_flush', False)
        iters = config.setdefault('iters', 200)
        init_type = config.setdefault('init_type', 'randn')

        dtype_a = config.setdefault('dtype_a', 'fp16')
        dtype_b = config.setdefault('dtype_b', 'fp16')
        dtype_c = config.setdefault('dtype_c', 'fp16')

        bias_vector = config.get('bias_vector', False)
        bias_size = M if bias_vector else 0

        # Always compile if the user did not specify
        os.environ.setdefault('TRITON_ALWAYS_COMPILE', '1')

        tune_gemm.run_bash_command(f"rm -rf {tune_gemm.get_filename_myKernels()}")
        tune_gemm.generate_matmul_kernels([runConfig])

        gpus = [0]
        jobs = 1
        benchmark = True
        skipWarmup = False
        num_threads = 32
        verbose_level = 0

        minTime, bestConfig, compile_time, profile_time, post_time = tune_gemm.tune_gemm_config(
            M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, [runConfig], benchmark, jobs, iters,
            skipWarmup=skipWarmup, num_threads=num_threads, gpus=gpus, verbose=verbose_level,
            rotating_buffer_size=rotating_buffer_size, bias_size=bias_size, icache_flush=icache_flush)

        # post processing the numbers
        perf_tflops = lambda us: 2 * M * N * K * 1e-12 / (us * 1e-6)
        tri_tflops = perf_tflops(minTime)

        record_property("TFlops", f"{tri_tflops:.2f}")
        record_property("MinTime", f"{minTime:.2f}")

        # Add to global results
        self.test_results.append({'config': config, 'tflops': float(tri_tflops)})

        # Look for reference run
        reference_run = None
        for run in self.reference_data:
            if run['config'] == config:
                reference_run = run
                break

        if reference_run is not None:
            performance_ratio = tri_tflops / reference_run['tflops']
            self.test_perf_ratios.append(performance_ratio)
            regression_percent = (100.0 * (1.0 - performance_ratio))
            record_property("Performance difference (lower is better)", f"{regression_percent:.2f}%")
            assert performance_ratio > self.slowdown_threshold, f'Performance regressed by {regression_percent:.2f}% (threshold={((1.0 - self.slowdown_threshold) * 100.0 ):.2f}%)'
        else:
            pytest.skip("No performance reference found!")

    def test_overall_performance_difference(self, record_property):
        if len(self.test_perf_ratios) < 2:
            pytest.skip("Overall results will be tested if test count > 2")

        perf_diff_mean = statistics.geometric_mean(self.test_perf_ratios)
        regression_percent = (100.0 * (1.0 - perf_diff_mean))

        record_property("Overall performance difference (mean)", f"{regression_percent:.2f}%")
        assert perf_diff_mean > self.slowdown_threshold, f'Performance regressed by {regression_percent:.2f}% (threshold={((1.0 - self.slowdown_threshold) * 100.0 ):.2f}%)'
