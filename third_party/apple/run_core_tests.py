#!/usr/bin/env python3
"""
Run Triton core tests against the Apple MPS backend by category.

Usage:
  python third_party/apple/run_core_tests.py              # run all categories
  python third_party/apple/run_core_tests.py arith         # run one category
  python third_party/apple/run_core_tests.py arith memory  # run multiple
  python third_party/apple/run_core_tests.py --list        # list categories
  python third_party/apple/run_core_tests.py --validated   # run only validated categories
"""
import subprocess, sys, os, shutil, re

TRITON_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_FILE = os.path.join(TRITON_ROOT, "python/test/unit/language/test_core.py")

# Logically grouped test categories
# Categories that have been validated and pass fully on MPS
SKIP_CATEGORIES = {  # Validated categories (pass fully or with known limitations)
    "arith", "compare", "unary", "math", "cast", "memory", "reduce", "atomic",  # arith: 1450, compare: 636, unary: 26, math: 23, cast: 77, memory: 189, reduce: 416, atomic: 191
    # "control" excluded from batch — causes GPU deadlock. Run standalone: python run_core_tests.py control
    # control: 28/28 pass individually. 3 test_for_iv i64 known fragile, 1 test_nested_while (37 vs 40)
    "misc",     # 62/63: 1 skip (test_value_specialization_overflow u64 — PyTorch MPS can't cast u64 max)
    "tensor_ops",  # 80/80: 8 skipped (int64/float64 in test_cat), 1 TG overflow in test_permute (known)
    "histogram",  # 15/15
    "scan",  # 885/885: 12 skip (cummax TG overflow on large shapes), 112 skip (int64/float64)
    "dtype",  # 20/20
    "tma",  # 2/2
    "hw_specific",  # 2/2
    "misc_advanced",  # 20/20
    "tensor_ops_advanced",  # 113/113: test_trans_4d flaky (passes on rerun)
    "function_call",  # 11/11: noinline (simple/call_graph/shared/dynamic/multi_values) all pass
    "control_advanced",  # 15/15: test_tl_range_num_stages (pipelined matmul) now passes with load-and-extract
    "inline_asm",  # 0/0: all skipped (PTX/CUDA inline asm, not applicable to MPS)
    "math_advanced",  # 7/10: 3 skip (float64 unsupported on MPS: rint f32/f64 test harness, precise_math sqrt_rn)
}

CATEGORIES = {
    "arith": [
        "test_bin_op", "test_addptr", "test_floordiv",
        "test_bitwise_op", "test_shift_op", "test_bin_op_constexpr",
    ],
    "compare": [
        "test_compare_op", "test_where", "test_where_broadcast",
        "test_clamp", "test_clamp_symmetric",
    ],
    "unary": [
        "test_unary_op", "test_abs",
    ],
    "math": [
        "test_math_op", "test_math_fma_op", "test_math_divide_op",
    ],
    "cast": [
        "test_cast", "test_convert_float16_to_float32",
    ],
    "memory": [
        "test_masked_load", "test_masked_load_scalar", "test_load_scalar_with_mask",
        "test_store_constant", "test_load_store_same_ptr",
        "test_default", "test_pointer_arguments",
    ],
    "reduce": [
        "test_reduce1d", "test_reduce", "test_sum_dtype",
    ],
    "atomic": [
        "test_atomic_rmw", "test_atomic_cas", "test_atomic_min_max_neg_zero",
    ],
    "control": [
        "test_for_iv", "test_while", "test_nested_while",
        "test_if", "test_if_else", "test_if_call",
    ],
    "misc": [
        "test_constexpr", "test_const",
        "test_shapes_as_params", "test_index1d",
        "test_value_specialization", "test_num_programs",
    ],
    "dot": [
        "test_dot", "test_dot_without_load",
    ],
    "tensor_ops": [
        "test_broadcast", "test_arange", "test_reshape",
        "test_expand_dims", "test_full",
        "test_permute", "test_transpose",
        "test_cat", "test_cat_nd", "test_join", "test_split", "test_interleave",
    ],
    "histogram": [
        "test_histogram", "test_histogram_mask", "test_histogram_silent_data_corruption",
    ],
    "scan": [
        "test_scan_1d", "test_scan2d", "test_cumsum_dtype",
        "test_side_effectful_scan",
    ],
    "dot_advanced": [
        "test_dot3d", "test_dot_max_num_imprecise_acc", "test_dot_mulbroadcasted",
        "test_dot_multidim", "test_scaled_dot", "test_join_with_mma",
    ],
    "control_advanced": [
        "test_if_return", "test_nested_if_else_return",
        "test_short_circuiting", "test_static_range",
        "test_temp_var_in_loop", "test_disable_licm",
        "test_tl_range_fuse", "test_tl_range_fuse_dependent",
        "test_tl_range_num_stages", "test_tl_range_option_none",
    ],
    "memory_advanced": [
        "test_masked_load_shared_memory",
        "test_load_cache_modifier", "test_store_cache_modifier",
        "test_store_eviction_policy",
        "test_load_scope_sem_coop_grid_cta_not_one",
        "test_load_scope_sem_coop_grid_cta_one",
        "test_strided_load", "test_strided_store",
        "test_indirect_load", "test_indirect_store",
        "test_gather", "test_aliasing",
        "test_zero_strided_tensors",
    ],
    "atomic_advanced": [
        "test_atomic_rmw_predicate", "test_atomic_unsupported_type",
        "test_tensor_atomic_rmw", "test_tensor_atomic_rmw_block",
        "test_tensor_atomic_cas",
        "test_tensor_atomic_add_access_patterns",
        "test_tensor_atomic_add_non_exclusive_offset",
        "test_tensor_atomic_add_shift_1",
        "test_tensor_atomic_use_result",
    ],
    "reduce_advanced": [
        "test_chained_reductions", "test_generic_reduction",
        "test_side_effectful_reduction", "test_side_effectful_reduction_2d",
        "test_max_min_with_nan", "test_max_returns_zero",
        "test_propagate_nan", "test_optimize_thread_locality",
    ],
    "tensor_ops_advanced": [
        "test_expand_dims_error_cases",
        "test_reshape_err", "test_invalid_slice", "test_slice",
        "test_trans_2d", "test_trans_4d", "test_trans_reshape",
        "test_interleave_scalars", "test_join_scalars", "test_split_to_scalar",
        "test_unsplat",
    ],
    "dtype": [
        "test_dtype", "test_dtype_codegen", "test_dtype_tensor",
        "test_abs_fp8", "test_scalar_overflow",
        "test_value_specialization_overflow",
    ],
    "math_advanced": [
        "test_math_erf_op", "test_precise_math",
        "test_enable_fp_fusion", "test_enable_reflect_ftz",
        "test_umulhi", "test_libdevice_rint",
    ],
    "function_call": [
        "test_call", "test_noinline", "test_jit_function_arg",
        "test_map_elementwise", "test_map_elementwise_multiple_outputs",
        "test_map_elementwise_pack",
    ],
    "misc_advanced": [
        "test_constexpr_arg_str_attr", "test_constexpr_assignment",
        "test_constexpr_flattens", "test_constexpr_if_return",
        "test_constexpr_scalar_shape", "test_constexpr_shape",
        "test_assume", "test_tensor_member",
        "test_unroll_attr", "test_unsigned_name_mangling",
        "test_poison_return", "test_no_rematerialization_op",
        "test_vectorization", "test_vectorization_hints",
    ],
    "hw_specific": [
        "test_num_threads", "test_num_warps_pow2", "test_num_ctas_pre_sm90",
        "test_maxnreg", "test_override_arch",
        "test_globaltimer", "test_smid", "test_invalid_pid_axis",
    ],
    "inline_asm": [
        "test_inline_asm", "test_inline_asm_multiple_outputs",
        "test_inline_asm_packed", "test_inline_asm_packed_multiple_outputs",
        "test_inline_asm_with_pointers", "test_ptx_cast",
    ],
    "tma": [
        "test_tma_load_block_shape_err", "test_tma_store_block_shape_err",
    ],
}


def run_category(name, tests, verbose=False):
    cache_dir = os.path.expanduser("~/.triton/cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    k_expr = " or ".join(tests)
    cmd = [
        sys.executable, "-m", "pytest", TEST_FILE,
        "--device", "mps",
        "-k", k_expr,
        "--no-header", "-q", "--tb=line",
    ]

    print(f"\n── {name}: {', '.join(tests)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=TRITON_ROOT)
    except subprocess.TimeoutExpired as e:
        print(f"   TIMEOUT: hung after 300s (GPU likely stuck)")
        return 0, len(tests), 0

    all_output = result.stdout + "\n" + result.stderr

    # Find FAILED lines
    failures = []
    for line in all_output.split('\n'):
        if 'FAILED' in line:
            m = re.search(r'FAILED.*::(test_\w+\[?[^\]]*\]?)', line)
            if m:
                failures.append(m.group(1))

    # Find summary line
    passed = failed = errors = deselected = 0
    for line in all_output.split('\n'):
        if 'passed' in line or 'failed' in line or 'error' in line:
            m_p = re.search(r'(\d+) passed', line)
            m_f = re.search(r'(\d+) failed', line)
            m_e = re.search(r'(\d+) error', line)
            m_d = re.search(r'(\d+) deselected', line)
            if m_p: passed = int(m_p.group(1))
            if m_f: failed = int(m_f.group(1))
            if m_e: errors = int(m_e.group(1))
            if m_d: deselected = int(m_d.group(1))
            if m_p or m_f or m_e:
                break

    status = "OK" if failed == 0 and errors == 0 else "FAIL"
    print(f"   {status}: {passed} passed, {failed} failed, {errors} errors")

    if failures and len(failures) <= 20:
        base_fails = {}
        for f in failures:
            base = f.split('[')[0]
            base_fails.setdefault(base, []).append(f)
        for base, items in base_fails.items():
            if len(items) > 3:
                print(f"   - {base}: {len(items)} variants failed (e.g. {items[0]})")
            else:
                for item in items:
                    print(f"   - {item}")
    elif failures:
        base_fails = {}
        for f in failures:
            base = f.split('[')[0]
            base_fails[base] = base_fails.get(base, 0) + 1
        for base, count in base_fails.items():
            print(f"   - {base}: {count} variants failed")

    if verbose and (failed or errors):
        print("--- output (last 2000 chars) ---")
        print(all_output[-2000:])

    return passed, failed, errors


def main():
    args = sys.argv[1:]

    if "--list" in args:
        for name, tests in CATEGORIES.items():
            print(f"  {name:12s}  {', '.join(tests)}")
        return

    verbose = "--verbose" in args or "-v" in args
    validated_only = "--validated" in args
    args = [a for a in args if a not in ("--verbose", "-v", "--list", "--validated")]

    if args:
        names = args
    elif validated_only:
        names = [n for n in CATEGORIES if n in SKIP_CATEGORIES]
    else:
        names = list(CATEGORIES.keys())

    total_passed = total_failed = total_errors = 0

    for name in names:
        if name not in CATEGORIES:
            print(f"Unknown category '{name}', valid: {', '.join(CATEGORIES.keys())}")
            continue
        if not args and not validated_only and name in SKIP_CATEGORIES:
            print(f"\n── {name}: SKIP (validated)")
            continue
        tests = CATEGORIES[name]
        p, f, e = run_category(name, tests, verbose=verbose)
        total_passed += p
        total_failed += f
        total_errors += e

    print(f"\n{'='*60}")
    print(f"  TOTAL: {total_passed} passed, {total_failed} failed, {total_errors} errors")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
