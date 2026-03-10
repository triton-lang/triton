#!/usr/bin/env python3
"""
Run Triton core tests against the Apple MPS backend by category.

Usage:
  python third_party/apple/run_core_tests.py              # run all categories
  python third_party/apple/run_core_tests.py arith         # run one category
  python third_party/apple/run_core_tests.py arith memory  # run multiple
  python third_party/apple/run_core_tests.py --list        # list categories
"""
import subprocess, sys, os, shutil, re

TRITON_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_FILE = os.path.join(TRITON_ROOT, "python/test/unit/language/test_core.py")

# Logically grouped test categories
CATEGORIES = {
    "arith": [
        "test_bin_op", "test_addptr", "test_floordiv",
        "test_bitwise_op", "test_shift_op", "test_bin_op_constexpr",
    ],
    "compare": [
        "test_compare_op", "test_where", "test_where_broadcast",
        "test_clamp",
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
        "test_atomic_rmw",
    ],
    "dot": [
        "test_dot", "test_dot_without_load",
    ],
    "control": [
        "test_if", "test_if_else",
        "test_for_iv", "test_while", "test_nested_while",
    ],
    "tensor_ops": [
        "test_broadcast", "test_arange", "test_reshape",
        "test_expand_dims", "test_full",
        "test_permute", "test_transpose",
        "test_cat", "test_join", "test_split", "test_interleave",
    ],
    "misc": [
        "test_constexpr", "test_const",
        "test_shapes_as_params", "test_index1d",
        "test_value_specialization", "test_num_programs",
    ],
    "scan": [
        "test_scan_1d", "test_scan2d", "test_histogram",
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

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=TRITON_ROOT)

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
    args = [a for a in args if a not in ("--verbose", "-v", "--list")]

    if args:
        names = args
    else:
        names = list(CATEGORIES.keys())

    total_passed = total_failed = total_errors = 0

    for name in names:
        if name not in CATEGORIES:
            print(f"Unknown category '{name}', valid: {', '.join(CATEGORIES.keys())}")
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
