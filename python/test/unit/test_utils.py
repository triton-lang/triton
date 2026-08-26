import pytest

from triton._utils import is_power_of_two, validate_block_shape


def test_is_power_of_two():
    assert is_power_of_two(1)
    assert is_power_of_two(2)
    assert is_power_of_two(8)
    assert is_power_of_two(1024)
    # 0 is not a power of two; x & (x - 1) == 0 alone wrongly accepts it.
    assert not is_power_of_two(0)
    assert not is_power_of_two(3)
    assert not is_power_of_two(6)
    assert not is_power_of_two(-4)


def test_validate_block_shape_rejects_zero():
    # validate_block_shape promises every element is a power of 2, but a 0
    # element used to slip through because is_power_of_two(0) returned True.
    with pytest.raises(ValueError, match="must be a power of 2"):
        validate_block_shape([0])
    with pytest.raises(ValueError, match="must be a power of 2"):
        validate_block_shape([8, 0])


def test_validate_block_shape_accepts_powers_of_two():
    assert validate_block_shape([8, 16]) == 128


@pytest.fixture
def one_hot_xor_report(tmp_path):
    """A small report with consistent measurements and real source digests."""
    import ast
    import copy
    import hashlib
    import pathlib
    import runpy

    root = pathlib.Path(__file__).resolve().parents[3]
    benchmark_path = pathlib.Path("python/test/microbenchmark/one_hot_xor_reduction.py")
    benchmark_source = (root / benchmark_path).read_text()
    kernel = next(node for node in ast.parse(benchmark_source).body
                  if isinstance(node, ast.FunctionDef) and node.name == "_classic_one_hot_xor_kernel")
    start = min(node.lineno for node in [kernel, *kernel.decorator_list]) - 1
    program_source = "".join(benchmark_source.splitlines(keepends=True)[start:kernel.end_lineno])
    sources = {
        str(benchmark_path): benchmark_source,
        "third_party/nvidia/backend/compiler.py": "# test backend\n",
        "lib/Dialect/TritonGPU/Transforms/OptimizeOneHotXorReduction.cpp": "// test pass\n",
        "lib/Dialect/TritonGPU/Transforms/CMakeLists.txt": "# test build\n",
        "include/triton/Dialect/TritonGPU/Transforms/Passes.td": "// test registration\n",
        "python/src/passes.cc": "// test binding\n",
        "python/triton/_C/libtriton.so": "test binary\n",
        "python/triton/__init__.py": "# test frontend\n",
    }
    digests = {}
    for relative, source in sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        digests[relative] = hashlib.sha256(source.encode()).hexdigest()

    def instructions(enabled):
        return {
            "ptx_total": 117 if enabled else 212,
            "ptx_shuffle": 32 if enabled else 0,
            "ptx_shuffle_indexed": 32 if enabled else 0,
            "ptx_lop3": 31 if enabled else 0,
            "ptx_redux": 0 if enabled else 32,
            "ptx_barrier": 0,
            "ptx_global_load": 1,
            "ptx_global_store": 1,
            "ptx_shared_load": 0,
            "ptx_shared_store": 0,
            "ttgir_reduce": 32,
            "ttgir_global_load": 1,
            "llir_shuffle": 32 if enabled else 0,
            "registers": 20 if enabled else 30,
            "shared_bytes": 0,
        }

    def timing(size, latency):
        return {
            "samples_us": [latency] * 10,
            "median_us": latency,
            "mean_us": latency,
            "min_us": latency,
            "max_us": latency,
            "p20_us": latency,
            "p80_us": latency,
            "relative_stddev": 0,
            "throughput_gigaelements_per_second": size / (latency * 1000),
            "effective_bandwidth_gb_per_second": size * 8 / (latency * 1000),
        }

    configuration = {
        "frontend": "classic",
        "basis_mode": "computed",
        "basis_storage": "registers",
        "basis_global_loads": 0,
        "optimization_observed": True,
        "block_size": 128,
        "seed": 0x31415926,
    }
    variants = {
        name: {"correct": True, "instructions": instructions(enabled)}
        for name, enabled in (("baseline", False), ("optimized", True))
    }
    cases = []
    for case_index, size in enumerate((128, 256)):
        case = {**configuration, "elements": size, "variants": copy.deepcopy(variants), "speedup": 2.0}
        for name, latency in (("baseline", 2.0), ("optimized", 1.0)):
            case["variants"][name]["timing"] = timing(size, latency)
        case["measurement_order"] = ["baseline", "optimized"] if case_index == 0 else ["optimized", "baseline"]
        cases.append(case)
    boundary_sizes = [1, 31, 32, 33, 127, 128, 129, 255, 257, 1000, 4097]
    report = {
        "schema_version": 4,
        "run": {"requested_sizes": [128, 256], "complete": True},
        "method": {
            "source_compilation": "same checkout @triton.jit source compiled twice through the real NVIDIA pipeline",
            "pre_pass_ir_identity": "exact byte equality of optimization-OFF and optimization-ON TTIR and TTGIR",
            "jit_cache_isolation": "fresh temporary compiler cache plus unused BENCHMARK_VARIANT constexpr",
            "program_source": program_source,
            "bases_source": "computed-in-registers-no-basis-buffer",
            "basis_global_memory_loads": 0,
            "effective_bandwidth_bytes_per_element": 8,
            "block_size": 128,
            "seed": configuration["seed"],
            "timing_samples": 10,
            "timing_repetitions_ms": 100,
            "timing_cache_policy": {"reuse_input_output_buffers": True, "flush_l2": False},
        },
        "environment": {
            "provenance": {
                "checkout_root": str(tmp_path),
                "triton_module": str(tmp_path / "python/triton/__init__.py"),
                "nvidia_backend_module": str(tmp_path / "third_party/nvidia/backend/compiler.py"),
                "native_extension": str(tmp_path / "python/triton/_C/libtriton.so"),
                "kernel_source": str(tmp_path / benchmark_path),
                "kernel_source_sha256": hashlib.sha256(program_source.encode()).hexdigest(),
                "benchmark_source_sha256": digests[str(benchmark_path)],
                "nvidia_backend_sha256": digests["third_party/nvidia/backend/compiler.py"],
                "native_extension_sha256": digests["python/triton/_C/libtriton.so"],
                "native_source_sha256": {
                    path: digest
                    for path, digest in digests.items()
                    if path.endswith((".cpp", ".cc", ".td", "CMakeLists.txt"))
                },
            }
        },
        "cases": cases,
        "boundary_correctness": {
            "sizes":
            boundary_sizes,
            "passed":
            11,
            "total":
            11,
            "masked_store_guard_int32":
            -0x13579BDF,
            "guard_elements_per_case":
            128,
            "cases": [{
                **configuration, "elements": size, "baseline_correct": True, "optimized_correct": True,
                "masked_store_guard_preserved": True, "variants": copy.deepcopy(variants)
            } for size in boundary_sizes],
        },
        "summary": {
            "performance_cases": 2,
            "bit_exact_performance_cases": 2,
            "optimized_cases": 2,
            "geomean_speedup": 2.0,
            "large_problem_geomean_speedup": None,
            "large_problem_minimum_elements": 1048576,
            "peak_speedup": 2.0,
            "peak_frontend": "classic",
            "peak_basis_mode": "computed",
            "peak_elements": 128,
        },
    }
    plot = runpy.run_path(str(root / "python/test/microbenchmark/plot_one_hot_xor_reduction.py"))
    return plot["_cases"], report, tmp_path


def test_one_hot_xor_report_accepts_consistent_measurements(one_hot_xor_report):
    validate, report, checkout = one_hot_xor_report
    assert [case["elements"] for case in validate(report, checkout=checkout)] == [128, 256]


def test_one_hot_xor_report_supports_python310_hashing(one_hot_xor_report, monkeypatch):
    import hashlib

    validate, report, checkout = one_hot_xor_report
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    assert len(validate(report, checkout=checkout)) == 2


@pytest.mark.parametrize("field, value", [
    (("schema_version", ), 5),
    (("run", "complete"), False),
    (("cases", 0, "variants", "optimized", "correct"), False),
    (("cases", 0, "speedup"), 999.0),
    (("cases", 0, "variants", "optimized", "timing", "median_us"), 999.0),
    (("cases", 0, "variants", "optimized", "timing", "samples_us", 0), float("nan")),
    (("cases", 0, "variants", "optimized", "timing", "throughput_gigaelements_per_second"), 999.0),
    (("cases", 1, "elements"), 128),
    (("summary", "performance_cases"), 18),
    (("boundary_correctness", "cases", 0, "frontend"), "gluon"),
    (("boundary_correctness", "cases", 0, "basis_storage"), "HBM"),
    (("boundary_correctness", "cases", 0, "basis_global_loads"), 9),
    (("boundary_correctness", "cases", 0, "optimization_observed"), False),
    (("boundary_correctness", "cases", 0, "variants", "optimized", "instructions", "ptx_shuffle_indexed"), 0),
    (("environment", "provenance", "native_extension_sha256"), "0" * 64),
    (("environment", "provenance", "nvidia_backend_sha256"), "1" * 64),
])
def test_one_hot_xor_report_rejects_inconsistent_evidence(one_hot_xor_report, field, value):
    validate, report, checkout = one_hot_xor_report
    target = report
    for component in field[:-1]:
        target = target[component]
    target[field[-1]] = value
    with pytest.raises(ValueError):
        validate(report, checkout=checkout)


def test_one_hot_xor_report_rejects_partial_run(one_hot_xor_report):
    validate, report, checkout = one_hot_xor_report
    report["cases"].pop()
    with pytest.raises(ValueError):
        validate(report, checkout=checkout)


def test_one_hot_xor_report_rejects_oversized_inputs(one_hot_xor_report):
    validate, report, checkout = one_hot_xor_report
    report["run"]["requested_sizes"][0] = 1 << 31
    report["cases"][0]["elements"] = 1 << 31
    with pytest.raises(ValueError, match="32-bit"):
        validate(report, checkout=checkout)


def test_one_hot_xor_report_rejects_escaping_provenance(one_hot_xor_report):
    validate, report, checkout = one_hot_xor_report
    report["environment"]["provenance"]["native_extension"] = f"{checkout}/../foreign/libtriton.so"
    with pytest.raises(ValueError):
        validate(report, checkout=checkout)


def test_one_hot_xor_report_rejects_changed_compiler_source(one_hot_xor_report):
    validate, report, checkout = one_hot_xor_report
    (checkout / "lib/Dialect/TritonGPU/Transforms/OptimizeOneHotXorReduction.cpp").write_text("// changed source\n")
    with pytest.raises(ValueError):
        validate(report, checkout=checkout)


@pytest.mark.parametrize("filename, reference", [("custom figure.pdf", "custom%20figure.png"),
                                                 ("custom.PNG", "custom.PNG")])
def test_one_hot_xor_report_links_custom_figure(tmp_path, filename, reference):
    import pathlib
    import runpy

    root = pathlib.Path(__file__).resolve().parents[3]
    plot = runpy.run_path(str(root / "python/test/microbenchmark/plot_one_hot_xor_reduction.py"))
    figure = tmp_path / "plots" / filename
    report = tmp_path / "reports" / "result.md"
    assert plot["_figure_reference"](figure, report) == f"../plots/{reference}"


@pytest.fixture
def one_hot_xor_benchmark():
    import pathlib
    import runpy

    pytest.importorskip("torch")
    root = pathlib.Path(__file__).resolve().parents[3]
    return runpy.run_path(str(root / "python/test/microbenchmark/one_hot_xor_reduction.py"))


def test_one_hot_xor_benchmark_rejects_oversized_inputs(one_hot_xor_benchmark):
    import argparse

    parse_sizes = one_hot_xor_benchmark["_sizes"]
    assert parse_sizes(str((1 << 31) - 1)) == [(1 << 31) - 1]
    with pytest.raises(argparse.ArgumentTypeError, match="32-bit"):
        parse_sizes(str(1 << 31))


@pytest.mark.parametrize("override", ["binary", "pipeline"])
def test_one_hot_xor_benchmark_rejects_compiler_overrides(one_hot_xor_benchmark, monkeypatch, override):
    from triton import knobs

    monkeypatch.setattr(knobs.compilation, "override", override == "binary")
    monkeypatch.setattr(knobs.runtime, "add_stages_inspection_hook", (lambda: None) if override == "pipeline" else None)
    with pytest.raises(RuntimeError, match="OVERRIDE|pipeline hooks"):
        with one_hot_xor_benchmark["_isolated_compiler_cache"]():
            pass
