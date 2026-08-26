"""Render the source-identical, register-computed-basis Triton benchmark.

    python python/test/microbenchmark/plot_one_hot_xor_reduction.py \
        benchmarks/one_hot_xor_reduction/results.json \
        --output benchmarks/one_hot_xor_reduction/figure.pdf \
        --report-output benchmarks/one_hot_xor_reduction/REPORT.md
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import pathlib
import posixpath
import statistics
import textwrap
from urllib.parse import quote

COLORS = {
    "baseline": "#A84A43",
    "optimized": "#286886",
    "speedup": "#267C65",
    "ink": "#243247",
    "muted": "#6A7888",
    "grid": "#E7ECF0",
}
LABELS = {"baseline": "optimization OFF", "optimized": "optimization ON"}
REGISTER_BASES = "computed-in-registers-no-basis-buffer"
BOUNDARY_SIZES = (1, 31, 32, 33, 127, 128, 129, 255, 257, 1000, 4097)
SCHEMA_VERSION = 4
TIMING_SAMPLES = 10
MAX_ELEMENTS = (1 << 31) - 1
NATIVE_SOURCE_PATHS = (
    "lib/Dialect/TritonGPU/Transforms/OptimizeOneHotXorReduction.cpp",
    "lib/Dialect/TritonGPU/Transforms/CMakeLists.txt",
    "include/triton/Dialect/TritonGPU/Transforms/Passes.td",
    "python/src/passes.cc",
)


def _human_elements(value: float, _position: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):g}M"
    if value >= 1024:
        return f"{value / 1024:g}K"
    return f"{value:g}"


def _positive_int(value, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _check_number(value, expected: float, label: str) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not math.isclose(value, expected, rel_tol=1e-9)):
        raise ValueError(f"{label} does not match the value recomputed from timing samples: {expected}")


def _normalized_path(value, label: str) -> pathlib.PurePosixPath:
    # Producer paths are resolved POSIX paths on Linux or macOS. Normalize them
    # lexically: a remote checkout need not exist on the rendering machine.
    if not isinstance(value, str) or "\x00" in value or not value.startswith("/"):
        raise ValueError(f"{label} must be an absolute POSIX path")
    return pathlib.PurePosixPath(posixpath.normpath(value))


def _check_digest(value, label: str) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value) or value == "0" * 64):
        raise ValueError(f"{label} must contain a SHA-256 digest, not a missing or placeholder value")


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _local_program_source(checkout: pathlib.Path) -> str:
    benchmark = checkout / "python/test/microbenchmark/one_hot_xor_reduction.py"
    module_source = benchmark.read_text()
    for node in ast.parse(module_source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_classic_one_hot_xor_kernel":
            start = min([node.lineno, *[decorator.lineno for decorator in node.decorator_list]]) - 1
            return "".join(module_source.splitlines(keepends=True)[start:node.end_lineno])
    raise ValueError(f"cannot locate the benchmarked Triton program in {benchmark}")


def _validate_provenance(report: dict, checkout: pathlib.Path) -> None:
    provenance = report.get("environment", {}).get("provenance", {})
    recorded_checkout = _normalized_path(provenance.get("checkout_root"), "checkout_root")
    recorded_paths = {}
    for label in ("triton_module", "nvidia_backend_module", "native_extension", "kernel_source"):
        path = _normalized_path(provenance.get(label), label)
        if path == recorded_checkout or not path.is_relative_to(recorded_checkout):
            raise ValueError(f"{label} is outside benchmark checkout {recorded_checkout}")
        recorded_paths[label] = path
    benchmark_relative = "python/test/microbenchmark/one_hot_xor_reduction.py"
    if recorded_paths["kernel_source"].relative_to(recorded_checkout).as_posix() != benchmark_relative:
        raise ValueError("kernel_source must identify this checkout's benchmark program")

    source = report.get("method", {}).get("program_source")
    if not isinstance(source, str) or hashlib.sha256(
            source.encode()).hexdigest() != provenance.get("kernel_source_sha256"):
        raise ValueError("displayed Triton source does not match its recorded source-compilation digest")
    native_digests = provenance.get("native_source_sha256", {})
    source_digests = {
        benchmark_relative: provenance.get("benchmark_source_sha256"),
        "third_party/nvidia/backend/compiler.py": provenance.get("nvidia_backend_sha256"),
        **{path: native_digests.get(path)
           for path in NATIVE_SOURCE_PATHS},
    }
    for relative_path, recorded_digest in source_digests.items():
        _check_digest(recorded_digest, relative_path)
        local_path = checkout / relative_path
        if not local_path.is_file():
            raise ValueError(f"cannot verify source digest; render from a checkout containing {local_path}")
        if hashlib.sha256(local_path.read_bytes()).hexdigest() != recorded_digest:
            raise ValueError(f"recorded source digest does not match this checkout: {relative_path}")
    if source != _local_program_source(checkout):
        raise ValueError("displayed Triton program differs from this checkout's benchmark function")

    _check_digest(provenance.get("native_extension_sha256"), "native_extension_sha256")
    if recorded_checkout == pathlib.PurePosixPath(checkout.as_posix()):
        native_path = pathlib.Path(recorded_paths["native_extension"]).resolve()
        if not native_path.is_relative_to(checkout) or not native_path.is_file():
            raise ValueError("the recorded local native extension is missing or outside the checkout")
        if _file_sha256(native_path) != provenance["native_extension_sha256"]:
            raise ValueError("recorded native-extension digest does not match the local binary")
    # A remote binary cannot be rehashed here. Its recorded digest identifies the
    # claimed build; it does not authenticate the run or prove a source-to-binary link.


def _validate_configuration(case: dict, method: dict) -> None:
    label = f"case N={case.get('elements')}"
    expected = {"frontend": "classic", "basis_mode": "computed", "basis_storage": "registers"}
    if any(case.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{label} must use ordinary Triton with register-computed bases")
    if type(case.get("basis_global_loads")) is not int or case["basis_global_loads"] != 0:
        raise ValueError(f"{label} must record zero basis global-memory loads")
    if case.get("optimization_observed") is not True:
        raise ValueError(f"{label} does not record an observed optimization")
    for key in ("block_size", "seed"):
        if type(case.get(key)) is not int or case[key] != method[key]:
            raise ValueError(f"{label} has a different {key} from the benchmark configuration")


def _validate_variants(case: dict) -> None:
    label = f"case N={case['elements']}"
    variants = case.get("variants", {})
    if set(variants) != {"baseline", "optimized"}:
        raise ValueError(f"{label} must contain exactly optimization-OFF and optimization-ON variants")
    for name, variant in variants.items():
        if variant.get("correct") is not True:
            raise ValueError(f"{label} {name} failed bit-exact correctness")
        counts = variant.get("instructions", {})
        for metric in (
                "ptx_total",
                "ptx_shuffle",
                "ptx_shuffle_indexed",
                "ptx_lop3",
                "ptx_redux",
                "ptx_barrier",
                "ptx_global_load",
                "ptx_global_store",
                "ptx_shared_load",
                "ptx_shared_store",
                "ttgir_reduce",
                "ttgir_global_load",
                "llir_shuffle",
                "registers",
                "shared_bytes",
        ):
            if type(counts.get(metric)) is not int or counts[metric] < 0:
                raise ValueError(f"{label} {name} is missing a nonnegative instruction/resource count for {metric}")
        if counts["ptx_global_load"] != 1 or counts["ptx_global_store"] != 1:
            raise ValueError(f"{label} {name} must have exactly one index load and output store")
        if counts["ttgir_reduce"] != 32:
            raise ValueError(f"{label} {name} must start from the same 32 source reductions")
    off = variants["baseline"]["instructions"]
    on = variants["optimized"]["instructions"]
    if off["ptx_shuffle_indexed"] != 0 or on["ptx_shuffle_indexed"] != 32:
        raise ValueError(f"{label} does not demonstrate the 0-to-32 indexed-shuffle rewrite")
    if on["ptx_redux"] != 0 or (off["ptx_redux"] == 0 and off["ptx_shuffle"] == 0):
        raise ValueError(f"{label} does not demonstrate warp-reduction removal")
    if on["ptx_lop3"] <= off["ptx_lop3"]:
        raise ValueError(f"{label} does not demonstrate conditional-XOR LOP3 fusion")


def _validated_timing(timing: dict, elements: int, label: str) -> dict:
    samples = timing.get("samples_us", [])
    if not isinstance(samples, list) or len(samples) != TIMING_SAMPLES:
        raise ValueError(f"{label} must record {TIMING_SAMPLES} timing samples")
    if any(
            isinstance(sample, bool) or not isinstance(sample, (int, float)) or not math.isfinite(sample) or sample <= 0
            for sample in samples):
        raise ValueError(f"{label} timing samples must be positive finite numbers")
    ordered = sorted(samples)
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    expected = {
        "median_us": median,
        "mean_us": mean,
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "p20_us": ordered[int(0.20 * (len(ordered) - 1))],
        "p80_us": ordered[int(0.80 * (len(ordered) - 1))],
        "relative_stddev": statistics.pstdev(samples) / mean,
        "throughput_gigaelements_per_second": elements / (median * 1000),
        "effective_bandwidth_gb_per_second": elements * 8 / (median * 1000),
    }
    for metric, value in expected.items():
        _check_number(timing.get(metric), value, f"{label} {metric}")
    return {"samples_us": samples, **expected}


def _cases(report: dict, *, checkout: pathlib.Path | None = None) -> list[dict]:
    if type(report.get("schema_version")) is not int or report["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"only schema-{SCHEMA_VERSION} complete source-compiled benchmark results are supported")
    checkout = pathlib.Path(checkout or pathlib.Path(__file__).resolve().parents[3]).resolve()
    _validate_provenance(report, checkout)
    method = report.get("method", {})
    if (method.get("source_compilation")
            != "same checkout @triton.jit source compiled twice through the real NVIDIA pipeline"):
        raise ValueError("benchmark variants must compile the same checkout JIT source through the real NVIDIA backend")
    if (method.get("pre_pass_ir_identity")
            != "exact byte equality of optimization-OFF and optimization-ON TTIR and TTGIR"):
        raise ValueError("benchmark results must establish identical optimization-OFF and optimization-ON pre-pass IR")
    cache_isolation = method.get("jit_cache_isolation", "")
    if "fresh temporary compiler cache" not in cache_isolation or "BENCHMARK_VARIANT" not in cache_isolation:
        raise ValueError("benchmark results must isolate the two source compilations in a fresh compiler cache")

    block = _positive_int(method.get("block_size"), "block_size")
    if block < 32 or block & (block - 1):
        raise ValueError("block_size must be a power of two and at least 32")
    if type(method.get("seed")) is not int or not -(1 << 31) <= method["seed"] < 1 << 31:
        raise ValueError("seed must be a signed 32-bit integer")
    _positive_int(method.get("timing_repetitions_ms"), "timing_repetitions_ms")
    if type(method.get("timing_samples")) is not int or method["timing_samples"] != TIMING_SAMPLES:
        raise ValueError(f"timing_samples must be {TIMING_SAMPLES}")
    cache_policy = method.get("timing_cache_policy", {})
    if cache_policy.get("reuse_input_output_buffers") is not True or cache_policy.get("flush_l2") is not False:
        raise ValueError("timing must disclose reused input/output buffers without L2 flushing")
    bases_source = method.get("bases_source")
    if bases_source != REGISTER_BASES:
        raise ValueError(f"bases must be computed in registers without a basis buffer, got {bases_source!r}")
    if type(method.get("basis_global_memory_loads")) is not int or method["basis_global_memory_loads"] != 0:
        raise ValueError("benchmark metadata must record zero basis global-memory loads")
    if method.get("effective_bandwidth_bytes_per_element") != 8:
        raise ValueError("effective bandwidth must count one 32-bit input load and one 32-bit output store")

    run = report.get("run", {})
    requested_sizes = run.get("requested_sizes", [])
    if not isinstance(requested_sizes, list) or not requested_sizes:
        raise ValueError("benchmark results must record the requested input sizes")
    for size in requested_sizes:
        _positive_int(size, "requested input size")
        if size > MAX_ELEMENTS:
            raise ValueError("requested input sizes exceed the kernel's signed 32-bit offsets")
    if len(set(requested_sizes)) != len(requested_sizes):
        raise ValueError("requested input sizes must not contain duplicates")
    raw_cases = report.get("cases", [])
    if run.get("complete") is not True or [case.get("elements") for case in raw_cases] != requested_sizes:
        raise ValueError("benchmark results are incomplete or do not match every requested input size exactly once")
    cases = []
    for index, case in enumerate(raw_cases):
        _positive_int(case.get("elements"), "case input size")
        _validate_configuration(case, method)
        _validate_variants(case)
        order = ["baseline", "optimized"] if index % 2 == 0 else ["optimized", "baseline"]
        if case.get("measurement_order") != order:
            raise ValueError("benchmark cases must alternate the recorded OFF/ON measurement order")
        variants = {
            name: {
                **variant,
                "timing":
                _validated_timing(variant.get("timing", {}), case["elements"], f"N={case['elements']} {name}"),
            }
            for name, variant in case["variants"].items()
        }
        speedup = variants["baseline"]["timing"]["median_us"] / variants["optimized"]["timing"]["median_us"]
        _check_number(case.get("speedup"), speedup, f"N={case['elements']} speedup")
        cases.append({**case, "variants": variants, "speedup": speedup})

    boundary = report.get("boundary_correctness", {})
    boundary_cases = boundary.get("cases", [])
    if boundary.get("sizes") != list(BOUNDARY_SIZES):
        raise ValueError("benchmark results are missing the 11 reproducible warp/block masked-tail sizes")
    if boundary.get("total") != len(BOUNDARY_SIZES) or boundary.get("passed") != len(BOUNDARY_SIZES):
        raise ValueError("all 11 masked-tail correctness cases must pass before rendering benchmark results")
    if len(boundary_cases) != len(BOUNDARY_SIZES):
        raise ValueError("benchmark results must contain one recorded case for every masked-tail size")
    for expected_size, case in zip(BOUNDARY_SIZES, boundary_cases):
        if type(case.get("elements")) is not int or case["elements"] != expected_size:
            raise ValueError(f"masked-tail cases must contain expected size {expected_size}")
        _validate_configuration(case, method)
        _validate_variants(case)
        if not (case.get("baseline_correct") is True and case.get("optimized_correct") is True
                and case.get("masked_store_guard_preserved") is True):
            raise ValueError(f"masked-tail case N={expected_size} did not preserve correctness and output guards")
    if boundary.get("guard_elements_per_case") != block or boundary.get("masked_store_guard_int32") != -0x13579BDF:
        raise ValueError("masked-tail correctness must record the benchmark's full-block output sentinel")

    large = [case for case in cases if case["elements"] >= 1_048_576]
    peak = max(cases, key=lambda case: case["speedup"])
    expected_summary = {
        "performance_cases": len(cases),
        "bit_exact_performance_cases": len(cases),
        "optimized_cases": len(cases),
        "geomean_speedup": statistics.geometric_mean(case["speedup"] for case in cases),
        "large_problem_geomean_speedup":
        statistics.geometric_mean(case["speedup"] for case in large) if large else None,
        "large_problem_minimum_elements": 1_048_576,
        "peak_speedup": peak["speedup"],
        "peak_frontend": "classic",
        "peak_basis_mode": "computed",
        "peak_elements": peak["elements"],
    }
    summary = report.get("summary", {})
    for key, expected in expected_summary.items():
        if type(expected) is int:
            if type(summary.get(key)) is not int or summary[key] != expected:
                raise ValueError(f"summary {key} does not match the validated cases")
        elif isinstance(expected, float):
            _check_number(summary.get(key), expected, f"summary {key}")
        elif key not in summary or summary[key] != expected:
            raise ValueError(f"summary {key} does not match the validated cases")
    return sorted(cases, key=lambda case: case["elements"])


def _program_source(report: dict) -> str:
    return textwrap.dedent(report["method"]["program_source"]).strip()


def _validate_program(source: str) -> None:
    if "bases_ptr" in source:
        raise ValueError("the plotted Triton program still accepts a basis-buffer pointer")
    if "tl.load(bases" in source or "tl.load(\n        bases" in source:
        raise ValueError("the plotted Triton program still reads its basis from memory")
    if "bases =" not in source:
        raise ValueError("the plotted Triton program does not show register-computed bases")


def _compact_program(source: str) -> list[str]:
    """Reformat only the function signature so the complete program stays legible."""
    lines = source.splitlines()
    definition = next((index for index, line in enumerate(lines) if line.startswith("def ")), None)
    if definition is None or not lines[definition].endswith("("):
        return lines
    closing = next((index for index in range(definition + 1, len(lines)) if lines[index].strip() == "):"), None)
    if closing is None:
        return lines
    arguments = [line.strip().rstrip(",") for line in lines[definition + 1:closing]]
    runtime = [argument for argument in arguments if "constexpr" not in argument]
    constants = [argument for argument in arguments if "constexpr" in argument]
    if not runtime or not constants:
        return lines
    first_line = lines[definition] + ", ".join(runtime) + ","
    second_line = " " * 4 + ", ".join(constants) + "):"
    return [*lines[:definition], first_line, second_line, *lines[closing + 1:]]


def _style_axes(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#C9D2DC")
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(length=3, color="#94A2AF", labelcolor=COLORS["ink"])


def _rounded_box(axis, x: float, y: float, width: float, height: float, *, facecolor: str, edgecolor: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.016",
            transform=axis.transAxes,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.9,
            clip_on=False,
        ))


def _program_panel(axis, source: str, representative: dict) -> None:
    import matplotlib.pyplot as plt

    axis.set_axis_off()
    axis.text(
        0.015,
        0.99,
        "One Triton program, compiled twice",
        transform=axis.transAxes,
        fontsize=11.5,
        fontweight="bold",
        va="top",
        color=COLORS["ink"],
    )
    axis.text(
        0.985,
        0.98,
        "Only the compiler optimization changes",
        transform=axis.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        color=COLORS["muted"],
    )

    _rounded_box(axis, 0.017, 0.065, 0.705, 0.84, facecolor="#F7F9FB", edgecolor="#DEE5EB")
    _rounded_box(axis, 0.752, 0.065, 0.23, 0.84, facecolor="#FFFFFF", edgecolor="#DEE5EB")

    lines = _compact_program(source)
    y_top = 0.855
    line_height = min(0.045, 0.745 / max(len(lines) - 1, 1))
    font_size = min(8.8, max(6.5, 205 / max(len(lines), 1)))
    for index, line in enumerate(lines):
        y = y_top - index * line_height
        is_basis = "bases =" in line
        is_reduction = "basis = tl.xor_sum" in line
        if is_basis or is_reduction:
            axis.add_patch(
                plt.Rectangle(
                    (0.024, y - line_height * 0.72),
                    0.688,
                    line_height * 1.04,
                    transform=axis.transAxes,
                    facecolor="#E3F1EB" if is_basis else "#E9EFF5",
                    edgecolor="none",
                ))
        axis.text(
            0.03,
            y,
            line,
            transform=axis.transAxes,
            va="top",
            fontfamily="DejaVu Sans Mono",
            fontsize=font_size,
            color="#185745" if is_basis else COLORS["ink"],
            fontweight="bold" if is_basis else "normal",
        )

    base_metrics = representative["variants"]["baseline"]["instructions"]
    optimized_metrics = representative["variants"]["optimized"]["instructions"]
    axis.text(0.772, 0.84, "Bases are calculated", transform=axis.transAxes, fontsize=9.1, fontweight="bold")
    axis.text(0.772, 0.757, "32 values stay in registers", transform=axis.transAxes, fontsize=8.0,
              color=COLORS["muted"])
    axis.text(
        0.772,
        0.691,
        "No basis pointer or HBM basis load",
        transform=axis.transAxes,
        fontsize=7.65,
        color=COLORS["muted"],
    )

    axis.plot([0.77, 0.967], [0.613, 0.613], transform=axis.transAxes, color=COLORS["grid"], linewidth=1)
    axis.text(
        0.772,
        0.551,
        LABELS["baseline"],
        transform=axis.transAxes,
        color=COLORS["baseline"],
        fontsize=8.8,
        fontweight="bold",
    )
    axis.text(
        0.772,
        0.472,
        f"{base_metrics.get('ptx_redux', 0)} warp reductions",
        transform=axis.transAxes,
        fontsize=8.0,
        color=COLORS["muted"],
    )
    axis.text(
        0.772,
        0.361,
        LABELS["optimized"],
        transform=axis.transAxes,
        color=COLORS["optimized"],
        fontsize=8.8,
        fontweight="bold",
    )
    axis.text(
        0.772,
        0.282,
        f"{optimized_metrics.get('ptx_shuffle_indexed', 0)} indexed shuffles",
        transform=axis.transAxes,
        fontsize=8.0,
        color=COLORS["muted"],
    )
    axis.text(0.772, 0.161, "Same source, inputs, and GPU", transform=axis.transAxes, fontsize=7.35,
              color=COLORS["muted"])


def _generated_code_panel(axis, representative: dict) -> None:
    axis.set_title("d   Generated code", loc="left", pad=12)
    axis.set_axis_off()

    _rounded_box(axis, 0.015, 0.66, 0.45, 0.24, facecolor="#FCF5F4", edgecolor="#EBD7D4")
    _rounded_box(axis, 0.515, 0.66, 0.465, 0.24, facecolor="#F0F6FA", edgecolor="#D8E5ED")
    axis.text(
        0.039,
        0.845,
        LABELS["baseline"],
        transform=axis.transAxes,
        color=COLORS["baseline"],
        fontsize=8.6,
        fontweight="bold",
    )
    axis.text(
        0.539,
        0.845,
        LABELS["optimized"],
        transform=axis.transAxes,
        color=COLORS["optimized"],
        fontsize=8.6,
        fontweight="bold",
    )
    axis.text(
        0.039,
        0.755,
        "redux.sync.xor.b32",
        transform=axis.transAxes,
        fontfamily="DejaVu Sans Mono",
        fontsize=7.8,
        color=COLORS["ink"],
    )
    axis.text(
        0.539,
        0.767,
        "shfl.sync.idx.b32",
        transform=axis.transAxes,
        fontfamily="DejaVu Sans Mono",
        fontsize=7.8,
        color=COLORS["ink"],
    )
    axis.text(
        0.539,
        0.692,
        "lop3.b32",
        transform=axis.transAxes,
        fontfamily="DejaVu Sans Mono",
        fontsize=7.8,
        color=COLORS["ink"],
    )

    axis.text(0.025, 0.567, "Static PTX / resource", transform=axis.transAxes, fontsize=8.0, color=COLORS["muted"])
    axis.text(
        0.715,
        0.567,
        "OFF",
        transform=axis.transAxes,
        ha="right",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["baseline"],
    )
    axis.text(
        0.950,
        0.567,
        "ON",
        transform=axis.transAxes,
        ha="right",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["optimized"],
    )
    metrics = (
        ("Total instructions", "ptx_total"),
        ("Warp-wide reductions", "ptx_redux"),
        ("Indexed shuffles", "ptx_shuffle_indexed"),
        ("Fused XOR operations", "ptx_lop3"),
        ("Registers per thread", "registers"),
        ("Global loads (index only)", "ptx_global_load"),
        ("Basis HBM loads", "basis_global_loads"),
    )
    for row, (label, metric) in enumerate(metrics):
        y = 0.463 - row * 0.07
        axis.plot([0.018, 0.978], [y + 0.048, y + 0.048], transform=axis.transAxes, color=COLORS["grid"], linewidth=0.7)
        off = representative["variants"]["baseline"]["instructions"].get(metric, representative.get(metric, 0)) or 0
        on = representative["variants"]["optimized"]["instructions"].get(metric, representative.get(metric, 0)) or 0
        axis.text(0.025, y, label, transform=axis.transAxes, fontsize=7.8, color=COLORS["ink"])
        axis.text(0.715, y, f"{off}", transform=axis.transAxes, ha="right", fontsize=8.0, color=COLORS["baseline"])
        axis.text(0.820, y, "→", transform=axis.transAxes, ha="center", fontsize=8.0, color=COLORS["muted"])
        axis.text(
            0.950,
            y,
            f"{on}",
            transform=axis.transAxes,
            ha="right",
            fontsize=8.0,
            fontweight="bold",
            color=COLORS["optimized"],
        )


def render(report: dict, output: pathlib.Path) -> None:
    cases = _cases(report)
    source = _program_source(report)
    _validate_program(source)
    representative = max(cases, key=lambda case: (case.get("optimization_observed", False), case["elements"]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "axes.labelsize": 9.0,
        "axes.titlesize": 10.7,
        "axes.titleweight": "bold",
        "axes.titlecolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })
    figure = plt.figure(figsize=(12.9, 11.2), layout="constrained")
    grid = figure.add_gridspec(3, 2, height_ratios=(1.32, 1, 1), hspace=0.16, wspace=0.12)
    program_axis = figure.add_subplot(grid[0, :])
    latency_axis = figure.add_subplot(grid[1, 0])
    throughput_axis = figure.add_subplot(grid[1, 1])
    speedup_axis = figure.add_subplot(grid[2, 0])
    code_axis = figure.add_subplot(grid[2, 1])
    _program_panel(program_axis, source, representative)

    sizes = np.array([case["elements"] for case in cases], dtype=float)
    for variant in ("baseline", "optimized"):
        timings = [case["variants"][variant]["timing"] for case in cases]
        medians = np.array([timing["median_us"] for timing in timings])
        latency_axis.plot(
            sizes,
            medians,
            color=COLORS[variant],
            marker="o",
            markersize=3.7,
            linewidth=1.9,
            label=LABELS[variant],
        )
        if all("p20_us" in timing and "p80_us" in timing for timing in timings):
            latency_axis.fill_between(
                sizes,
                [timing["p20_us"] for timing in timings],
                [timing["p80_us"] for timing in timings],
                color=COLORS[variant],
                alpha=0.12,
                linewidth=0,
            )
        throughput_axis.plot(
            sizes,
            [timing["throughput_gigaelements_per_second"] for timing in timings],
            color=COLORS[variant],
            marker="o",
            markersize=3.7,
            linewidth=1.9,
        )

    speedups = np.array([case["speedup"] for case in cases])
    speedup_axis.axhline(1, color="#8E9BA8", linewidth=1.0, linestyle=(0, (3, 3)))
    speedup_axis.fill_between(sizes, 1, speedups, color=COLORS["speedup"], alpha=0.11)
    speedup_axis.plot(sizes, speedups, color=COLORS["speedup"], marker="o", markersize=4.0, linewidth=2)
    peak_index = int(np.argmax(speedups))
    speedup_axis.annotate(
        f"peak {speedups[peak_index]:.2f}×",
        (sizes[peak_index], speedups[peak_index]),
        xytext=(-12, 11),
        textcoords="offset points",
        ha="right",
        fontsize=8.4,
        fontweight="bold",
        color=COLORS["speedup"],
    )

    tick_stride = max(1, math.ceil(len(sizes) / 7))
    ticks = sizes[::tick_stride].tolist()
    if sizes[-1] not in ticks:
        ticks.append(sizes[-1])
    for axis in (latency_axis, throughput_axis, speedup_axis):
        axis.set_xscale("log", base=2)
        axis.set_xticks(ticks)
        axis.xaxis.set_major_formatter(FuncFormatter(_human_elements))
        axis.set_xlabel("Output elements")
        _style_axes(axis)

    latency_axis.set_title("a   Steady-state GPU execution time", loc="left")
    latency_axis.set_yscale("log")
    latency_axis.set_ylabel("Time (μs; lower is better)")
    latency_axis.legend(frameon=False, loc="upper left")

    throughput_axis.set_title("b   Throughput", loc="left")
    throughput_axis.set_ylabel("Billion elements / second")
    throughput_axis.set_ylim(bottom=0)

    speedup_axis.set_title("c   Speedup", loc="left")
    speedup_axis.set_ylabel("Execution time OFF / ON")
    speedup_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}×"))

    _generated_code_panel(code_axis, representative)

    gpu = report.get("environment", {}).get("gpu", "GPU")
    capability = report.get("environment", {}).get("compute_capability")
    gpu_label = f"{gpu} · SM {capability}" if capability else gpu
    geomean = statistics.geometric_mean(case["speedup"] for case in cases)
    figure.suptitle(
        "One-hot XOR reduction → indexed warp shuffle\n"
        f"{gpu_label}   ·   {len(cases)} matched input sizes   ·   "
        f"{geomean:.2f}× geometric-mean speedup",
        x=0.038,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.supxlabel(
        "CUDA Graph replay: reused input/output buffers, no L2 flush; median with 20th–80th percentile shading.\n"
        "Warm-cache timing where the working set fits; larger working sets may still access HBM.",
        fontsize=8.0,
        color=COLORS["muted"],
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor="white")
    if output.suffix.lower() != ".pdf":
        figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    if output.suffix.lower() != ".png":
        figure.savefig(output.with_suffix(".png"), dpi=200, facecolor="white")
    plt.close(figure)


def _representative_cases(cases: list[dict]) -> list[dict]:
    chosen = {cases[0]["elements"], cases[-1]["elements"], max(cases, key=lambda case: case["speedup"])["elements"]}
    for target in (65536, 1048576):
        chosen.add(min(cases, key=lambda case: abs(math.log2(case["elements"]) - math.log2(target)))["elements"])
    return [case for case in cases if case["elements"] in chosen]


def _figure_reference(figure_path: pathlib.Path, report_path: pathlib.Path) -> str:
    png_path = (figure_path if figure_path.suffix.lower() == ".png" else figure_path.with_suffix(".png")).resolve()
    relative = pathlib.Path(os.path.relpath(png_path, report_path.parent.resolve())).as_posix()
    return quote(relative, safe="/")


def markdown_report(report: dict, *, figure_reference: str = "figure.png") -> str:
    cases = _cases(report)
    source = _program_source(report)
    _validate_program(source)
    method = report.get("method", {})
    environment = report.get("environment", {})
    provenance = environment.get("provenance", {})
    peak = max(cases, key=lambda case: case["speedup"])
    representative = max(cases, key=lambda case: (case.get("optimization_observed", False), case["elements"]))
    large_cases = [case for case in cases if case["elements"] >= 1_048_576]
    geomean = statistics.geometric_mean(case["speedup"] for case in cases)
    large_geomean = statistics.geometric_mean(case["speedup"] for case in large_cases) if large_cases else None
    baseline = representative["variants"]["baseline"]["instructions"]
    optimized = representative["variants"]["optimized"]["instructions"]
    peak_off = peak["variants"]["baseline"]["timing"]
    peak_on = peak["variants"]["optimized"]["timing"]
    boundary = [
        case for case in report.get("boundary_correctness", {}).get("cases", [])
        if case.get("frontend") == "classic" and case.get("basis_mode") == "computed"
    ]
    large_sentence = (f" For inputs with at least 1,048,576 elements, the geometric mean is **{large_geomean:.3f}×**."
                      if large_geomean is not None else "")
    lines = [
        "# One Triton program, one compiler optimization",
        "",
        "## Result",
        "",
        f"On **{environment.get('gpu', 'the measured GPU')}**, compiling the same ordinary Triton program with a late one-hot-XOR-reduction optimization disabled and enabled gives a **{geomean:.3f}× geometric-mean speedup** across {len(cases)} input sizes, with a **{peak['speedup']:.3f}× peak**.{large_sentence}",
        "",
        "The 32 basis values are computed from a runtime scalar and `tl.arange`; they remain in registers. There is no basis pointer, basis buffer, or HBM basis read. Both generated kernels contain exactly one global input load and one global output store.",
        "",
        f"![Same Triton program: execution time, throughput, speedup, and generated code]({figure_reference})",
        "",
        "## Exact benchmarked Triton program",
        "",
        "The following single kernel is compiled twice. Its source, inputs, runtime seed, GPU, and launch configuration are unchanged; only the compiler optimization is switched off or on. `BENCHMARK_VARIANT` isolates the compiler-cache entries and does not participate in the kernel computation.",
        "",
        "```python",
        source,
        "```",
        "",
        "The optimization recognizes each one-hot `tl.xor_sum(tl.where(...))` after layout assignment. For a basis distributed one value per warp lane, it replaces the warp-wide XOR reduction with an existing singleton gather and unsplat; NVIDIA lowering emits an indexed warp shuffle. Subsequent conditional XORs are combined into `lop3.b32` when applicable, with an LLVM freeze preserving the original select's poison masking. No new Triton language or IR primitive is introduced.",
        "",
        "## Experimental method",
        "",
        f"- **GPU:** {environment.get('gpu', 'unknown')}; SM {environment.get('compute_capability', '?')}; {environment.get('multiprocessors', '?')} multiprocessors.",
        f"- **Inputs:** {len(cases)} output sizes from {cases[0]['elements']:,} to {cases[-1]['elements']:,} 32-bit elements; block size {representative.get('block_size', '?')}.",
        "- **Compared programs:** ordinary Triton only; one register-computed-basis kernel; optimization OFF versus optimization ON.",
        f"- **Compiler provenance:** the benchmark records that the Python frontend, NVIDIA backend, and native pass were loaded from `{provenance.get('checkout_root', 'the benchmark checkout')}`; both variants compile the same JIT source through that backend.",
        "- **Artifact validation:** source digests are checked against the renderer's checkout; correctness flags, timing statistics, requested-size coverage, and summary metrics must agree. Recorded provenance is not authentication of a benchmark run or proof that a remote native binary was built from those sources. A remote binary's digest cannot be independently rehashed on the rendering machine.",
        f"- **Compiler-cache isolation:** {method.get('jit_cache_isolation', 'unused BENCHMARK_VARIANT constexpr')}; the pre-pass TTIR and TTGIR are checked for exact equality.",
        f"- **Global memory operations:** {baseline.get('ptx_global_load', '?')} → {optimized.get('ptx_global_load', '?')} index loads and {baseline.get('ptx_global_store', '?')} → {optimized.get('ptx_global_store', '?')} output stores; {method.get('basis_global_memory_loads', 0)} basis HBM reads in either variant.",
        "- **Execution time:** steady-state GPU kernel execution time in microseconds. CUDA events measure replay of a graph containing many copies of the kernel; elapsed replay time divided by the number of copies gives per-kernel execution time. Replay reduces CPU launch overhead and repeatedly reuses the same input/output buffers without flushing L2. This is a warm-cache measurement when the working set fits; larger working sets are not assumed to remain entirely in L2. The figure reports the median over ten replays, with the 20th–80th percentiles shaded.",
        f"- **Sampling:** `{method.get('timing_repetitions_ms', '?')} ms` requested repetition budget; `do_bench_cudagraph` chooses the graph length from a runtime estimate, so actual replay duration can differ. Measurement order alternates between successive requested input sizes.",
        "- **Throughput:** output elements divided by median execution time, reported in billions of elements per second.",
        "- **Correctness:** both variants are compared bit-for-bit with a PyTorch implementation of the same runtime-seeded basis arithmetic.",
        "",
        "## Execution time and throughput",
        "",
        "| Output elements | Optimization OFF, μs | Optimization ON, μs | Speedup | Throughput OFF → ON, billion elements/s |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in _representative_cases(cases):
        off = case["variants"]["baseline"]["timing"]
        on = case["variants"]["optimized"]["timing"]
        lines.append(
            f"| {case['elements']:,} | {off['median_us']:.3f} | {on['median_us']:.3f} | {case['speedup']:.3f}× | "
            f"{off['throughput_gigaelements_per_second']:.2f} → {on['throughput_gigaelements_per_second']:.2f} |")
    lines.extend([
        "",
        f"At the peak, **N = {peak['elements']:,}**: **{peak_off['median_us']:.3f} → {peak_on['median_us']:.3f} μs**, or **{peak['speedup']:.3f}×**.",
        "",
        "## Generated code",
        "",
        "| Static PTX / resource | Optimization OFF | Optimization ON |",
        "| --- | ---: | ---: |",
        f"| Total PTX instructions | {baseline.get('ptx_total', 0)} | {optimized.get('ptx_total', 0)} |",
        f"| `redux.sync.xor.b32` | {baseline.get('ptx_redux', 0)} | {optimized.get('ptx_redux', 0)} |",
        f"| `shfl.sync.idx.b32` | {baseline.get('ptx_shuffle_indexed', 0)} | {optimized.get('ptx_shuffle_indexed', 0)} |",
        f"| `lop3.b32` | {baseline.get('ptx_lop3', 0)} | {optimized.get('ptx_lop3', 0)} |",
        f"| Registers per thread | {baseline.get('registers', 0)} | {optimized.get('registers', 0)} |",
        f"| Global index loads | {baseline.get('ptx_global_load', '?')} | {optimized.get('ptx_global_load', '?')} |",
        f"| Global output stores | {baseline.get('ptx_global_store', '?')} | {optimized.get('ptx_global_store', '?')} |",
        f"| Basis HBM loads | {method.get('basis_global_memory_loads', 0)} | {method.get('basis_global_memory_loads', 0)} |",
        "",
        "These are static instruction counts from generated PTX, not dynamic hardware performance-counter measurements.",
    ])
    if boundary:
        passed = sum(bool(case.get("baseline_correct")) and bool(case.get("optimized_correct")) for case in boundary)
        lines.extend([
            "",
            "## Boundary correctness",
            "",
            f"Both optimization settings match the reference for **{passed}/{len(boundary)}** additional masked-tail input sizes; sentinel values beyond each logical output remain unchanged.",
            "",
            f"Checked sizes: {', '.join(str(case['elements']) for case in boundary)}.",
        ])
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=python python python/test/microbenchmark/one_hot_xor_reduction.py \\",
        "  --output benchmarks/one_hot_xor_reduction/results.json \\",
        f"  --sizes {','.join(str(size) for size in report['run']['requested_sizes'])} \\",
        f"  --block {method['block_size']} --seed {method['seed']} \\",
        f"  --repetitions-ms {method.get('timing_repetitions_ms', 100)}",
        "",
        "python python/test/microbenchmark/plot_one_hot_xor_reduction.py \\",
        "  benchmarks/one_hot_xor_reduction/results.json \\",
        "  --output benchmarks/one_hot_xor_reduction/figure.pdf \\",
        "  --report-output benchmarks/one_hot_xor_reduction/REPORT.md",
        "```",
        "",
        "Kernel timing is a microbenchmark, not an end-to-end training-step measurement. Small inputs include per-kernel scheduling and graph-replay overhead; static PTX metrics do not measure dynamic memory traffic.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=pathlib.Path, help="Benchmark JSON report.")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Figure path (.pdf or .png).")
    parser.add_argument("--report-output", type=pathlib.Path, help="Optional Markdown report output path.")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    render(report, args.output)
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        reference = _figure_reference(args.output, args.report_output)
        args.report_output.write_text(markdown_report(report, figure_reference=reference))
        print(f"Saved source-identical benchmark report to {args.report_output}")
    print(f"Saved figure to {args.output} (PDF and PNG companions are also written)")


if __name__ == "__main__":
    main()
