"""Capture and replay portable Triton and Gluon IR golden corpora on a CPU."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import dataclasses
import functools
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator

SCHEMA_VERSION = 1
STRATEGIES = ("legacy", "global")
MAX_SHARD_BYTES = 12_000_000
REPRODUCER_SPLIT = "\n// -----\n"
_FUNCTION = re.compile(r"\btt\.func(?:\s+(?:public|private))?\s+@([^\s(]+)")
_SSA_VALUE = re.compile(r"%[-.$A-Za-z0-9_]+(?:#\d+)?")
_ASSIGNMENT = re.compile(r"^(\s*)(%[-.$A-Za-z0-9_]+(?::\d+)?)\s*=\s*(.*)$")
_PURE_OPERATIONS = {
    "tt.addptr", "tt.broadcast", "tt.expand_dims", "tt.join", "tt.make_range", "tt.reshape", "tt.split", "tt.splat",
    "tt.trans", "ttg.convert_layout"
}


class GoldenCorpusError(RuntimeError):
    """An IR corpus is incomplete, corrupt, or no longer matches its goldens."""


@dataclasses.dataclass(frozen=True)
class GoldenCase:
    case_id: str
    kernel: str
    family: str
    language: str
    arch: int
    shard: str
    member: str

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> GoldenCase:
        required = ("id", "kernel", "family", "language", "arch", "shard", "member")
        missing = [key for key in required if key not in value]
        if missing:
            raise GoldenCorpusError(f"A golden case is missing required fields: {', '.join(missing)}")
        if value["language"] not in {"triton", "gluon"}:
            raise GoldenCorpusError(f"Unsupported golden source language: {value['language']}")
        return cls(value["id"], value["kernel"], value["family"], value["language"], int(value["arch"]), value["shard"],
                   value["member"])


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if all(hasattr(value, field) for field in ("backend", "arch", "warp_size")):
        return {field: _json_safe(getattr(value, field)) for field in ("backend", "arch", "warp_size")}
    return str(value)


def ir_metrics(text: str) -> dict[str, int]:
    return {
        "conversions": len(re.findall(r"\bttg\.convert_layout\b", text)),
        "reductions": len(re.findall(r"\btt\.reduce\b", text)),
        "loads": len(re.findall(r"\btt\.load\b", text)),
        "stores": len(re.findall(r"\btt\.store\b", text)),
    }


def function_names(text: str) -> list[str]:
    return _FUNCTION.findall(text)


def normalize_ttgir(text: str) -> str:
    """Stabilize independent deallocation order without changing real operations."""
    lines = text.splitlines(keepends=True)
    result = []
    position = 0
    while position < len(lines):
        match = re.match(r"^\s*ttg\.local_dealloc\s+(%[^\s:]+)", lines[position])
        if match is None:
            result.append(lines[position])
            position += 1
            continue
        group = []
        operands = []
        while position < len(lines):
            current = re.match(r"^\s*ttg\.local_dealloc\s+(%[^\s:]+)", lines[position])
            if current is None:
                break
            group.append(lines[position])
            operands.append(current.group(1))
            position += 1
        result.extend(sorted(group) if len(operands) == len(set(operands)) else group)
    return "".join(result)


def ttgir_dependency_signature(text: str) -> tuple[str, ...]:
    """Compare complete IR while ignoring only independent, pure SSA scheduling."""
    known_values: dict[str, str] = {}
    pure_operations = []
    signature = []

    def replace_values(value: str) -> str:

        def substitute(match: re.Match[str]) -> str:
            original = match.group(0)
            name, separator, index = original.partition("#")
            replacement = known_values.get(name, name)
            return replacement + (separator + index if separator else "")

        return _SSA_VALUE.sub(substitute, value)

    def flush() -> None:
        if pure_operations:
            signature.extend(sorted(pure_operations))
            pure_operations.clear()

    for line in text.splitlines():
        assignment = _ASSIGNMENT.match(line)
        if assignment is None:
            flush()
            signature.append(replace_values(line))
            continue
        indent, result, expression = assignment.groups()
        rewritten = replace_values(expression)
        operation = expression.split(None, 1)[0]
        if operation.startswith(("arith.", "math.")) or operation in _PURE_OPERATIONS:
            identity = hashlib.sha256(rewritten.encode()).hexdigest()
            known_values[result.split(":", 1)[0]] = "%value_" + identity
            pure_operations.append(indent + rewritten)
        else:
            flush()
            identity = hashlib.sha256(rewritten.encode()).hexdigest()
            known_values[result.split(":", 1)[0]] = "%value_" + identity
            signature.append(indent + "%effect_" + identity + " = " + rewritten)
    flush()
    return tuple(signature)


def _backend(target: dict[str, Any]):
    from triton.backends.compiler import GPUTarget

    if target.get("backend") != "cuda":
        raise GoldenCorpusError(f"Unsupported golden backend: {target.get('backend')}")

    from triton.backends.nvidia.compiler import CUDABackend

    return CUDABackend(GPUTarget("cuda", int(target["arch"]), int(target["warp_size"])))


@contextlib.contextmanager
def _parsed_module(text: str, backend: Any, language: str) -> Iterator[Any]:
    from triton._C.libtriton import ir

    suffix = ".ttir" if language == "triton" else ".source"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=suffix) as source:
        source.write(text)
        source.flush()
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        module = ir.parse_mlir_module(source.name, context)
        module.context = context
        yield module


def normalize_source(text: str, target: dict[str, Any], language: str) -> str:
    with _parsed_module(text, _backend(target), language) as module:
        return module.str_nodebug()


def _options(backend: Any, metadata: dict[str, Any], launch: dict[str, Any], strategy: str) -> Any:
    from triton.backends.nvidia.compiler import CUDAOptions

    if strategy not in STRATEGIES:
        raise GoldenCorpusError(f"Unsupported layout strategy: {strategy}")
    ignored = {"extern_libs", "ptx_options", "ir_override"}
    options = {
        key: metadata[key]
        for key in CUDAOptions.__dataclass_fields__
        if key in metadata and metadata[key] is not None and key not in ignored
    }
    for key in ("num_warps", "num_ctas", "num_stages"):
        if key in launch:
            options[key] = int(launch[key])
    options["optimize_layouts"] = strategy == "global"
    try:
        return backend.parse_options(options)
    except (AssertionError, TypeError, ValueError, FileNotFoundError) as error:
        raise GoldenCorpusError(f"Cannot reconstruct production compiler options: {error}") from error


def compile_case(payload: dict[str, Any], strategy: str, *, normalize: bool = True) -> tuple[str, dict[str, int]]:
    target = payload.get("target")
    metadata = payload.get("metadata")
    launch = payload.get("launch")
    language = payload.get("language")
    source = payload.get("source")
    if not isinstance(target, dict) or not isinstance(metadata, dict) or not isinstance(launch, dict):
        raise GoldenCorpusError("A golden case has incomplete compiler metadata")
    if not isinstance(source, str) or language not in {"triton", "gluon"}:
        raise GoldenCorpusError("A golden case has incomplete source IR")
    backend = _backend(target)
    options = _options(backend, metadata, launch, strategy)
    stage_metadata: dict[str, Any] = {}
    with _parsed_module(source, backend, language) as module:
        try:
            if language == "triton":
                output = backend.make_ttgir(module, stage_metadata, options, int(target["arch"]))
            else:
                output = backend.gluon_to_ttgir(module, stage_metadata, options, int(target["arch"]))
        except Exception as error:
            raise GoldenCorpusError(f"The {strategy} {language} TTGIR pipeline failed: {error}") from error
        output_text = output.str_nodebug()
    if "tensordesc_meta" not in stage_metadata:
        raise GoldenCorpusError("The TTGIR pipeline did not finish")
    result = normalize_ttgir(output_text) if normalize else output_text
    return result, ir_metrics(result)


def triton_opt_reproducer(payload: dict[str, Any], strategy: str) -> tuple[str, str]:
    """Capture the actual compiler pass pipeline as a standalone MLIR reproducer."""
    previous = os.environ.get("TRITON_REPRODUCER_PATH")
    with tempfile.TemporaryDirectory(prefix="triton-ir-golden-") as directory:
        prefix = Path(directory) / "golden"
        os.environ["TRITON_REPRODUCER_PATH"] = str(prefix)
        try:
            output, _ = compile_case(payload, strategy, normalize=False)
        finally:
            if previous is None:
                os.environ.pop("TRITON_REPRODUCER_PATH", None)
            else:
                os.environ["TRITON_REPRODUCER_PATH"] = previous
        stage = "make_ttgir" if payload["language"] == "triton" else "gluon_to_ttgir"
        path = prefix.with_name(f"{prefix.name}.{stage}.repro.mlir")
        try:
            reproducer = path.read_text()
        except OSError as error:
            raise GoldenCorpusError(f"The compiler did not emit its {stage} pass pipeline") from error
    if "mlir_reproducer" not in reproducer or not re.search(r'\bpipeline:\s*"', reproducer):
        raise GoldenCorpusError("The compiler reproducer has no executable pass pipeline")
    reference = payload.get("goldens", {}).get(strategy)
    if isinstance(reference, dict):
        expected = reference.get("ttgir")
        normalized = normalize_ttgir(output)
        if not isinstance(expected, str) or _sha256(expected.encode()) != reference.get("sha256"):
            raise GoldenCorpusError(f"The frozen {strategy} TTGIR reference is corrupt")
        if normalized != expected and ttgir_dependency_signature(normalized) != ttgir_dependency_signature(expected):
            raise GoldenCorpusError(f"The {strategy} reproducer no longer matches its frozen TTGIR golden")
    return reproducer, output


def write_reproducer_shard(path: Path, reproductions: list[dict[str, Any]]) -> dict[str, Any]:
    """Write directly executable, split-input MLIR and its frozen output digest."""
    if not reproductions:
        raise GoldenCorpusError("A triton-opt reproducer shard cannot be empty")
    chunks = []
    outputs = []
    for reproduction in reproductions:
        for key in ("id", "strategy", "language", "source", "expected"):
            if not isinstance(reproduction.get(key), str):
                raise GoldenCorpusError(f"A triton-opt reproducer is missing {key}")
        if reproduction["strategy"] not in STRATEGIES:
            raise GoldenCorpusError(f"Unsupported layout strategy: {reproduction['strategy']}")
        header = {key: reproduction[key] for key in ("id", "strategy", "language")}
        chunks.append("// IR-GOLDEN: " + json.dumps(header, sort_keys=True) + "\n" +
                      reproduction["source"].rstrip("\n"))
        outputs.append(normalize_ttgir(reproduction["expected"]))
    data = (REPRODUCER_SPLIT.join(chunks) + "\n").encode()
    stored = _compress(data) if path.name.endswith(".mlir.zst") else data
    if not (path.name.endswith(".mlir") or path.name.endswith(".mlir.zst")):
        raise GoldenCorpusError("Triton-opt reproducer shards must end in .mlir or .mlir.zst")
    if len(stored) >= MAX_SHARD_BYTES:
        raise GoldenCorpusError(f"Reproducer shard {path.name} exceeds the {MAX_SHARD_BYTES:,}-byte limit")
    expected_text = REPRODUCER_SPLIT.join(outputs) + "\n"
    expected = ("\n".join(ttgir_dependency_signature(expected_text)) + "\n").encode()
    expected_sha256 = _sha256(expected)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stored)
    shard_id = path.name.removesuffix(".mlir.zst").removesuffix(".mlir")
    checksum_path = path.with_name(shard_id + ".sha256")
    checksum_path.write_text(f"{expected_sha256}  -\n")
    return {
        "id": shard_id, "path": path.name, "sha256": _sha256(stored), "bytes": len(stored), "replays":
        len(reproductions), "expected_sha256": expected_sha256, "checksum": checksum_path.name
    }


def replay_reproducer_shard(path: str | Path, triton_opt: str = "triton-opt", *, expected_sha256: str | None = None,
                            compressed_sha256: str | None = None) -> dict[str, Any]:
    """Run all embedded pipelines directly in one triton-opt process."""
    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as error:
        raise GoldenCorpusError(f"Triton-opt reproducer shard {source} is missing") from error
    if compressed_sha256 is not None and _sha256(data) != compressed_sha256:
        raise GoldenCorpusError(f"Triton-opt reproducer shard {source.name} failed its SHA-256 check")
    try:
        result = subprocess.run([
            triton_opt, "--mlir-disable-threading", "--mlir-diagnostic-verbosity-level=errors", "--run-reproducer",
            "--split-input-file", "-"
        ], input=_decompress(data), capture_output=True, check=False)
    except OSError as error:
        raise GoldenCorpusError(f"Cannot execute triton-opt: {error}") from error
    if result.returncode:
        raise GoldenCorpusError(f"Triton-opt replay failed for {source.name}: "
                                f"{result.stderr.decode(errors='replace').strip()}")
    normalized = normalize_ttgir(result.stdout.decode())
    actual_sha256 = _sha256(("\n".join(ttgir_dependency_signature(normalized)) + "\n").encode())
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise GoldenCorpusError(f"Triton-opt golden output changed for {source.name}: "
                                f"expected {expected_sha256}, observed {actual_sha256}")
    return {"path": source.name, "sha256": actual_sha256}


def freeze_payload(payload: dict[str, Any]) -> dict[str, Any]:
    frozen = dict(payload)
    references = {}
    for strategy in STRATEGIES:
        output, metrics = compile_case(payload, strategy)
        references[strategy] = {"ttgir": output, "sha256": _sha256(output.encode()), "metrics": metrics}
    frozen["goldens"] = references
    return frozen


def verify_payload(payload: dict[str, Any], strategies: tuple[str, ...] = STRATEGIES) -> dict[str, Any]:
    references = payload.get("goldens")
    if not isinstance(references, dict):
        raise GoldenCorpusError("A golden case has no frozen TTGIR references")
    result = {}
    for strategy in strategies:
        reference = references.get(strategy)
        if not isinstance(reference, dict) or not isinstance(reference.get("ttgir"), str):
            raise GoldenCorpusError(f"A golden case has no {strategy} TTGIR reference")
        expected = reference["ttgir"]
        if _sha256(expected.encode()) != reference.get("sha256"):
            raise GoldenCorpusError(f"The frozen {strategy} TTGIR reference is corrupt")
        actual, metrics = compile_case(payload, strategy)
        if actual != expected and ttgir_dependency_signature(actual) != ttgir_dependency_signature(expected):
            raise GoldenCorpusError(f"The {strategy} TTGIR golden changed: expected {reference['sha256']}, "
                                    f"observed {_sha256(actual.encode())}")
        if metrics != reference.get("metrics"):
            raise GoldenCorpusError(f"The {strategy} TTGIR metrics changed")
        result[strategy] = metrics
    if set(strategies) == set(STRATEGIES):
        global_metrics = result["global"]
        legacy_metrics = result["legacy"]
        frozen_global = references["global"]["metrics"]
        frozen_legacy = references["legacy"]["metrics"]
        for metric in ("conversions", "reductions"):
            if global_metrics[metric] - legacy_metrics[metric] > frozen_global[metric] - frozen_legacy[metric]:
                raise GoldenCorpusError(f"The global layout optimization regressed {metric} against legacy")
    return result


def _compress(data: bytes) -> bytes:
    try:
        import zstandard
    except ImportError:
        command = shutil.which("zstd")
        if command is None:
            raise GoldenCorpusError("Reading production shards requires the zstandard package or zstd command")
        process = subprocess.run([command, "-q", "-3", "-c"], input=data, capture_output=True, check=False)
        if process.returncode:
            raise GoldenCorpusError("The zstd command could not compress a golden shard")
        return process.stdout
    return zstandard.ZstdCompressor(level=3).compress(data)


def _decompress(data: bytes) -> bytes:
    if data.startswith(b"\x28\xb5\x2f\xfd"):
        try:
            import zstandard
        except ImportError:
            command = shutil.which("zstd")
            if command is None:
                raise GoldenCorpusError("Reading production shards requires the zstandard package or zstd command")
            process = subprocess.run([command, "-q", "-d", "-c"], input=data, capture_output=True, check=False)
            if process.returncode:
                raise GoldenCorpusError("A golden shard could not be decompressed")
            return process.stdout
        try:
            return zstandard.ZstdDecompressor().decompress(data)
        except zstandard.ZstdError as error:
            raise GoldenCorpusError(f"A golden shard could not be decompressed: {error}") from error
    return data


def write_shard(path: Path, payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for member, payload in sorted(payloads.items()):
            content = _json_bytes(payload)
            info = tarfile.TarInfo(member)
            info.size = len(content)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
    compressed = _compress(buffer.getvalue())
    if len(compressed) >= MAX_SHARD_BYTES:
        raise GoldenCorpusError(f"Golden shard {path.name} exceeds the {MAX_SHARD_BYTES:,}-byte limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return {
        "id": path.stem.removesuffix(".tar"), "path": path.name, "sha256": _sha256(compressed), "bytes":
        len(compressed), "cases": len(payloads)
    }


class GoldenCorpus:

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise GoldenCorpusError(f"No golden corpus manifest exists at {manifest_path}")
        try:
            self.manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise GoldenCorpusError(f"The golden corpus manifest cannot be read: {error}") from error
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise GoldenCorpusError(f"Unsupported golden corpus schema: {self.manifest.get('schema_version')}")
        shards = self.manifest.get("shards")
        cases = self.manifest.get("cases")
        if not isinstance(shards, list) or not isinstance(cases, list):
            raise GoldenCorpusError("The golden corpus manifest has no case or shard index")
        self.shards = {entry["id"]: entry for entry in shards}
        self.cases = sorted((GoldenCase.from_json(entry) for entry in cases), key=lambda case:
                            (case.shard, case.case_id))
        if len(self.shards) != len(shards):
            raise GoldenCorpusError("The golden corpus contains duplicate shard identifiers")
        seen = set()
        for case in self.cases:
            if case.case_id in seen:
                raise GoldenCorpusError(f"The golden corpus contains duplicate case {case.case_id}")
            if case.shard not in self.shards:
                raise GoldenCorpusError(f"Golden case {case.case_id} references an unknown shard")
            seen.add(case.case_id)

    def select(self, *, family: str | None = None, kernel: str | None = None, language: str | None = None,
               arch: int | None = None) -> list[GoldenCase]:
        return [
            case for case in self.cases
            if (family is None or case.family == family) and (kernel is None or case.kernel == kernel) and (
                language is None or case.language == language) and (arch is None or case.arch == arch)
        ]

    @functools.lru_cache(maxsize=8)
    def _read_shard(self, shard_id: str) -> dict[str, dict[str, Any]]:
        entry = self.shards[shard_id]
        path = (self.root / entry["path"]).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise GoldenCorpusError(f"Golden shard {shard_id} is missing or escapes the corpus")
        data = path.read_bytes()
        if len(data) != entry.get("bytes") or _sha256(data) != entry.get("sha256"):
            raise GoldenCorpusError(f"Golden shard {shard_id} failed its size or SHA-256 check")
        result = {}
        try:
            with tarfile.open(fileobj=io.BytesIO(_decompress(data)), mode="r:") as archive:
                for member in archive:
                    if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                        raise GoldenCorpusError(f"Golden shard {shard_id} has an unsafe archive member")
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise GoldenCorpusError(f"Golden shard {shard_id} has an unreadable member")
                    if member.name in result:
                        raise GoldenCorpusError(f"Golden shard {shard_id} has duplicate archive members")
                    result[member.name] = json.loads(handle.read())
        except (tarfile.TarError, json.JSONDecodeError) as error:
            raise GoldenCorpusError(f"Golden shard {shard_id} is not a valid case archive: {error}") from error
        if len(result) != entry.get("cases"):
            raise GoldenCorpusError(f"Golden shard {shard_id} has the wrong number of cases")
        return result

    def payload(self, case: GoldenCase) -> dict[str, Any]:
        try:
            payload = self._read_shard(case.shard)[case.member]
        except KeyError as error:
            raise GoldenCorpusError(f"Golden case {case.case_id} is missing from its archive") from error
        if payload.get("id") != case.case_id or payload.get("language") != case.language:
            raise GoldenCorpusError(f"Golden case {case.case_id} has inconsistent archive metadata")
        return payload

    def refresh(self, selected: list[GoldenCase]) -> int:
        selected_ids = {case.case_id for case in selected}
        changed = 0
        for shard_id in sorted({case.shard for case in selected}):
            original = self._read_shard(shard_id)
            updated = {}
            for member, payload in original.items():
                if payload["id"] in selected_ids:
                    payload = freeze_payload(payload)
                    changed += 1
                updated[member] = payload
            entry = self.shards[shard_id]
            replacement = write_shard(self.root / entry["path"], updated)
            entry.update(replacement)
        self._read_shard.cache_clear()
        (self.root / "manifest.json").write_bytes(_json_bytes(self.manifest))
        return changed


def compilation_capture_listener(directory: str | Path,
                                 previous: Callable[..., Any] | None = None) -> Callable[..., None]:
    destination = Path(directory)

    def capture(*, src: Any, metadata: dict[str, Any], metadata_group: dict[str, Any], times: Any,
                cache_hit: bool) -> None:
        safe_metadata = _json_safe(metadata)
        cache_key = safe_metadata.get("hash")
        if not isinstance(cache_key, str) or not cache_key:
            raise GoldenCorpusError("A compilation event has no stable cache hash")
        entry = destination / cache_key
        entry.mkdir(parents=True, exist_ok=True)
        copied = []
        for name, source in sorted(metadata_group.items()):
            if Path(name).suffix not in {".source", ".ttir", ".ttgir", ".json"}:
                continue
            source_path = Path(source)
            if source_path.is_file():
                shutil.copyfile(source_path, entry / Path(name).name)
                copied.append(Path(name).name)
        event = {
            "schema_version": SCHEMA_VERSION, "cache_hit": cache_hit, "metadata": safe_metadata, "language":
            str(getattr(src, "language", "unknown")), "name": str(getattr(src, "name", "")), "artifacts": copied,
            "times": _json_safe(times)
        }
        (entry / "capture.json").write_bytes(_json_bytes(event))
        if previous is not None:
            previous(src=src, metadata=metadata, metadata_group=metadata_group, times=times, cache_hit=cache_hit)

    return capture


@contextlib.contextmanager
def capture_compilations(directory: str | Path) -> Iterator[Callable[..., None]]:
    from triton import knobs

    previous = knobs.compilation.listener
    listener = compilation_capture_listener(directory, previous)
    knobs.compilation.listener = listener
    try:
        yield listener
    finally:
        knobs.compilation.listener = previous


@functools.lru_cache(maxsize=4)
def _worker_corpus(root: str) -> GoldenCorpus:
    return GoldenCorpus(root)


def _verify_worker(arguments: tuple[str, dict[str, Any], tuple[str, ...]]) -> dict[str, Any]:
    root, descriptor, strategies = arguments
    corpus = _worker_corpus(root)
    case = GoldenCase.from_json(descriptor)
    metrics = verify_payload(corpus.payload(case), strategies)
    return {
        "id": case.case_id, "kernel": case.kernel, "family": case.family, "language": case.language, "metrics": metrics
    }


def _reproducer_worker(arguments: tuple[str, dict[str, Any], str]) -> dict[str, Any]:
    root, descriptor, strategy = arguments
    corpus = _worker_corpus(root)
    case = GoldenCase.from_json(descriptor)
    source, expected = triton_opt_reproducer(corpus.payload(case), strategy)
    return {
        "id": case.case_id, "strategy": strategy, "language": case.language, "family": case.family, "kernel":
        case.kernel, "arch": case.arch, "source": source, "expected": expected
    }


def _replay_reproducer_worker(arguments: tuple[str, dict[str, Any], str]) -> dict[str, Any]:
    root, descriptor, executable = arguments
    return replay_reproducer_shard(
        Path(root) / descriptor["path"], executable, expected_sha256=descriptor["expected_sha256"],
        compressed_sha256=descriptor["sha256"])


def _descriptor(case: GoldenCase) -> dict[str, Any]:
    return {
        "id": case.case_id, "kernel": case.kernel, "family": case.family, "language": case.language, "arch": case.arch,
        "shard": case.shard, "member": case.member
    }


def export_triton_opt_reproducers(corpus: GoldenCorpus, selected: list[GoldenCase], destination: str | Path, *,
                                  strategies: tuple[str, ...] = STRATEGIES, workers: int = 1,
                                  max_uncompressed_bytes: int = 8_000_000, compress: bool = False) -> dict[str, Any]:
    """Freeze executable pass pipelines without exposing production IR in Triton."""
    root = Path(destination)
    if root.exists() and not root.is_dir():
        raise GoldenCorpusError(f"The triton-opt destination is not a directory: {root}")
    if (root / "manifest.json").exists():
        raise GoldenCorpusError(f"The triton-opt destination already contains a frozen manifest: {root}")
    if workers < 1 or max_uncompressed_bytes < 1:
        raise GoldenCorpusError("Triton-opt export requires positive worker and shard limits")
    for strategy in strategies:
        if strategy not in STRATEGIES:
            raise GoldenCorpusError(f"Unsupported layout strategy: {strategy}")
    root.mkdir(parents=True, exist_ok=True)
    pending = []
    pending_bytes = 0
    descriptors = []
    completed = set()

    def flush(reproductions: list[dict[str, Any]]) -> None:
        if not reproductions:
            return
        suffix = ".mlir.zst" if compress else ".mlir"
        path = root / f"replay-{len(descriptors):05d}{suffix}"
        try:
            descriptor = write_reproducer_shard(path, reproductions)
        except GoldenCorpusError as error:
            if "exceeds" not in str(error) or len(reproductions) < 2:
                raise
            midpoint = len(reproductions) // 2
            flush(reproductions[:midpoint])
            flush(reproductions[midpoint:])
            return
        descriptor["languages"] = dict(collections.Counter(item["language"] for item in reproductions))
        descriptor["strategies"] = dict(collections.Counter(item["strategy"] for item in reproductions))
        descriptor["architectures"] = sorted({item["arch"] for item in reproductions})
        descriptors.append(descriptor)

    jobs = ((str(corpus.root), _descriptor(case), strategy) for case in selected for strategy in strategies)
    if workers == 1:
        results = map(_reproducer_worker, jobs)
        executor = contextlib.nullcontext()
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        results = None
    with executor as pool:
        if pool is not None:
            results = pool.map(_reproducer_worker, jobs, chunksize=4)
        assert results is not None
        for reproduction in results:
            identity = (reproduction["id"], reproduction["strategy"])
            if identity in completed:
                raise GoldenCorpusError(f"Duplicate triton-opt replay case: {identity}")
            completed.add(identity)
            size = len(reproduction["source"].encode())
            if pending and pending_bytes + size > max_uncompressed_bytes:
                flush(pending)
                pending = []
                pending_bytes = 0
            pending.append(reproduction)
            pending_bytes += size
        flush(pending)
    expected_replays = len(selected) * len(strategies)
    if len(completed) != expected_replays:
        raise GoldenCorpusError("The triton-opt export did not cover every selected strategy and case")
    source_manifest = (corpus.root / "manifest.json").read_bytes()
    manifest = {
        "schema_version": SCHEMA_VERSION, "source_manifest_sha256": _sha256(source_manifest), "cases": len(selected),
        "replays": expected_replays, "strategies": list(strategies), "shards": descriptors
    }
    manifest_bytes = _json_bytes(manifest)
    if len(manifest_bytes) >= MAX_SHARD_BYTES:
        raise GoldenCorpusError("The triton-opt manifest exceeds the hosted repository file limit")
    (root / "manifest.json").write_bytes(manifest_bytes)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("replay", "refresh", "inventory", "export-opt", "replay-opt"))
    parser.add_argument("--corpus", type=Path, default=Path(value) if
                        (value := os.environ.get("TRITON_PRODUCTION_IR_CORPUS")) else None)
    parser.add_argument("--family")
    parser.add_argument("--kernel")
    parser.add_argument("--language", choices=("triton", "gluon"))
    parser.add_argument("--arch", type=int)
    parser.add_argument("--strategy", choices=(*STRATEGIES, "both"), default="both")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, help="directory for executable triton-opt reproducer shards")
    parser.add_argument("--triton-opt", default="triton-opt", help="path to the standalone triton-opt executable")
    parser.add_argument("--compress-opt", action="store_true", help="store exported MLIR as Zstandard streams")
    parser.add_argument("--accept", action="store_true", help="explicitly permit rewriting frozen references")
    args = parser.parse_args(argv)
    if args.corpus is None:
        parser.error("--corpus or TRITON_PRODUCTION_IR_CORPUS is required")
    if args.workers < 1:
        parser.error("--workers must be positive")
    corpus = GoldenCorpus(args.corpus)
    selected = corpus.select(family=args.family, kernel=args.kernel, language=args.language, arch=args.arch)
    if not selected:
        raise GoldenCorpusError("No golden cases match the requested filters")
    if args.command == "inventory":
        report = {
            "cases": len(selected), "families": dict(collections.Counter(case.family for case in selected)),
            "languages": dict(collections.Counter(case.language for case in selected))
        }
    elif args.command == "refresh":
        if not args.accept:
            parser.error("refresh requires --accept")
        report = {"refreshed": corpus.refresh(selected)}
    elif args.command == "export-opt":
        if args.output is None:
            parser.error("export-opt requires --output")
        strategies = STRATEGIES if args.strategy == "both" else (args.strategy, )
        manifest = export_triton_opt_reproducers(corpus, selected, args.output, strategies=strategies,
                                                 workers=args.workers, compress=args.compress_opt)
        report = {"cases": manifest["cases"], "replays": manifest["replays"], "shards": len(manifest["shards"])}
    elif args.command == "replay-opt":
        if args.output is None:
            parser.error("replay-opt requires --output")
        try:
            manifest = json.loads((args.output / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise GoldenCorpusError(f"The triton-opt reproducer manifest cannot be read: {error}") from error
        if manifest.get("schema_version") != SCHEMA_VERSION or not isinstance(manifest.get("shards"), list):
            raise GoldenCorpusError("The triton-opt reproducer manifest is invalid")
        current_manifest_sha256 = _sha256((corpus.root / "manifest.json").read_bytes())
        if manifest.get("source_manifest_sha256") != current_manifest_sha256:
            raise GoldenCorpusError("The triton-opt reproducers were generated from a different golden corpus")
        jobs = ((str(args.output), descriptor, args.triton_opt) for descriptor in manifest["shards"])
        if args.workers == 1:
            results = [_replay_reproducer_worker(job) for job in jobs]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                results = list(executor.map(_replay_reproducer_worker, jobs, chunksize=1))
        report = {
            "cases": manifest["cases"], "replays": manifest["replays"], "shards": len(results), "workers": args.workers
        }
    else:
        strategies = STRATEGIES if args.strategy == "both" else (args.strategy, )
        jobs = [(str(corpus.root), _descriptor(case), strategies) for case in selected]
        if args.workers == 1:
            results = [_verify_worker(job) for job in jobs]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                results = list(executor.map(_verify_worker, jobs, chunksize=max(1, len(jobs) // (args.workers * 8))))
        report = {
            "cases": len(results), "strategies": list(strategies), "workers": args.workers, "languages":
            dict(collections.Counter(item["language"] for item in results))
        }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
