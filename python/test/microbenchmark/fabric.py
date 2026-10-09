"""Compare PR #12176, plain Gluon, and Gluon with thread availability.

Compile each variant under its own PYTHONPATH and TRITON_CACHE_DIR:
  python fabric.py --compile pr --output /tmp/fabric-stack-comparison
  python fabric.py --compile plain --output /tmp/fabric-stack-comparison
  python fabric.py --compile optimized --output /tmp/fabric-stack-comparison
Then run --measure under the optimized checkout. All three saved cubins execute
in one process on identical allocations with balanced execution order.

The cases cover ready/pending/aborted operations, single and repeated calls,
and warp-specialized workers. They measure protocol overhead, not transport
throughput. No TW runtime is needed; the queue and counters are GPU fixtures.
"""

import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import triton
from triton._C import libtriton
from triton.compiler.compiler import CompiledKernel
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import fabric
from triton.experimental.gluon.nvidia.fabric import Protocol, RequestLayout, _State, _Queue, _SynchronizedBufferArgs


@gluon.jit
def protocol_body(Argument, Out, iterations, KIND: gl.constexpr, BLOCKING: gl.constexpr, WIDTH: gl.constexpr,
                  NATIVE: gl.constexpr):
    if NATIVE:
        buffer = fabric.bind(Argument)
    else:
        buffer = Argument
    if KIND == "submit" or KIND == "arrive":
        gl.atomic_store(buffer._queue.tail, 0, sem="relaxed", scope="gpu")
        gl.atomic_store(buffer._queue.cached_head, 0, sem="relaxed", scope="gpu")
        gl.atomic_store(buffer._state.send_count, 0, sem="relaxed", scope="gpu")
    total = 0
    for i in range(iterations):
        if KIND == "recv":
            ready = fabric.barrier.wait(buffer.recv_barrier, blocking=BLOCKING, consume=False)
        elif KIND == "ack":
            ready = fabric.barrier.wait(buffer.ack_barrier, blocking=BLOCKING, consume=False)
        elif KIND == "send":
            ready = buffer.store_wait(blocking=BLOCKING)
        elif KIND == "abort":
            ready = not buffer.is_aborted()
        elif KIND == "submit":
            ready = buffer.async_store(buffer.ptr + (i & 63), 1)
        else:
            ready = fabric.barrier.arrive(buffer.peer.ack_barrier)
        total += ready.to(gl.int32)
    if WIDTH == 1:
        gl.store(Out, total)
    else:
        offsets = gl.arange(0, WIDTH, layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0]))
        gl.store(Out + offsets, total)


@gluon.jit
def idle():
    pass


@gluon.jit
def comparison_kernel(Argument, Out, iterations, KIND: gl.constexpr, BLOCKING: gl.constexpr, WIDTH: gl.constexpr,
                      WORKER_WARPS: gl.constexpr, NATIVE: gl.constexpr):
    if WORKER_WARPS:
        gl.warp_specialize([(idle, ()), (protocol_body, (Argument, Out, iterations, KIND, BLOCKING, WIDTH, NATIVE))],
                           [WORKER_WARPS])
    else:
        protocol_body(Argument, Out, iterations, KIND, BLOCKING, WIDTH, NATIVE)


def cases():
    kinds = ("recv", "ack", "send", "abort", "submit", "arrive")
    scenarios = [
        ("repeat_cta", kinds, 4, 0, 1, 4096, False, "ready"),
        ("repeat_warp", kinds, 1, 0, 1, 4096, False, "ready"),
        ("single_cta", kinds, 4, 0, 128, 1, False, "ready"),
        ("worker_one", kinds, 4, 1, 128, 1024, False, "ready"),
        ("worker_four", kinds, 4, 4, 128, 1024, False, "ready"),
        ("blocking_ready", kinds[:3], 4, 0, 1, 4096, True, "ready"),
        ("pending", kinds[:3], 4, 0, 1, 4096, False, "pending"),
        ("aborted", kinds, 4, 0, 1, 4096, False, "aborted"),
    ]
    return [
        dict(name=f"{scenario}-{kind}", scenario=scenario, kind=kind, num_warps=warps, worker_warps=worker, width=width,
             iterations=iterations, blocking=blocking, state=state)
        for scenario, operations, warps, worker, width, iterations, blocking, state in scenarios
        for kind in operations
    ]


def inputs():
    layout = RequestLayout(stride=56, ready_offset=48, type_offset=0, handle_offset=8, src_offset=16, dst_offset=24,
                           length_offset=32, bypass_ar_offset=40)
    protocol = Protocol(layout, 1, 6, 0, 1, 1 << 63, 33)
    state = _State(*(torch.zeros(1, dtype=torch.int64, device="cuda") for _ in _State._fields))
    queue = _Queue(*(torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(3)),
                   torch.zeros(8192 * layout.stride, dtype=torch.int8, device="cuda"))
    buffer = _SynchronizedBufferArgs(torch.arange(64, dtype=torch.float32, device="cuda"), state, queue, 7,
                                     gl.constexpr(8192), gl.constexpr(protocol))
    return buffer, torch.empty(128, dtype=torch.int32, device="cuda")


def prepare(buffer, out, case):
    values = dict(recv_lock=1, ack_lock=1, sends_completed=-1, aborted=32)
    if case["state"] == "pending":
        values.update(recv_lock=0, ack_lock=0, send_count=2, sends_completed=0)
    if case["state"] == "aborted":
        values["aborted"] = 33
    for name, tensor in buffer.state._asdict().items():
        tensor.fill_(values.get(name, 0))
    for tensor in buffer.queue:
        tensor.zero_()
    out.fill_(-1)


def arguments(buffer, out, case, variant):
    return (buffer, out, case["iterations"], case["kind"], case["blocking"], case["width"], case["worker_warps"],
            variant != "pr")


def validate(buffer, out, case):
    expected = case["iterations"] if case["state"] == "ready" else 0
    assert out[:case["width"]].tolist() == [expected] * case["width"], case
    if case["kind"] in ("submit", "arrive"):
        assert buffer.queue.tail.item() == expected, case
        assert buffer.state.send_count.item() == (expected if case["kind"] == "submit" else 0), case
        # Check publication and the data-dependent source offset on the last slot.
        if expected:
            raw = buffer.queue.buffer.cpu().numpy().tobytes()
            import struct
            pos = (expected - 1) * 56
            assert struct.unpack_from("<Q", raw, pos + 48)[0] == 1 << 63
            assert struct.unpack_from("<I", raw, pos)[0] == (1 if case["kind"] == "submit" else 6)
            assert struct.unpack_from("<I", raw, pos + 8)[0] == 7
            if case["kind"] == "submit":
                assert struct.unpack_from("<QQQ", raw, pos + 16) == (4 * ((expected - 1) & 63), 0, 4)


def file_hash(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def compile_variant(variant, output, llvm_ptx90=False):
    assert hasattr(fabric, "bind") == (variant != "pr"), (variant, fabric.__file__)
    has_availability = hasattr(libtriton.passes.ttgpuir, "add_optimize_thread_availability")
    assert has_availability == (variant == "optimized"), (variant, libtriton.__file__)
    output.mkdir(parents=True, exist_ok=True)
    buffer, out = inputs()
    manifest = dict(variant=variant, llvm_ptx90=llvm_ptx90, triton=triton.__file__, binary=libtriton.__file__,
                    binary_sha256=file_hash(libtriton.__file__), device=torch.cuda.get_device_name(), cases=[])
    for case in cases():
        prepare(buffer, out, case)
        compiled = comparison_kernel[(1, )](*arguments(buffer, out, case, variant), num_warps=case["num_warps"])
        torch.cuda.synchronize()
        validate(buffer, out, case)
        directory = output / case["name"]
        directory.mkdir(exist_ok=True)
        metadata = {}
        for name, source in compiled.metadata_group.items():
            dest = directory / Path(source).name
            shutil.copyfile(source, dest)
            metadata[name] = str(dest)
        record = dict(case=case, hash=compiled.hash, signature=compiled.src.signature, metadata=metadata,
                      registers=compiled.n_regs, shared=compiled.metadata.shared,
                      cubin_sha256=hashlib.sha256(compiled.asm["cubin"]).hexdigest())
        manifest["cases"].append(record)
        print(
            json.dumps({
                "compiled": variant, "case": case["name"], "registers": compiled.n_regs, "shared":
                compiled.metadata.shared
            }), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value


def load_kernel(record):
    signature = {key: tuples(value) for key, value in record["signature"].items()}
    src = SimpleNamespace(signature=signature, constants={}, fn=SimpleNamespace(arg_names=list(signature)))
    kernel = CompiledKernel(src, record["metadata"], record["hash"])
    assert hashlib.sha256(kernel.asm["cubin"]).hexdigest() == record["cubin_sha256"]
    kernel._init_handles()
    return kernel


def measure(root):
    variants = ("pr", "plain", "optimized")
    manifests = {variant: json.loads((root / variant / "manifest.json").read_text()) for variant in variants}
    assert len({len(m["cases"]) for m in manifests.values()}) == 1
    buffer, out = inputs()
    records = []
    for comparisons in zip(*(manifests[v]["cases"] for v in variants)):
        case = comparisons[0]["case"]
        assert all(r["case"] == case for r in comparisons)
        assert all(r["signature"] == comparisons[0]["signature"] for r in comparisons)
        kernels = {v: load_kernel(r) for v, r in zip(variants, comparisons)}
        timings = {variant: [] for variant in variants}
        # Cover all six permutations so each variant occupies every position.
        for round_index in range(6):
            offset = round_index % 3
            order = variants[offset:] + variants[:offset]
            if round_index >= 3:
                order = tuple(reversed(order))
            for variant in order:
                prepare(buffer, out, case)
                run = lambda: kernels[variant][(1, 1, 1)](*arguments(buffer, out, case, variant))
                run()
                torch.cuda.synchronize()
                validate(buffer, out, case)
                timings[variant].append(triton.testing.do_bench_cudagraph(run, rep=5, return_mode="median") * 1000)
                validate(buffer, out, case)
        medians = {variant: statistics.median(samples) for variant, samples in timings.items()}
        record = dict(
            case=case, us=timings, median_us=medians, optimized_over_plain=medians["optimized"] / medians["plain"],
            optimized_over_pr=medians["optimized"] / medians["pr"], code={
                variant: dict(registers=kernel.n_regs, shared=kernel.metadata.shared)
                for variant, kernel in kernels.items()
            })
        records.append(record)
        print(json.dumps(record), flush=True)
    (root / "results.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", choices=["pr", "plain", "optimized"])
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--llvm-ptx90", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/fabric-stack-comparison"))
    args = parser.parse_args()
    if args.compile:
        if args.llvm_ptx90:
            from triton.backends.nvidia import compiler as nvidia_compiler
            nvidia_compiler.get_features = lambda options, arch: "+ptx90"
        compile_variant(args.compile, args.output / args.compile, args.llvm_ptx90)
    elif args.measure:
        measure(args.output)
    else:
        parser.error("choose --compile or --measure")
