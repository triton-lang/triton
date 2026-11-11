"""Compare GSan acquire imports using the same launch on each Triton build.

Run with torchrun --standalone --nproc-per-node=4. Every CTA acquires the
same flags, distributed over the participating GPUs, then checks their data.
This isolates a many-CTA acquire pattern; it is not an application benchmark.
CUDA-event timings include GSan's device bookkeeping.
A queued delay before the events gives the host time to submit the launch, so
the measurement does not include an idle GPU waiting for Python.
"""

import argparse
import json
import os
import statistics
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.experimental.gsan import symmetric_memory


@triton.jit
def publish(state, RANK: tl.constexpr, FLAGS: tl.constexpr):
    offsets = tl.arange(0, FLAGS)
    tl.store(state + FLAGS + offsets, 1000 + RANK * FLAGS + offsets)
    tl.atomic_store(state + offsets, 1, sem="release", scope="sys")


@triton.jit
def acquire(flags, payload, out, FLAGS: tl.constexpr, KIND: tl.constexpr):
    offsets = tl.arange(0, FLAGS)
    pointers = tl.load(flags + offsets).to(tl.pointer_type(tl.int32))
    if KIND == "poll":
        ready = tl.atomic_poll(pointers, 1, sem="acquire", scope="sys")
    else:
        ready = tl.atomic_load(pointers, sem="acquire", scope="sys") == 1
    data = tl.load(tl.load(payload + offsets).to(tl.pointer_type(tl.int32)))
    tl.store(out + tl.program_id(0) * FLAGS + offsets, tl.where(ready, data, -1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--ctas", type=int, nargs="+", default=[152, 1824])
    parser.add_argument("--flags", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--kinds", nargs="+", default=["poll", "load"])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--measure-publication", action="store_true")
    parser.add_argument("--dump-dir", type=Path)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", timeout=timedelta(minutes=10))
    triton.knobs.compilation.instrumentation_mode = "gsan"
    symmetric_memory.configure(dist.group.WORLD, clock_buffer_size=1024)

    def dump(compiled, name):
        if rank == 0 and args.dump_dir is not None:
            args.dump_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("llir", "ptx"):
                (args.dump_dir / f"{name}.{suffix}").write_text(compiled.asm[suffix])

    if rank == 0:
        properties = torch.cuda.get_device_properties(rank)
        print(
            json.dumps(
                dict(metadata=True, label=args.label, world=world, gpu=properties.name,
                     sms=properties.multi_processor_count, clock_entries=world * properties.multi_processor_count,
                     clock_buffer_size=1024, num_warps=4, torch=torch.__version__, triton=triton.__version__,
                     triton_path=triton.__file__)), flush=True)
    for flags in args.flags:
        state = symmetric_memory.empty((2 * flags, ), dtype=torch.int32, device=f"cuda:{rank}", shadow_granularity=4)
        state.zero_()
        handle = symmetric_memory.rendezvous(state, group=dist.group.WORLD)
        buffers = [handle.get_buffer(peer, state.shape, state.dtype) for peer in range(world)]
        sources = [(rank + i + 1) % world for i in range(flags)]
        flag_ptrs = torch.tensor([buffers[peer].data_ptr() + 4 * i for i, peer in enumerate(sources)],
                                 dtype=torch.int64, device=f"cuda:{rank}")
        data_ptrs = flag_ptrs + 4 * flags
        expected = torch.tensor([1000 + peer * flags + i for i, peer in enumerate(sources)], dtype=torch.int32,
                                device=f"cuda:{rank}")
        published = publish[(1, )](state, rank, flags, num_warps=4)
        dump(published, f"publish-{flags}")
        torch.cuda.synchronize()
        dist.barrier()
        if args.measure_publication:
            samples = []
            for _ in range(args.samples):
                dist.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda._sleep(20_000_000)
                start.record()
                publish[(1, )](state, rank, flags, num_warps=4)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            row = dict(label=args.label, rank=rank, world=world, ctas=1, flags=flags, kind="publish",
                       samples_ms=samples, median_ms=statistics.median(samples), registers=published.n_regs,
                       spills=published.n_spills)
            rows = [None] * world
            dist.all_gather_object(rows, row)
            if rank == 0:
                print(json.dumps(rows), flush=True)
        for ctas in args.ctas:
            out = torch.empty((ctas, flags), dtype=torch.int32, device=f"cuda:{rank}")
            for kind in args.kinds:

                def launch():
                    return acquire[(ctas, )](flag_ptrs, data_ptrs, out, flags, kind, num_warps=4)

                compiled = launch()
                dump(compiled, f"{kind}-{flags}")
                torch.cuda.synchronize()
                torch.testing.assert_close(out, expected.expand_as(out))
                samples = []
                for _ in range(args.samples):
                    dist.barrier()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda._sleep(20_000_000)
                    start.record()
                    launch()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
                torch.testing.assert_close(out, expected.expand_as(out))
                row = dict(label=args.label, rank=rank, world=world, ctas=ctas, flags=flags, kind=kind,
                           samples_ms=samples, median_ms=statistics.median(samples), registers=compiled.n_regs,
                           spills=compiled.n_spills, correct=True)
                rows = [None] * world
                dist.all_gather_object(rows, row)
                if rank == 0:
                    print(json.dumps(rows), flush=True)
        # Import peer completion before the allocator can reuse this storage.
        # A host-only Gloo barrier does not update GSan's vector clocks.
        handle.barrier()
        torch.cuda.synchronize()
        dist.barrier()
        del buffers
        handle.close()
        del handle, state
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
