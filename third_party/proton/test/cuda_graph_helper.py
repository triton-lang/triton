"""Triton kernels and a C-built CUDA graph runtime for Proton tests."""

from __future__ import annotations

import ctypes
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.runtime import driver

CASES = ("pdl", "event", "no-pdl")
CASE_IDS = {case: index for index, case in enumerate(CASES)}
BLOCK_SIZE = 256
KERNELS_PER_GRAPH = 4
CONSUMERS_PER_GRAPH = 3


@triton.jit(do_not_specialize=["count", "iterations", "tag"])
def pdl_producer(output, count, iterations, tag, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offset < count
    tl.extra.cuda.gdc_launch_dependents()
    value = offset.to(tl.float32) + tag.to(tl.float32)
    iteration = 0
    while iteration < iterations:
        value = value * 1.0000001192092896 + 0.0000001192092896
        iteration += 1
    tl.store(output + tag * count + offset, value, mask=active)


@triton.jit(do_not_specialize=["count", "iterations", "tag"])
def pdl_consumer(output, count, iterations, tag, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offset < count
    value = offset.to(tl.float32) + tag.to(tl.float32)
    iteration = 0
    while iteration < iterations:
        value = value * 1.0000001192092896 + 0.0000001192092896
        iteration += 1
    tl.store(output + tag * count + offset, value, mask=active)


C_SOURCE = r"""
#include <cuda.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    PDL = 0,
    EVENT = 1,
    NO_PDL = 2,
    KERNELS_PER_GRAPH = 4,
    CONSUMERS_PER_GRAPH = 3,
    BLOCK_SIZE = 256,
    COUNT = 4096,
};

static void check_cuda(CUresult result, const char *operation) {
    if (result == CUDA_SUCCESS)
        return;
    const char *description = NULL;
    cuGetErrorString(result, &description);
    fprintf(stderr, "%s failed: %s\n", operation,
            description ? description : "unknown CUDA error");
    exit(2);
}

#define CUDA(call) check_cuda((call), #call)

struct runtime {
    int mode;
    CUdevice device;
    CUcontext context;
    CUmodule producer_module;
    CUmodule consumer_module;
    CUfunction producer;
    CUfunction consumer;
    CUstream stream;
    CUgraph graph;
    CUgraphExec executable;
    CUevent graph_start;
    CUevent graph_end;
    CUgraphNode kernels[KERNELS_PER_GRAPH];
    CUdeviceptr output;
    CUdeviceptr null_scratch;
    uint32_t count;
    uint32_t iterations[KERNELS_PER_GRAPH];
    uint32_t tags[KERNELS_PER_GRAPH];
    void *parameters[KERNELS_PER_GRAPH][6];
};

static void add_kernel(struct runtime *runtime, int index, CUfunction function,
                       const CUgraphNode *dependencies,
                       size_t dependency_count) {
    runtime->parameters[index][0] = &runtime->output;
    runtime->parameters[index][1] = &runtime->count;
    runtime->parameters[index][2] = &runtime->iterations[index];
    runtime->parameters[index][3] = &runtime->tags[index];
    runtime->parameters[index][4] = &runtime->null_scratch;
    runtime->parameters[index][5] = &runtime->null_scratch;

    CUDA_KERNEL_NODE_PARAMS params;
    memset(&params, 0, sizeof(params));
    params.func = function;
    params.gridDimX = (runtime->count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    params.gridDimY = 1;
    params.gridDimZ = 1;
    params.blockDimX = BLOCK_SIZE;
    params.blockDimY = 1;
    params.blockDimZ = 1;
    params.kernelParams = runtime->parameters[index];
    CUDA(cuGraphAddKernelNode(&runtime->kernels[index], runtime->graph,
                              dependencies, dependency_count, &params));
}

static void build_graph(struct runtime *runtime) {
    CUDA(cuGraphCreate(&runtime->graph, 0));
    runtime->count = COUNT;
    for (int index = 0; index < KERNELS_PER_GRAPH; ++index) {
        runtime->tags[index] = (uint32_t)index;
        runtime->iterations[index] = 10000;
    }

    if (runtime->mode == EVENT) {
        CUgraphNode dependency;
        CUDA(cuEventCreate(&runtime->graph_start, CU_EVENT_DEFAULT));
        CUDA(cuEventCreate(&runtime->graph_end, CU_EVENT_DEFAULT));
        CUDA(cuGraphAddEventRecordNode(&dependency, runtime->graph, NULL, 0,
                                       runtime->graph_start));
        for (int index = 0; index < KERNELS_PER_GRAPH; ++index) {
            add_kernel(runtime, index, runtime->consumer, &dependency, 1);
            dependency = runtime->kernels[index];
        }
        CUDA(cuGraphAddEventRecordNode(&dependency, runtime->graph,
                                       &runtime->kernels[KERNELS_PER_GRAPH - 1],
                                       1, runtime->graph_end));
    } else {
        runtime->iterations[0] = 25000;
        add_kernel(runtime, 0, runtime->producer, NULL, 0);
        for (int index = 1; index < KERNELS_PER_GRAPH; ++index) {
            if (runtime->mode == PDL) {
                add_kernel(runtime, index, runtime->consumer, NULL, 0);
                CUgraphEdgeData edge;
                memset(&edge, 0, sizeof(edge));
                edge.from_port = CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC;
                edge.type = CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC;
                CUDA(cuGraphAddDependencies_v2(
                    runtime->graph, &runtime->kernels[0],
                    &runtime->kernels[index], &edge, 1));
            } else {
                add_kernel(runtime, index, runtime->consumer,
                           &runtime->kernels[0], 1);
            }
        }
    }
    CUDA(cuGraphInstantiateWithFlags(&runtime->executable, runtime->graph, 0));
}

__attribute__((visibility("default")))
void *graph_runtime_create(int mode, const char *producer_path,
                           const char *consumer_path,
                           const char *producer_name,
                           const char *consumer_name) {
    struct runtime *runtime = calloc(1, sizeof(*runtime));
    if (!runtime)
        return NULL;
    runtime->mode = mode;

    CUDA(cuInit(0));
    CUDA(cuDeviceGet(&runtime->device, 0));
    CUDA(cuDevicePrimaryCtxRetain(&runtime->context, runtime->device));
    CUDA(cuCtxSetCurrent(runtime->context));
    CUDA(cuModuleLoad(&runtime->producer_module, producer_path));
    CUDA(cuModuleLoad(&runtime->consumer_module, consumer_path));
    CUDA(cuModuleGetFunction(&runtime->producer, runtime->producer_module,
                             producer_name));
    CUDA(cuModuleGetFunction(&runtime->consumer, runtime->consumer_module,
                             consumer_name));
    CUDA(cuMemAlloc(&runtime->output,
                    KERNELS_PER_GRAPH * COUNT * sizeof(float)));
    CUDA(cuStreamCreate(&runtime->stream, CU_STREAM_NON_BLOCKING));
    build_graph(runtime);
    return runtime;
}

__attribute__((visibility("default")))
void graph_runtime_launch(void *handle) {
    struct runtime *runtime = handle;
    CUDA(cuGraphLaunch(runtime->executable, runtime->stream));
    if (runtime->mode == EVENT)
        CUDA(cuEventSynchronize(runtime->graph_end));
    else
        CUDA(cuStreamSynchronize(runtime->stream));
}

__attribute__((visibility("default")))
void graph_runtime_destroy(void *handle) {
    struct runtime *runtime = handle;
    if (!runtime)
        return;
    CUDA(cuCtxSetCurrent(runtime->context));
    CUDA(cuGraphExecDestroy(runtime->executable));
    CUDA(cuGraphDestroy(runtime->graph));
    if (runtime->graph_start)
        CUDA(cuEventDestroy(runtime->graph_start));
    if (runtime->graph_end)
        CUDA(cuEventDestroy(runtime->graph_end));
    CUDA(cuStreamDestroy(runtime->stream));
    CUDA(cuMemFree(runtime->output));
    CUDA(cuModuleUnload(runtime->consumer_module));
    CUDA(cuModuleUnload(runtime->producer_module));
    CUDA(cuDevicePrimaryCtxRelease(runtime->device));
    free(runtime);
}
"""


@dataclasses.dataclass(frozen=True)
class RuntimeBuild:
    library_path: Path
    producer_path: Path
    consumer_path: Path
    producer_name: str
    consumer_name: str


class GraphRuntime:

    def __init__(self, build: RuntimeBuild, case: str) -> None:
        self.library = ctypes.CDLL(str(build.library_path))
        self.library.graph_runtime_create.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self.library.graph_runtime_create.restype = ctypes.c_void_p
        self.library.graph_runtime_launch.argtypes = [ctypes.c_void_p]
        self.library.graph_runtime_destroy.argtypes = [ctypes.c_void_p]
        self.handle = self.library.graph_runtime_create(
            CASE_IDS[case],
            os.fsencode(build.producer_path),
            os.fsencode(build.consumer_path),
            os.fsencode(build.producer_name),
            os.fsencode(build.consumer_name),
        )
        if not self.handle:
            raise RuntimeError("C graph runtime creation failed")

    def launch(self) -> None:
        self.library.graph_runtime_launch(self.handle)

    def close(self) -> None:
        if self.handle:
            self.library.graph_runtime_destroy(self.handle)
            self.handle = None


def _compile_kernel(kernel, target) -> tuple[bytes, str]:
    source = ASTSource(
        kernel,
        {
            "output": "*fp32",
            "count": "u32",
            "iterations": "u32",
            "tag": "u32",
        },
        constexprs={"BLOCK_SIZE": BLOCK_SIZE},
    )
    compiled = triton.compile(
        source,
        target=target,
        options={"num_warps": 8, "num_stages": 1},
    )
    if compiled.metadata.global_scratch_size or compiled.metadata.profile_scratch_size:
        raise RuntimeError(f"{compiled.name} unexpectedly requires scratch storage")
    return compiled.asm["cubin"], compiled.name


def build_runtime(build_dir: Path, cuda_home: Path) -> RuntimeBuild:
    target = driver.active.get_current_target()
    if target.backend != "cuda" or int(target.arch) < 100:
        raise RuntimeError(f"CUDA Blackwell is required, got {target}")

    producer_cubin, producer_name = _compile_kernel(pdl_producer, target)
    consumer_cubin, consumer_name = _compile_kernel(pdl_consumer, target)
    producer_path = build_dir / "pdl_producer.cubin"
    consumer_path = build_dir / "pdl_consumer.cubin"
    producer_path.write_bytes(producer_cubin)
    consumer_path.write_bytes(consumer_cubin)

    source_path = build_dir / "cuda_graph_runtime.c"
    source_path.write_text(C_SOURCE)
    library_path = build_dir / "cuda_graph_runtime.so"
    subprocess.run(
        [
            "gcc",
            "-O2",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fPIC",
            "-shared",
            f"-I{cuda_home / 'include'}",
            str(source_path),
            f"-L{cuda_home / 'lib64'}",
            "-lcuda",
            "-o",
            str(library_path),
        ],
        check=True,
    )
    return RuntimeBuild(
        library_path=library_path,
        producer_path=producer_path,
        consumer_path=consumer_path,
        producer_name=producer_name,
        consumer_name=consumer_name,
    )


def main() -> None:
    triton.knobs.proton.cupti_lib_dir = triton.knobs.proton.cupti_lib_blackwell_dir
    triton.knobs.proton.enable_hw_trace = True

    from triton._C.libproton import proton as libproton

    case = sys.argv[1]
    replays = int(sys.argv[2])
    trace_path = Path(sys.argv[3])
    build = RuntimeBuild(
        library_path=Path(sys.argv[4]),
        producer_path=Path(sys.argv[5]),
        consumer_path=Path(sys.argv[6]),
        producer_name=sys.argv[7],
        consumer_name=sys.argv[8],
    )

    # cuInit initializes the driver, not a CUDA context. Start Proton through
    # the low-level entry point next so HES is enabled before GraphRuntime
    # retains and sets the primary context.
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    if status := cuda.cuInit(0):
        raise RuntimeError(f"cuInit failed with status {status}")

    session = libproton.start(
        str(trace_path.with_suffix("")),
        "shadow",
        "trace",
        "cupti",
        "",
    )
    runtime = GraphRuntime(build, case)
    finalized = False
    try:
        libproton.deactivate(session, False)
        runtime.launch()
        libproton.activate(session)
        for replay in range(replays):
            scope_name = f"{case}_{replay}"
            scope_id = libproton.record_scope()
            libproton.enter_scope(scope_id, scope_name)
            try:
                runtime.launch()
            finally:
                libproton.exit_scope(scope_id, scope_name)
        libproton.finalize(session, "")
        finalized = True
    finally:
        try:
            if not finalized:
                libproton.finalize(session, "")
        finally:
            runtime.close()


if __name__ == "__main__":
    main()
