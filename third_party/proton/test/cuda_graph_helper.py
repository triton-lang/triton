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
        value = value * 1.02 + 1.02
        iteration += 1
    tl.store(output + tag * count + offset, value, mask=active)


@triton.jit(do_not_specialize=["count", "iterations", "tag"])
def pdl_consumer(output, count, iterations, tag, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offset < count
    value = offset.to(tl.float32) + tag.to(tl.float32)
    iteration = 0
    while iteration < iterations:
        value = value * 1.02 + 1.02
        iteration += 1
    tl.store(output + tag * count + offset, value, mask=active)


C_SOURCE = r"""
#include <cuda.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    Pdl = 0,
    Event = 1,
    NoPdl = 2,
    KernelsPerGraph = 4,
    ConsumersPerGraph = 3,
    BlockSize = 256,
    Count = 4096,
};

static void checkCuda(CUresult result, const char *operation) {
    if (result == CUDA_SUCCESS)
        return;
    const char *description = NULL;
    cuGetErrorString(result, &description);
    fprintf(stderr, "%s failed: %s\n", operation,
            description ? description : "unknown CUDA error");
    exit(2);
}

#define CUDA(call) checkCuda((call), #call)

struct Runtime {
    int mode;
    CUdevice device;
    CUcontext context;
    CUmodule producerModule;
    CUmodule consumerModule;
    CUfunction producer;
    CUfunction consumer;
    CUstream stream;
    CUgraph graph;
    CUgraphExec executable;
    CUevent graphStart;
    CUevent graphEnd;
    CUgraphNode kernels[KernelsPerGraph];
    CUdeviceptr output;
    CUdeviceptr nullScratch;
    uint32_t count;
    uint32_t iterations[KernelsPerGraph];
    uint32_t tags[KernelsPerGraph];
    void *parameters[KernelsPerGraph][6];
};

static void addKernel(struct Runtime *runtime, int index, CUfunction function,
                      const CUgraphNode *dependencies,
                      size_t dependencyCount) {
    runtime->parameters[index][0] = &runtime->output;
    runtime->parameters[index][1] = &runtime->count;
    runtime->parameters[index][2] = &runtime->iterations[index];
    runtime->parameters[index][3] = &runtime->tags[index];
    runtime->parameters[index][4] = &runtime->nullScratch;
    runtime->parameters[index][5] = &runtime->nullScratch;

    CUDA_KERNEL_NODE_PARAMS params;
    memset(&params, 0, sizeof(params));
    params.func = function;
    params.gridDimX = (runtime->count + BlockSize - 1) / BlockSize;
    params.gridDimY = 1;
    params.gridDimZ = 1;
    params.blockDimX = BlockSize;
    params.blockDimY = 1;
    params.blockDimZ = 1;
    params.kernelParams = runtime->parameters[index];
    CUDA(cuGraphAddKernelNode(&runtime->kernels[index], runtime->graph,
                              dependencies, dependencyCount, &params));
}

static void buildGraph(struct Runtime *runtime) {
    CUDA(cuGraphCreate(&runtime->graph, 0));
    runtime->count = Count;
    for (int index = 0; index < KernelsPerGraph; ++index) {
        runtime->tags[index] = (uint32_t)index;
        runtime->iterations[index] = 10000;
    }

    if (runtime->mode == Event) {
        CUgraphNode dependency;
        CUDA(cuEventCreate(&runtime->graphStart, CU_EVENT_DEFAULT));
        CUDA(cuEventCreate(&runtime->graphEnd, CU_EVENT_DEFAULT));
        CUDA(cuGraphAddEventRecordNode(&dependency, runtime->graph, NULL, 0,
                                       runtime->graphStart));
        for (int index = 0; index < KernelsPerGraph; ++index) {
            addKernel(runtime, index, runtime->consumer, &dependency, 1);
            dependency = runtime->kernels[index];
        }
        CUDA(cuGraphAddEventRecordNode(&dependency, runtime->graph,
                                       &runtime->kernels[KernelsPerGraph - 1],
                                       1, runtime->graphEnd));
    } else {
        runtime->iterations[0] = 25000;
        addKernel(runtime, 0, runtime->producer, NULL, 0);
        for (int index = 1; index < KernelsPerGraph; ++index) {
            if (runtime->mode == Pdl) {
                addKernel(runtime, index, runtime->consumer, NULL, 0);
                CUgraphEdgeData edge;
                memset(&edge, 0, sizeof(edge));
                edge.from_port = CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC;
                edge.type = CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC;
                CUDA(cuGraphAddDependencies_v2(
                    runtime->graph, &runtime->kernels[0],
                    &runtime->kernels[index], &edge, 1));
            } else {
                addKernel(runtime, index, runtime->consumer,
                          &runtime->kernels[0], 1);
            }
        }
    }
    CUDA(cuGraphInstantiateWithFlags(&runtime->executable, runtime->graph, 0));
}

__attribute__((visibility("default")))
void *graphRuntimeCreate(int mode, const char *producerPath,
                         const char *consumerPath,
                         const char *producerName,
                         const char *consumerName) {
    struct Runtime *runtime = calloc(1, sizeof(*runtime));
    if (!runtime)
        return NULL;
    runtime->mode = mode;

    CUDA(cuInit(0));
    CUDA(cuDeviceGet(&runtime->device, 0));
    CUDA(cuDevicePrimaryCtxRetain(&runtime->context, runtime->device));
    CUDA(cuCtxSetCurrent(runtime->context));
    CUDA(cuModuleLoad(&runtime->producerModule, producerPath));
    CUDA(cuModuleLoad(&runtime->consumerModule, consumerPath));
    CUDA(cuModuleGetFunction(&runtime->producer, runtime->producerModule,
                             producerName));
    CUDA(cuModuleGetFunction(&runtime->consumer, runtime->consumerModule,
                             consumerName));
    CUDA(cuMemAlloc(&runtime->output,
                    KernelsPerGraph * Count * sizeof(float)));
    CUDA(cuStreamCreate(&runtime->stream, CU_STREAM_NON_BLOCKING));
    buildGraph(runtime);
    return runtime;
}

__attribute__((visibility("default")))
void graphRuntimeLaunch(void *handle) {
    struct Runtime *runtime = handle;
    CUDA(cuGraphLaunch(runtime->executable, runtime->stream));
    if (runtime->mode == Event)
        CUDA(cuEventSynchronize(runtime->graphEnd));
    else
        CUDA(cuStreamSynchronize(runtime->stream));
}

__attribute__((visibility("default")))
void graphRuntimeDestroy(void *handle) {
    struct Runtime *runtime = handle;
    if (!runtime)
        return;
    CUDA(cuCtxSetCurrent(runtime->context));
    CUDA(cuGraphExecDestroy(runtime->executable));
    CUDA(cuGraphDestroy(runtime->graph));
    if (runtime->graphStart)
        CUDA(cuEventDestroy(runtime->graphStart));
    if (runtime->graphEnd)
        CUDA(cuEventDestroy(runtime->graphEnd));
    CUDA(cuStreamDestroy(runtime->stream));
    CUDA(cuMemFree(runtime->output));
    CUDA(cuModuleUnload(runtime->consumerModule));
    CUDA(cuModuleUnload(runtime->producerModule));
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
        self.library.graphRuntimeCreate.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self.library.graphRuntimeCreate.restype = ctypes.c_void_p
        self.library.graphRuntimeLaunch.argtypes = [ctypes.c_void_p]
        self.library.graphRuntimeDestroy.argtypes = [ctypes.c_void_p]
        self.handle = self.library.graphRuntimeCreate(
            CASE_IDS[case],
            os.fsencode(build.producer_path),
            os.fsencode(build.consumer_path),
            os.fsencode(build.producer_name),
            os.fsencode(build.consumer_name),
        )
        if not self.handle:
            raise RuntimeError("C graph runtime creation failed")

    def launch(self) -> None:
        self.library.graphRuntimeLaunch(self.handle)

    def close(self) -> None:
        if self.handle:
            self.library.graphRuntimeDestroy(self.handle)
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

    import triton.profiler as proton

    case = sys.argv[1]
    replays = int(sys.argv[2])
    profile_path = Path(sys.argv[3])
    build = RuntimeBuild(
        library_path=Path(sys.argv[4]),
        producer_path=Path(sys.argv[5]),
        consumer_path=Path(sys.argv[6]),
        producer_name=sys.argv[7],
        consumer_name=sys.argv[8],
    )

    # cuInit initializes the driver, not a CUDA context. Start Proton next so
    # HES is enabled before GraphRuntime retains and sets the primary context.
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    if status := cuda.cuInit(0):
        raise RuntimeError(f"cuInit failed with status {status}")

    session = proton.start(
        str(profile_path.with_suffix("")),
        context="shadow",
        data="tree",
        backend="cupti",
    )
    runtime = GraphRuntime(build, case)
    proton.deactivate(session, flushing=False)
    runtime.launch()
    proton.activate(session)
    for replay in range(replays):
        with proton.scope(f"{case}_{replay}"):
            runtime.launch()
    proton.finalize(session, "")
    runtime.close()


if __name__ == "__main__":
    main()
