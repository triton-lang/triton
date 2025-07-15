#ifndef PROTON_TO_PROTONGPU_PASSES
#define PROTON_TO_PROTONGPU_PASSES

include "mlir/Pass/PassBase.td"

def ConvertProtonToProtonGPU: Pass<"convert-proton-to-protongpu", "mlir::ModuleOp"> {
  let summary = "Lowering pass of ProtonIR to ProtonGPU IR";

  let description = "Convert the Proton Op into ProtonGPU Op. This includes scaffolding operations"
                    "such as allocation for internal profiling buffers, resources binding, and final cleanup.";

  let constructor = "createConvertProtonToProtonGPUPass()";

  let dependentDialects = ["ProtonDialect",
                           "gpu::ProtonGPUDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

    let options = [
       Option<"metricType", "metric-type",
              "MetricType", /*default*/"MetricType::CYCLE",
              "The performance counter metric type we are profiling",
              /*parser*/[{::llvm::cl::values(
                    clEnumValN(MetricType::CYCLE, "cycle", "Cycle")
              )}]>,
       Option<"granularity", "granularity",
              "gpu::Granularity", /*default*/"gpu::Granularity::WARP",
              "Profiling granularity: warp, warp_group, or cta",
              /*parser*/[{::llvm::cl::values(
                    clEnumValN(gpu::Granularity::THREAD, "thread", "Thread"),
                    clEnumValN(gpu::Granularity::WARP, "warp", "Warp"),
                    clEnumValN(gpu::Granularity::WARP_2, "warp-2", "2 Warps"),
                    clEnumValN(gpu::Granularity::WARP_4, "warp-4", "4 Warps"),
                    clEnumValN(gpu::Granularity::WARP_8, "warp-8", "8 Warps"),
                    clEnumValN(gpu::Granularity::CTA, "cta", "CTA"),
                    clEnumValN(gpu::Granularity::WARP_GROUP, "warp-group", "Warp Group"),
                    clEnumValN(gpu::Granularity::WARP_GROUP_2, "warp-group-2", "2 Warp Groups"),
                    clEnumValN(gpu::Granularity::WARP_GROUP_4, "warp-group-4", "4 Warp Groups"),
                    clEnumValN(gpu::Granularity::WARP_GROUP_8, "warp-group-8", "8 Warp Groups")
              )}]>,
       Option<"samplingStrategy", "sampling-strategy",
              "SamplingStrategy", /*default*/"SamplingStrategy::NONE",
              "Profiling sampling strategy",
              /*parser*/[{::llvm::cl::values(
                    clEnumValN(SamplingStrategy::NONE, "none", "No Sampling"),
                    clEnumValN(SamplingStrategy::SELECTIVE, "selective", "Selective Sampling")
              )}]>,
       Option<"samplingOptions", "sampling-options",
              "std::string", /*default*/"\"\"",
              "Profiling sampling options">,
       Option<"bufferStrategy", "buffer-strategy", "gpu::BufferStrategy", /*default*/"gpu::BufferStrategy::CIRCULAR",
              "Profiler buffer recording strategy (circular or flush)",
              /*parser*/[{::llvm::cl::values(
                    clEnumValN(gpu::BufferStrategy::CIRCULAR, "circular", "Circular Buffer"),
                    clEnumValN(gpu::BufferStrategy::FLUSH, "flush", "Flush Buffer")
              )}]>,
       Option<"bufferType", "buffer-type", "gpu::BufferType", /*default*/"gpu::BufferType::SHARED",
              "Internal buffer type (SHARED, GLOBAL) that stores the profiling data",
              /*parser*/[{::llvm::cl::values(
                    clEnumValN(gpu::BufferType::SHARED, "shared", "Shared Memory"),
                    clEnumValN(gpu::BufferType::GLOBAL, "global", "Global Memory")
              )}]>,
       Option<"bufferSize", "buffer-size", "int32_t", /*default*/"0",
              "Internal buffer byte size that stores the profiling data. 0 means auto-size based on the device's `maxSharedMemSize`">,
       Option<"maxSharedMemSize", "max-shared-mem-size",
              "int32_t", /*default*/"32768",
              "Maximum available shared memory size per CTA">,
       Option<"profileScratchSize", "scratch-mem-size",
              "int64_t", /*default*/"32768",
              "Profiler global scratch memory size per CTA">,
       Option<"profileScratchAlignment", "scratch-mem-alignment",
              "int32_t", /*default*/"128",
              "Profiler global scratch memory alignment">,
       Option<"clockExtension", "clock-extension",
              "bool", /*default*/"false",
              "Use long clock if true, otherwise use 32-bit clock">,
  ];
}

#endif
