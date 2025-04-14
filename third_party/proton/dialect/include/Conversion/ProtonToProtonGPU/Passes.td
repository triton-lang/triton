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
              "The performance counter metric type we are profiling">,
       Option<"granularity", "granularity",
              "gpu::Granularity", /*default*/"gpu::Granularity::WARP",
              "Profiling granularity: warp, warp_group, or cta">,
       Option<"samplingStrategy", "sampling-strategy",
              "SamplingStrategy", /*default*/"SamplingStrategy::NONE",
              "Profiling sampling strategy">,
       Option<"samplingOptions", "sampling-options",
              "std::string", /*default*/"\"\"",
              "Profiling sampling options">,
       Option<"bufferStrategy", "buffer-strategy", "gpu::BufferStrategy", /*default*/"gpu::BufferStrategy::CIRCULAR",
              "Profiler buffer recording strategy (circular or flush)">,
       Option<"bufferType", "buffer-type", "gpu::BufferType", /*default*/"gpu::BufferType::SHARED",
              "Internal buffer type (SHARED, GLOBAL, LOCAL) that stores the profiling data">,
       Option<"bufferSize", "buffer-size", "int32_t", /*default*/"0",
              "Internal buffer byte size that stores the profiling data">,
       Option<"profileScratchSize", "scratch-mem-size",
              "int64_t", /*default*/"32768",
              "Profiler global scratch memory size per CTA">,
       Option<"profileScratchAlignment", "scratch-mem-alignment",
              "int32_t", /*default*/"128",
              "Profiler global scratch memory alignment">,
  ];
}

#endif
