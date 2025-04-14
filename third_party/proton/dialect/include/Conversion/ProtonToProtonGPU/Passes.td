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
              "profiling granularity: warp, warp_group, or cta">,
       Option<"selectIds", "select-ids",
              "std::string", /*default*/"\"\"",
              "comma-separated list of selected ids">,
       Option<"maxSharedMem", "max-shared-mem",
              "int32_t", /*default*/"0",
              "max shared memory size per CTA">,
       Option<"scratchMem", "scratch-mem",
              "int32_t", /*default*/"32768",
              "profiler global scratch memory size per CTA">,
       Option<"alignment", "alignment",
              "int32_t", /*default*/"128",
              "profiler global scratch memory alignment">,
       Option<"bufferStrategy", "buffer-strategy", "gpu::BufferStrategy", /*default*/"gpu::BufferStrategy::CIRCULAR",
              "profiler buffer recording strategy (circular or flush)">,
       Option<"bufferType", "buffer-type", "gpu::BufferType", /*default*/"gpu::BufferType::SHARED",
              "internal buffer type (SHARED, GLOBAL, LOCAL) that backs the profiling data">,
       Option<"bufferSize", "buffer-size", "int32_t", /*default*/"0",
              "internal buffer byte size that backs the profiling data">,
  ];
}

#endif
