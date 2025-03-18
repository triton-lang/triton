#ifndef PROTON_TO_PROTONGPU_PASSES
#define PROTON_TO_PROTONGPU_PASSES

include "mlir/Pass/PassBase.td"

def ConvertProtonToProtonGPU: Pass<"convert-proton-to-protongpu", "mlir::ModuleOp"> {
  let summary = "Lowering pass of ProtonIR to ProtonGPU IR";

  let description = "Convert the Proton Op into ProtonGPU Op. This includes scaffolding operations"
                    "such as allocation for internal profiling buffers, resources binding, and final cleanup.";

  let constructor = "mlir::triton::proton::createConvertProtonToProtonGPUPass(\"cycle\", \"warpgroup\", 0, 32768, 128, \"circular\", \"shared_mem\", 0)";

  let dependentDialects = ["mlir::triton::proton::ProtonDialect",
                           "mlir::triton::proton::gpu::ProtonGPUDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

    let options = [
       Option<"metric", "metric",
              "std::string", /*default*/"\"cycle\"",
              "performance counter metric our lowering targets to">,
       Option<"granularity", "granularity",
              "std::string", /*default*/"\"warpgroup\"",
              "profiling granularity: warp or warpgroup">,
       Option<"maxSharedMem", "max-shared-mem",
              "int32_t", /*default*/"0",
              "max shared memory size per CTA">,
       Option<"scratchMem", "scratch-mem",
              "int32_t", /*default*/"32768",
              "profiler global scratch memory size per CTA">,
       Option<"alignment", "alignment",
              "int32_t", /*default*/"128",
              "profiler global scratch memory alignment">,
       Option<"strategy", "strategy", "std::string", /*default*/"\"circular\"",
              "profiler record strategy (circular or flush)">,
       Option<"bufferType", "buffer-type", "std::string", /*default*/"\"shared_mem\"",
              "internal buffer type (shared_mem or stack_mem) that backs the profiling data">,
       Option<"bufferSize", "buffer-size", "int32_t", /*default*/"0",
              "internal buffer byte size that backs the profiling data">,
  ];
}

#endif
