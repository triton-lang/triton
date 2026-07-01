#ifndef TRITONGPU_CONVERSION_PASSES
#define TRITONGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonGPUToLLVM : Pass<"convert-triton-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let description = [{

    }];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::instrument::TritonInstrumentDialect",
                             "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                             "mlir::triton::nvgpu::NVGPUDialect",
                             "mlir::NVVM::NVVMDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
        Option<"ptxVersion", "ptx-version",
               "int32_t", /*default*/"80",
               "PTX version">,
        Option<"enableConcurrencySanitizer", "enable-concurrency-sanitizer",
               "bool", /*default*/"false",
               "run ConSan after fence and membar preparation">,
    ];
}

def InitializeWSClusterBarriers : Pass<"initialize-ws-cluster-barriers", "mlir::ModuleOp"> {
    let summary = "Initialize warp-specialized cluster barrier mbarriers";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::NVVM::NVVMDialect",
                             "mlir::triton::gpu::TritonGPUDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
        Option<"ptxVersion", "ptx-version",
               "int32_t", /*default*/"80",
               "PTX version">,
    ];
}

def AllocateSharedMemoryNv : Pass<"allocate-shared-memory-nv", "mlir::ModuleOp"> {
  let summary = "Add metadata for shared memory allocation for Nvidia";

  let description = [{
    See `allocate-shared-memory` for more details.
  }];

  let options = [
      Option<"computeCapability", "compute-capability",
             "int32_t", /*default*/"80",
             "device compute capability">,
      Option<"ptxVersion", "ptx-version",
             "int32_t", /*default*/"80",
             "PTX version">,
  ];
}


def ConvertWarpSpecializeToLLVM : Pass<"convert-warp-specialize-to-llvm", "mlir::ModuleOp"> {
  let summary = "lower `ttg.warp_specialize` to LLVM";
  let description = [{
    The `convert-warp-specialize-to-llvm` pass performs codegen for warp
    specialization. It is a function-level transformation that rewrites
    warp-specialized kernels by using shared memory and barriers to communicate
    states between the default warpgroup and the worker warps.
  }];
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::NVVM::NVVMDialect"];
}

def NVPerLaneLoopRetirement : Pass<"nv-per-lane-loop-retirement", "mlir::ModuleOp"> {
  let summary = "Let lanes retire independently from divergent while-loops";
  let description = [{
    Rewrites warp-uniform (lock-step) while-loop latches -- an
    `nvvm.redux.sync max` over a per-lane predicate feeding the conditional
    branch -- so the branch uses the per-lane predicate directly. Each lane
    then exits the loop as soon as its own condition fails (independent
    thread scheduling, sm_70+) instead of iterating to the warp's maximum
    trip count; the warp reconverges via `nvvm.bar.warp.sync` at the loop
    exit (on the `activemask` captured at loop entry). The rewrite is
    guarded by a structural verifier and refused unless it can prove
    observational equivalence with the lock-step schedule: the loop body
    contains no warp-collective or otherwise convergent operation, the loop
    has a single exit, every loop-carried value live-out of the loop is
    frozen on lane-inactive iterations (Triton's masked-update idioms
    `tl.where(active, x, old)` lower to recognized select/masked-identity
    forms), and the per-lane predicate is monotone (once false for a lane
    it can never become true again, so early retirement is unobservable).
  }];
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::NVVM::NVVMDialect"];
}

#endif // TRITONGPU_CONVERSION_PASSES
