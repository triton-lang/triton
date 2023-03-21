#ifndef TRITONGPU_CONVERSION_PASSES
#define TRITONGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertTritonGPUToLLVM : Pass<"convert-triton-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonGPUToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::ROCDL::ROCDLDialect",
                             "mlir::NVVM::NVVMDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">
    ];
}

def TritonConvertArithToIndex : Pass<"triton-convert-arith-to-index", "mlir::ModuleOp"> {

    let summary = "Convert arith to index";
    
    let constructor = "mlir::triton::createTritonConvertArithToIndexPass()";

    let description = [{
      Convert arith operation on index values to corresponding ops in the index dialect.
      We need this because SCFToCF conversion currently generates arith ops on indices.
    }];

    let dependentDialects = ["mlir::index::IndexDialect"];
}

#endif
