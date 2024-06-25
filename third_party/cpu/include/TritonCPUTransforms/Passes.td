#ifndef TRITONCPUOPT_CONVERSION_PASSES
#define TRITONCPUOPT_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertUnsupportedOps : Pass<"triton-cpu-add-casts-for-unsupported-ops", "mlir::ModuleOp"> {
    let summary = "Convert operations on unsupported types.";
    let description = [{
        This pass converts various operations on data types that are not supported
        by the target natively. Operations are converted to a supported data type
        with casts added for inputs and the result.
    }];
    // TODO: add options to specify which operations to convert.
    let constructor = "mlir::triton::cpu::createConvertUnsupportedOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def DecomposeFpConversions : Pass<"triton-cpu-decompose-fp-conversions", "mlir::ModuleOp"> {
    let summary = "Decompose fp conversion ops.";
    let description = [{
        This pass is used for targets lacking native instructions to convert FP
        vectors. By default, LLVM would decompose them using scalar FP conversion
        intrinsics. This pass transforms such conversions into vector code
        instead.
    }];
    // TODO: add options to specify which FP conversions to decompose.
    let constructor = "mlir::triton::cpu::createDecomposeFpConversions()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

#endif
