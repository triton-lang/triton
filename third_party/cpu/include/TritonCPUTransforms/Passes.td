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

    let options = [
        Option<"promoteBf16ToFp32", "promote-bf16-to-fp32",
               "bool", /*default*/"false",
               "Convert BF16 operations to FP32.">,
        Option<"convertMixedPrecisionMatmul", "convert-mixed-precision-matmul",
               "bool", /*default*/"false",
               "Convert inputs of a mixed-precision matmul to a destination type.">,
        Option<"promoteLibMathToFp32", "promote-lib-math-to-fp32",
               "bool", /*default*/"true",
               "Promote FP8, FP16, BF16 math operations mapped to libm function to FP32.">,
    ];

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

    let options = [
        Option<"decomposeBf16Conversions", "decompose-bf16-conversions",
               "bool", /*default*/"false",
               "Lower BF16 conversions to arith operations.">,
        Option<"decomposeFp8Conversions", "decompose-fp8-conversions",
               "bool", /*default*/"false",
               "Lower FP8 conversions to arith operations.">,
    ];

    let constructor = "mlir::triton::cpu::createDecomposeFpConversions()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

#endif
