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

def OptimizeMasks : Pass<"triton-cpu-optimize-masks", "mlir::ModuleOp"> {
    let summary = "Optimize masked memory accesses.";
    let description = [{
        This pass tries to detect masked memory accesses with mask values that
        can be proven to be all-ones or all-zeros.
    }];

    let options = [
    ];

    let constructor = "mlir::triton::cpu::createOptimizeMasks()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertDotProduct : Pass<"triton-cpu-convert-dot-product", "mlir::ModuleOp"> {
    let summary = "Convert dot product op.";
    let description = [{
        This pass is used for indentifying dot product pattern
        (for example, elementwise mul followed by a sum) and
        converting it to dot product intrinsics like bfdot.
    }];

    let options = [
        Option<"useHorizontalSum", "use-horizontal-sum",
               "bool", /*default*/"true",
               "Use Horizontal Sum kernel for the dot product (gemv). Otherwise use a kernel with packing.">,
    ];

    let constructor = "mlir::triton::cpu::createConvertDotProduct()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertDotToAMX : Pass<"triton-cpu-convert-dot-to-amx", "mlir::ModuleOp"> {
    let summary = "Convert dot product op to AMX dialect.";
    let description = [{
        This pass is used to lower matmul operations to amx dialect.
    }];

    let options = [
        Option<"convertInt8", "convert-i8",
               "bool", /*default*/"false",
               "Use AMX extensions for int8 type.">,
        Option<"convertFp16", "convert-fp16",
               "bool", /*default*/"false",
               "Use AMX extensions for ifp16 type.">,
        Option<"convertBf16", "convert-bf16",
               "bool", /*default*/"false",
               "Use AMX extensions for bf16 type.">,
    ];

    let constructor = "mlir::triton::cpu::createConvertDotToAMX()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::x86::X86Dialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertDotToFMA : Pass<"triton-cpu-convert-dot-to-fma", "mlir::ModuleOp"> {
    let summary = "Decompose dot product op to a series of FMA operations.";
    let description = [{ }];

    let constructor = "mlir::triton::cpu::createConvertDotToFMA()";
    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertDotOpToUkernelOps : Pass<"triton-cpu-convert-dot-to-ukernels", "mlir::ModuleOp"> {
    let summary = "Convert dot product op to ukernel ops.";
    let description = [{
        This pass is used to lower DotOp operations to ukernel ops.
    }];

    let options = [
        Option<"ukernels", "ukernels",
               "mlir::triton::cpu::Ukernels", /*default*/"mlir::triton::cpu::Ukernels::OneDNN",
               "Ukernels provider to be used for replacement of dot op (OneDNN/XSMM/etc).",
               [{::llvm::cl::values(
               clEnumValN(mlir::triton::cpu::Ukernels::OneDNN, "oneDNN",
                "Use OneDNN as a ukernels provider"),
               clEnumValN(mlir::triton::cpu::Ukernels::XSMM, "XSMM",
                "Use XSMM as a ukernels provider")
              )}]>,
    ];

    let constructor = "mlir::triton::cpu::createConvertDotOpToUkernelOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertDotGeneric : Pass<"triton-cpu-convert-dot-generic", "mlir::ModuleOp"> {
    let summary = "Generic convertion of dot product op.";
    let description = [{
        This pass is used to lower matmul operations to generic vector code.
    }];

    let constructor = "mlir::triton::cpu::createConvertDotGeneric()";

    let dependentDialects = ["mlir::vector::VectorDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertVectorContractToArmNeonBFMMLA : Pass<"triton-cpu-convert-vector-contract-to-arm-neon-bfmmla", "mlir::ModuleOp"> {
    let summary = "Lower eligible vector contractions to Arm Neon BFMMLA.";
    let description = [{
        Normalize eligible rank-2 BF16 contractions with FP32 accumulators
        from ordinary matmul to MMT by transposing the RHS and updating its
        indexing map, then replace eligible MMT contractions with shuffle
        packing and Arm Neon BFMMLA instructions. Already-MMT contractions
        need no transpose. Unsupported contractions remain unchanged for
        existing unsupported-operation legalization.
    }];

    let constructor = "mlir::triton::cpu::createConvertVectorContractToArmNeonBFMMLA()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::arm_neon::ArmNeonDialect"];
}

def ConvertDotToNanokernel : Pass<"triton-cpu-convert-dot-to-nanokernel", "mlir::ModuleOp"> {
    let summary = "Convert dot product op to vector contraction and apply upstream patterns.";
    let description = [{
        This pass attempts to lower tl.dot operations inside an accumulation
        loop to optimal vector code for AMX, AVX-512, AVX10.2 and
        AVX_NE_CONVERT, using upstream MLIR patterns.
    }];

    let options = [
        Option<"cpuFeatures", "cpu-features",
               "std::string", /*default*/"",
               "Comma-separated list of CPU features available for lowering of dot op.">
    ];

    let constructor = "mlir::triton::cpu::createConvertDotToNanokernel()";

    let dependentDialects = ["mlir::vector::VectorDialect",
                             "mlir::x86::X86Dialect"];
}

def UnrollAndReorderElementwiseOps : Pass<"triton-cpu-unroll-and-reorder-elementwise-ops", "mlir::ModuleOp"> {
    let summary = "Unroll and reorder elementwise computations.";
    let description = [{
        This pass transforms DAGs of elementwise operations to heuristically shorten value lifetimes/lower register
        pressure.
    }];

    let options = [
        Option<"cpuFeatures", "cpu-features",
               "std::string", /*default*/"",
               "Comma-separated list of CPU features.">
    ];

    let constructor = "mlir::triton::cpu::createUnrollAndReorderElementwiseOps()";

    let dependentDialects = ["mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect"];
}

def Canonicalize : Pass<"triton-cpu-canonicalize", "mlir::ModuleOp"> {
    let summary = "Canonicalization pass.";
    let description = [{
        This pass applies various foldings to simplify analysis and transformations
        in optimization passes.
    }];

    let constructor = "mlir::triton::cpu::createCanonicalize()";

    let dependentDialects = ["mlir::vector::VectorDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

#endif
