#ifndef TRITONCPU_CONVERSION_PASSES
#define TRITONCPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def FuncOpToLLVM : Pass<"triton-cpu-func-op-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert FuncOp to LLVM for CPU.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createFuncOpToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def MemoryOpToLLVM : Pass<"triton-cpu-memory-op-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Triton memory operations to LLVM for CPU.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createMemoryOpToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def GetProgramIdOpToLLVM : Pass<"triton-cpu-get-program-id-op-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Triton GetProgramId to LLVM for CPU.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createGetProgramIdOpToLLVMPass()";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect"];
}

#endif
