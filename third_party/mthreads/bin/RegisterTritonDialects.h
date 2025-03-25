#pragma once

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/LLVMIR/MTGPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/InitAllPasses.h"

#include <dlfcn.h>
#include <iostream>
#include <string>

#include "python/src/plugin.h"

using BackendRegisterFunc = void (*)();

BackendRegisterFunc load_backend_register_func(const char *backend_name,
                                               const char *func_name) {
  void *symbol = load_backend_symbol(backend_name, func_name);
  return reinterpret_cast<BackendRegisterFunc>(symbol);
}

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::registerTritonPasses();

  mlir::triton::gpu::registerTritonGPUPasses();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerAllocateSharedMemoryPass();
  mlir::registerLLVMDIScope();

  // TODO(mthreads): registerMthreadsPasses is not working currently,
  // since both libtriton.so and mthreadsTritonPlugin.so are linked the
  // MLIRPass.a
  auto backend_register_func =
      load_backend_register_func("mthreads", "registerMthreadsPasses");
  backend_register_func();

  // TODO: register Triton & TritonGPU passes
  registry.insert<mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
                  mlir::triton::gpu::TritonGPUDialect, mlir::math::MathDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                  mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
                  mlir::NVVM::NVVMDialect, mlir::MTGPU::MTGPUDialect>();
}
