//===-- TritonDialects.cpp - Extension module ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "triton-c/TritonDialects.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"

#include <vector>

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_tritonDialects, mainModule) {
#ifndef NDEBUG
  static std::string executable =
      llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  llvm::sys::PrintStackTraceOnErrorSignal(executable);
#endif

  //===--------------------------------------------------------------------===//
  // Triton dialect.
  //===--------------------------------------------------------------------===//
  auto tritonModule = mainModule.def_submodule("triton");

  //
  // Dialect
  //

  tritonModule.def(
      "register_dialect",
      [](MlirContext context, bool doLoad) {
        MlirDialectHandle handle = mlirGetDialectHandle__triton__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (doLoad) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  //===--------------------------------------------------------------------===//
  // Triton GPU dialect.
  //===--------------------------------------------------------------------===//
  auto tritonGpuModule = mainModule.def_submodule("triton_gpu");

  //
  // Dialect
  //

  tritonGpuModule.def(
      "register_dialect",
      [](MlirContext context, bool doLoad) {
        MlirDialectHandle handle = mlirGetDialectHandle__triton_gpu__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (doLoad) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
