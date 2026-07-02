#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

void init_triton_metal(nb::module_ &m) {
  auto passes = m.def_submodule("passes");

  m.def("load_dialects", [](mlir::MLIRContext &ctx) {
    ctx.loadDialect<mlir::triton::TritonDialect,
                    mlir::triton::gpu::TritonGPUDialect>();
  });
}
