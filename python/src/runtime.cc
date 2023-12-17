#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

enum backend_t {
  HOST,
  CUDA,
  ROCM,
};

void init_triton_runtime(py::module &&m) {
  // wrap backend_t
  py::enum_<backend_t>(m, "backend", py::module_local())
      .value("HOST", HOST)
      .value("CUDA", CUDA)
      .value("ROCM", ROCM)
      .export_values();

  py::enum_<mlir::triton::Target>(m, "TARGET")
      .value("NVVM", mlir::triton::NVVM)
      .value("ROCDL", mlir::triton::ROCDL)
      .export_values();
}
