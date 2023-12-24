#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Target/PTX/TmaMetadata.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <fstream>
#include <sstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <signal.h>

namespace py = pybind11;

void init_triton_translation(py::module &&m) {
  using ret = py::return_value_policy;

  m.def("add_external_libs",
        [](mlir::ModuleOp &module, const std::vector<std::string> &names,
           const std::vector<std::string> &paths) {
          if (names.empty() || names.size() != paths.size())
            return;

          llvm::SmallVector<mlir::NamedAttribute, 2> attrs;

          for (size_t i = 0; i < names.size(); ++i) {
            auto name = mlir::StringAttr::get(module->getContext(), names[i]);
            auto path = mlir::StringAttr::get(module->getContext(), paths[i]);
            mlir::NamedAttribute attr(name, path);
            attrs.push_back(attr);
          }

          mlir::DictionaryAttr dict =
              mlir::DictionaryAttr::get(module->getContext(), attrs);
          module.getOperation()->setAttr("triton_gpu.externs", dict);
        });
}
