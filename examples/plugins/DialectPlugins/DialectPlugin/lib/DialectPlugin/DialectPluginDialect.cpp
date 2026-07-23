#include "DialectPlugin/DialectPluginDialect.h"
#include "DialectPlugin/DialectPluginOps.h"
#include "DialectPlugin/DialectPluginTypes.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::plugin;

#include "DialectPlugin/DialectPluginOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// DialectPlugin dialect.
//===----------------------------------------------------------------------===//

void DialectPluginDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "DialectPlugin/DialectPluginOps.cpp.inc"
      >();
  registerTypes();
}

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"

#include "DialectPlugin/DialectPluginDialect.h"
#include "DialectPlugin/DialectPluginPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "triton/Tools/PluginUtils.h"
#include "llvm/Config/llvm-config.h"
#include <nanobind/stl/string.h>

namespace py = nanobind;

static const char *PLUGIN_NAME = "DialectPlugin";
static const char *DIALECT_NAME = "DialectPlugin";
static const char *PASS_NAME = "plugingpu_conversion";
static const char *VERSION = "0.1.0";

static void addTritonPluginPass(mlir::PassManager *pm,
                                const std::vector<std::string> &args) {
  pm->addPass(mlir::triton::plugin::createConvertPluginGPUToLLVMPass());
}

static void registerTritonPluginPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::plugin::createConvertPluginGPUToLLVMPass();
  });
}

static void registerTritonPluginDialect(DialectRegistry *registry) {
  registry->insert<mlir::triton::plugin::DialectPluginDialect>();
  mlir::triton::plugin::registerpluginPasses();
}

static void addTritonPluginCustomOp(TritonOpBuilder &self,
                                    std::vector<mlir::Value> &operands) {
  ::mlir::Value &dst = operands[0];
  ::mlir::Value &src = operands[1];

  dst = self.create<arith::AddFOp>(src, src);
  operands[0] = dst;
}

static PyObject *addPyArgCustomOp(TritonOpBuilder &self, PyObject *argsObj,
                                  PyObject *kwargsObj) {
  py::args args = py::borrow<py::args>(argsObj);
  py::kwargs kwargs = py::borrow<py::kwargs>(kwargsObj);

  if (args.empty())
    return py::none().release().ptr();

  std::string mode = "add";
  if (kwargs.contains("mode"))
    mode = py::cast<std::string>(kwargs["mode"]);

  Value acc = py::cast<Value>(args[0]);
  for (size_t i = 1; i < args.size(); ++i) {
    Value v = py::cast<Value>(args[i]);
    if (mode == "mul")
      acc = self.create<arith::MulFOp>(acc, v);
    else
      acc = self.create<arith::AddFOp>(acc, v);
  }
  return py::cast(acc).release().ptr();
}

TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  static plugin::PassInfo pass = {PASS_NAME, VERSION, addTritonPluginPass,
                                  registerTritonPluginPass};
  static plugin::PassInfo passes[] = {pass};
  static plugin::DialectInfo dialect = {DIALECT_NAME, VERSION,
                                        registerTritonPluginDialect};
  static plugin::DialectInfo dialects[] = {dialect};
  static plugin::OpInfo op = {"custom_op", addTritonPluginCustomOp};
  static plugin::OpInfo pyOp = {"py_custom_op", nullptr, addPyArgCustomOp};
  static plugin::OpInfo ops[] = {op, pyOp};
  static plugin::PluginInfo info = {TRITON_PLUGIN_API_VERSION,
                                    PLUGIN_NAME,
                                    VERSION,
                                    passes,
                                    1,
                                    dialects,
                                    1,
                                    ops,
                                    2,
                                    TRITON_VERSION};
  return &info;
}
