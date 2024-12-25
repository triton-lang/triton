#include "Dialect/Proton/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

using namespace mlir;
using namespace triton;

class ProtonOpBuilder {
public:
  ProtonOpBuilder(MLIRContext *context) {
    builder = std::make_unique<OpBuilder>(context);
    lastLoc = std::make_unique<Location>(builder->getUnknownLoc());
  }

  OpBuilder &getBuilder() { return *builder; }
  MLIRContext *getContext() { return builder->getContext(); }

  Location getLastLoc() {
    assert(lastLoc);
    return *lastLoc;
  }
  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    auto loc = getLastLoc();
    return builder->create<OpTy>(loc, std::forward<Args>(args)...);
  }

private:
  std::unique_ptr<OpBuilder> builder;
  std::unique_ptr<Location> lastLoc;
};

} // anonymous namespace



void init_triton_proton(py::module &&m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;	
  using namespace ::mlir::triton::proton;
  auto passes = m.def_submodule("passes");

  py::class_<ProtonOpBuilder>(m, "builder", py::module_local(),
                              py::dynamic_attr())
      .def(py::init<mlir::MLIRContext *>())
      
  .def("create_record",
    [](ProtonOpBuilder &self, bool isStart, int32_t regionId) -> void{
          self.create<RecordOp>(isStart, regionId);
    });

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::proton::ProtonDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });


}

