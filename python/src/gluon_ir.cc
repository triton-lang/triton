#include "ir.h"
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

using namespace mlir;
namespace py = pybind11;
namespace ttg = triton::gpu;

struct GluonOpBuilder : public TritonOpBuilder {};

void init_gluon_ir(py::module &&m) {
  py::class_<GluonOpBuilder, TritonOpBuilder>(
      m, "GluonOpBuilder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *>())
      .def("get_distributed_ty",
           [](GluonOpBuilder &self, Type &elementType,
              std::vector<int64_t> &shape, Attribute layout) -> Type {
             return RankedTensorType::get(shape, elementType, layout);
           })
      .def("get_blocked_layout",
           [](GluonOpBuilder &self, std::vector<unsigned> &sizePerThread,
              std::vector<unsigned> &threadsPerWarp,
              std::vector<unsigned> &warpsPerCta, std::vector<unsigned> &order,
              std::vector<unsigned> &ctasPerCga,
              std::vector<unsigned> &ctaSplitNum,
              std::vector<unsigned> &ctaOrder) -> Attribute {
             auto ctx = self.getContext();
             auto ctaLayout = ttg::CTALayoutAttr::get(ctx, ctasPerCga,
                                                      ctaSplitNum, ctaOrder);
             return ttg::BlockedEncodingAttr::get(ctx, sizePerThread,
                                                  threadsPerWarp, warpsPerCta,
                                                  order, ctaLayout);
           });
}
