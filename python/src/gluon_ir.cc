#include "ir.h"
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
namespace py = pybind11;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

struct GluonOpBuilder : public TritonOpBuilder {};

void init_gluon_ir(py::module &&m) {
  using ret = py::return_value_policy;

  py::class_<GluonOpBuilder, TritonOpBuilder>(
      m, "GluonOpBuilder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *>())
      .def("get_distributed_ty",
           [](GluonOpBuilder &self, Type &elementType,
              std::vector<int64_t> &shape, Attribute layout) -> Type {
             return RankedTensorType::get(shape, elementType, layout);
           })
      .def("get_shared_mem_desc_ty",
           [](GluonOpBuilder &self, Type &elementType,
              std::vector<int64_t> &shape, Attribute layout,
              std::vector<int64_t> &allocShape) -> Type {
             auto ctx = self.getContext();
             return ttg::MemDescType::get(shape, elementType, layout,
                                          ttg::SharedMemorySpaceAttr::get(ctx),
                                          /*mutableMemory=*/true,
                                          /*allocShape=*/allocShape);
           })
      .def("get_tensor_mem_desc_ty",
           [](GluonOpBuilder &self, Type &elementType,
              std::vector<int64_t> &shape, Attribute layout,
              std::vector<int64_t> &allocShape) -> Type {
             auto ctx = self.getContext();
             return ttg::MemDescType::get(shape, elementType, layout,
                                          ttng::TensorMemorySpaceAttr::get(ctx),
                                          /*mutableMemory=*/true,
                                          /*allocShape=*/allocShape);
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
           })
      .def("get_slice_layout",
           [](GluonOpBuilder &self, unsigned dim,
              Attribute parent) -> Attribute {
             auto ctx = self.getContext();
             auto dist = cast<ttg::DistributedEncodingTrait>(parent);
             return ttg::SliceEncodingAttr::get(ctx, dim, dist);
           })
      .def("get_nvmma_shared_layout",
           [](GluonOpBuilder &self, unsigned swizzleByteWidth,
              unsigned elementBitwidth, bool transposed, bool fp4Padded,
              std::vector<unsigned> &ctasPerCga,
              std::vector<unsigned> &ctaSplitNum,
              std::vector<unsigned> &ctaOrder) -> Attribute {
             auto ctx = self.getContext();
             auto ctaLayout = ttg::CTALayoutAttr::get(ctx, ctasPerCga,
                                                      ctaSplitNum, ctaOrder);
             return ttg::NVMMASharedEncodingAttr::get(
                 ctx, swizzleByteWidth, transposed, elementBitwidth, fp4Padded,
                 ctaLayout);
           })
      .def("get_swizzled_shared_layout",
           [](GluonOpBuilder &self, int vec, int perPhase, int maxPhase,
              std::vector<unsigned> &order, std::vector<unsigned> &ctasPerCga,
              std::vector<unsigned> &ctaSplitNum,
              std::vector<unsigned> &ctaOrder) -> Attribute {
             auto ctx = self.getContext();
             auto ctaLayout = ttg::CTALayoutAttr::get(ctx, ctasPerCga,
                                                      ctaSplitNum, ctaOrder);
             return ttg::SwizzledSharedEncodingAttr::get(
                 ctx, vec, perPhase, maxPhase, order, ctaLayout);
           })
      .def("get_tensor_memory_layout",
           [](GluonOpBuilder &self, std::vector<unsigned> &block, bool unpacked,
              std::vector<unsigned> &ctaSplitNum) -> Attribute {
             auto ctx = self.getContext();
             assert(block.size() == 2);
             assert(ctaSplitNum.size() == 2);
             return ttng::TensorMemoryEncodingAttr::get(
                 ctx, block[0], block[1], unpacked, ctaSplitNum[0],
                 ctaSplitNum[1]);
           })
      .def("create_convert_layout",
           [](GluonOpBuilder &self, Type resultTy, Value value) -> Value {
             return self.create<ttg::ConvertLayoutOp>(resultTy, value);
           })
      .def("create_local_alloc",
           [](GluonOpBuilder &self, Type resultTy) -> Value {
             return self.create<ttg::LocalAllocOp>(resultTy);
           })
      .def("create_local_alloc",
           [](GluonOpBuilder &self, Type resultTy, Value value) -> Value {
             return self.create<ttg::LocalAllocOp>(resultTy, value);
           })
      .def("create_local_store",
           [](GluonOpBuilder &self, Value memDesc, Value value) {
             self.create<ttg::LocalStoreOp>(value, memDesc);
           })
      .def("create_local_load",
           [](GluonOpBuilder &self, Type resultTy, Value memDesc) -> Value {
             return self.create<ttg::LocalLoadOp>(resultTy, memDesc);
           })
      .def("create_local_dealloc",
           [](GluonOpBuilder &self, Value memDesc) -> Operation * {
             return self.create<ttg::LocalDeallocOp>(memDesc);
           })

      .def("create_tmem_alloc",
           [](GluonOpBuilder &self, Type resultTy, Value value) -> Value {
             return self.create<ttng::TMEMAllocOp>(resultTy, value);
           })
      .def("create_tmem_alloc",
           [](GluonOpBuilder &self, Type resultTy, py::none value) -> Value {
             return self.create<ttng::TMEMAllocOp>(resultTy, Value{});
           })
      .def("create_tmem_store",
           [](GluonOpBuilder &self, Value memDesc, Value value, Value pred) {
             self.create<ttng::TMEMStoreOp>(memDesc, value, pred);
           })
      .def("create_tmem_load",
           [](GluonOpBuilder &self, Type resultTy, Value memDesc) -> Value {
             return self.create<ttng::TMEMLoadOp>(resultTy, memDesc);
           })
      .def("create_tmem_subslice",
           [](GluonOpBuilder &self, Type resultTy, Value memDesc,
              int N) -> Value {
             return self.create<ttng::TMEMSubSliceOp>(resultTy, memDesc, N);
           })
      .def("create_mbarrier_init",
           [](GluonOpBuilder &self, Value memDesc, int count) {
             self.create<ttng::InitBarrierOp>(memDesc, count);
           })
      .def("create_mbarrier_inval",
           [](GluonOpBuilder &self, Value memDesc) {
             self.create<ttng::InvalBarrierOp>(memDesc);
           })
      .def("create_mbarrier_expect",
           [](GluonOpBuilder &self, Value memDesc, int bytes, Value pred) {
             self.create<ttng::BarrierExpectOp>(memDesc, bytes, pred);
           })
      .def("create_mbarrier_wait",
           [](GluonOpBuilder &self, Value memDesc, Value phase, Value pred,
              std::vector<Value> &deps) {
             self.create<ttng::WaitBarrierOp>(memDesc, phase, pred, deps);
           })
      .def("create_mbarrier_arrive",
           [](GluonOpBuilder &self, Value memDesc, int count, Value pred) {
             self.create<ttng::ArriveBarrierOp>(memDesc, count, pred);
           })
      .def("create_tcgen05_mma",
           [](GluonOpBuilder &self, Value a, Value b, Value acc, Value useAcc,
              Value pred, std::vector<Value> &mbarriers,
              std::vector<Value> &mbarrier_preds) {
             Value accDep;
             bool two_ctas = false;
             auto tokType = self.getBuilder().getType<ttg::AsyncTokenType>();
             self.create<ttng::TCGen5MMAOp>(tokType, a, b, acc, accDep, useAcc,
                                            pred, two_ctas, mbarriers,
                                            mbarrier_preds);
           })
      .def("create_warp_return",
           [](GluonOpBuilder &self) -> Operation * {
             return self.create<ttg::WarpReturnOp>();
           })
      .def("create_warp_yield",
           [](GluonOpBuilder &self, std::vector<Value> &values) -> Operation * {
             return self.create<ttg::WarpYieldOp>(values);
           })
      .def("create_warp_specialize_partitions",
           [](GluonOpBuilder &self, int numPartitions) -> Operation * {
             return self.create<ttg::WarpSpecializePartitionsOp>(numPartitions);
           })
      .def("create_warp_specialize", [](GluonOpBuilder &self,
                                        std::vector<Type> &resultTypes,
                                        std::vector<Value> &explicitCaptures,
                                        std::vector<int> &partitionNumWarps) {
        return self.create<ttg::WarpSpecializeOp>(resultTypes, explicitCaptures,
                                                  partitionNumWarps);
      });

  py::class_<ttg::WarpSpecializeOp, OpState>(m, "WarpSpecializeOp",
                                             py::module_local())
      .def("get_default_region", &ttg::WarpSpecializeOp::getDefaultRegion,
           ret::reference)
      .def("get_partition_op_holder",
           &ttg::WarpSpecializeOp::getPartitionOpHolder, ret::reference)
      .def("set_requested_registers", [](ttg::WarpSpecializeOp &self,
                                         std::vector<int> &requestedRegisters) {
        self.setRequestedRegisters(requestedRegisters);
      });
}
