#include "ir.h"
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

using namespace mlir;
namespace py = pybind11;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

// Helper to check if an MLIR type or attribute has a verifier method.
template <typename AttrOrType>
static constexpr auto hasVerifier(AttrOrType t)
    -> decltype(t.verifyInvariants, true) {
  return true;
}
static constexpr auto hasVerifier(...) { return false; }

// Print a diagnostic without its location. The frontend will attach the AST
// location to the error message.
static void printDiagStr(llvm::raw_ostream &os, const Diagnostic &diag) {
  for (const DiagnosticArgument &arg : diag.getArguments())
    arg.print(os);
  os << "\n";
  for (const Diagnostic &note : diag.getNotes())
    printDiagStr(os, note);
}

struct GluonOpBuilder : public TritonOpBuilder {
  // Construct an attribute or type while calling its verifier. Error messages
  // are intercepted and sent back to Python via a C++ exception.
  template <typename AttrOrType, typename... ArgTs>
  std::enable_if_t<hasVerifier(AttrOrType()), AttrOrType>
  getChecked(ArgTs &&...args) {
    // Set up a scoped handler to intercept errors.
    std::string msg;
    llvm::raw_string_ostream os(msg);
    ScopedDiagnosticHandler handler(
        getContext(), [&](Diagnostic &diag) { printDiagStr(os, diag); });

    auto result =
        AttrOrType::getChecked([&] { return mlir::emitError(getLastLoc()); },
                               std::forward<ArgTs>(args)...);
    if (!result)
      throw std::runtime_error(os.str());
    return result;
  }

  // A variant of the above due to issues with C++ overload resolution and how
  // MLIR sets up the default `getChecked` implementation.
  template <typename AttrOrType, typename... ArgTs>
  std::enable_if_t<hasVerifier(AttrOrType()), AttrOrType>
  getChecked(MLIRContext *ctx, ArgTs &&...args) {
    // Set up a scoped handler to intercept errors.
    std::string msg;
    llvm::raw_string_ostream os(msg);
    ScopedDiagnosticHandler handler(
        getContext(), [&](Diagnostic &diag) { printDiagStr(os, diag); });

    if (failed(AttrOrType::verifyInvariants(
            [&] { return mlir::emitError(getLastLoc()); },
            std::forward<ArgTs>(args)...)))
      throw std::runtime_error(os.str());

    return AttrOrType::get(ctx, std::forward<ArgTs>(args)...);
  }

  // Fallback method for types or attributes that do not have a verifier.
  template <typename AttrOrType, typename... ArgTs>
  std::enable_if_t<!hasVerifier(AttrOrType()), AttrOrType>
  getChecked(ArgTs &&...args) {
    return AttrOrType::get(std::forward<ArgTs>(args)...);
  }
};

struct GluonLayouts {
  py::handle BlockedLayout;
  py::handle SliceLayout;
  py::handle DistributedLinearLayout;
  py::handle NVMMADistributedLayout;
  py::handle NVMMASharedLayout;
  py::handle SwizzledSharedLayout;

  GluonLayouts() {
    auto layouts =
        py::module::import("triton.experimental.gluon.language._layouts");
    BlockedLayout = py::object(layouts.attr("BlockedLayout")).release();
    SliceLayout = py::object(layouts.attr("SliceLayout")).release();
    DistributedLinearLayout =
        py::object(layouts.attr("DistributedLinearLayout")).release();
    NVMMADistributedLayout =
        py::object(layouts.attr("NVMMADistributedLayout")).release();
    NVMMASharedLayout = py::object(layouts.attr("NVMMASharedLayout")).release();
    SwizzledSharedLayout =
        py::object(layouts.attr("SwizzledSharedLayout")).release();
  }
};

template <typename T> std::vector<T> toStdVector(llvm::ArrayRef<T> array) {
  return std::vector<T>(array.begin(), array.end());
}

py::object layoutToGluon(Attribute layout) {
  static GluonLayouts layouts;
  if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(layout)) {
    auto ctaLayout = blocked.getCTALayout();
    return layouts.BlockedLayout(toStdVector(blocked.getSizePerThread()),
                                 toStdVector(blocked.getThreadsPerWarp()),
                                 toStdVector(blocked.getWarpsPerCTA()),
                                 toStdVector(blocked.getOrder()),
                                 toStdVector(ctaLayout.getCTAsPerCGA()),
                                 toStdVector(ctaLayout.getCTASplitNum()),
                                 toStdVector(ctaLayout.getCTAOrder()));
  } else if (auto sliced = dyn_cast<ttg::SliceEncodingAttr>(layout)) {
    return layouts.SliceLayout(sliced.getDim(),
                               layoutToGluon(sliced.getParent()));
  } else if (auto linear = dyn_cast<ttg::LinearEncodingAttr>(layout)) {
    auto ll = linear.getLinearLayout();
    auto ctx = layout.getContext();
    auto kReg = mlir::StringAttr::get(ctx, "register");
    auto kLane = mlir::StringAttr::get(ctx, "lane");
    auto kWarp = mlir::StringAttr::get(ctx, "warp");
    auto kBlock = mlir::StringAttr::get(ctx, "block");
    return layouts.DistributedLinearLayout(
        ll.getBases().lookup(kReg), ll.getBases().lookup(kLane),
        ll.getBases().lookup(kWarp), ll.getBases().lookup(kBlock),
        toStdVector(ArrayRef(llvm::to_vector(ll.getOutDimSizes()))));
  } else if (auto mma = dyn_cast<ttg::NvidiaMmaEncodingAttr>(layout)) {
    auto ctaLayout = mma.getCTALayout();
    return layouts.NVMMADistributedLayout(
        std::vector<unsigned>{mma.getVersionMajor(), mma.getVersionMinor()},
        toStdVector(mma.getWarpsPerCTA()),
        toStdVector(ctaLayout.getCTAsPerCGA()),
        toStdVector(ctaLayout.getCTASplitNum()),
        toStdVector(ctaLayout.getCTAOrder()), toStdVector(mma.getInstrShape()));
  } else if (auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(layout)) {
    auto ctaLayout = nvmma.getCTALayout();
    return layouts.NVMMASharedLayout(
        nvmma.getSwizzlingByteWidth(), nvmma.getElementBitWidth(),
        ctaLayout.getRank(), nvmma.getTransposed(), nvmma.getFp4Padded(),
        toStdVector(ctaLayout.getCTAsPerCGA()),
        toStdVector(ctaLayout.getCTASplitNum()),
        toStdVector(ctaLayout.getCTAOrder()));
  } else if (auto swizzled =
                 dyn_cast<ttg::SwizzledSharedEncodingAttr>(layout)) {
    auto ctaLayout = nvmma.getCTALayout();
    return layouts.SwizzledSharedLayout(
        swizzled.getVec(), swizzled.getPerPhase(), swizzled.getMaxPhase(),
        swizzled.getOrder(), toStdVector(ctaLayout.getCTAsPerCGA()),
        toStdVector(ctaLayout.getCTASplitNum()),
        toStdVector(ctaLayout.getCTAOrder()));
  }
  throw py::value_error("Unhandled encoding encountered");
}

void init_gluon_ir(py::module &&m) {
  using ret = py::return_value_policy;

  py::class_<GluonOpBuilder, TritonOpBuilder>(
      m, "GluonOpBuilder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *>())
      .def("get_op_builder", &GluonOpBuilder::getBuilder, ret::reference)
      .def("get_distributed_ty",
           [](GluonOpBuilder &self, Type &elementType,
              std::vector<int64_t> &shape, Attribute layout) -> Type {
             return self.getChecked<RankedTensorType>(shape, elementType,
                                                      layout);
           })
      .def("get_shared_mem_desc_ty",
           [](GluonOpBuilder &self, Type &elementType,
              std::vector<int64_t> &shape, Attribute layout,
              std::vector<int64_t> &allocShape) -> Type {
             auto ctx = self.getContext();
             return self.getChecked<ttg::MemDescType>(
                 shape, elementType, layout,
                 ttg::SharedMemorySpaceAttr::get(ctx), /*mutableMemory=*/true,
                 /*allocShape=*/allocShape);
           })
      .def("get_tensor_mem_desc_ty",
           [](GluonOpBuilder &self, Type &elementType,
              std::vector<int64_t> &shape, Attribute layout,
              std::vector<int64_t> &allocShape) -> Type {
             auto ctx = self.getContext();
             return self.getChecked<ttg::MemDescType>(
                 shape, elementType, layout,
                 ttng::TensorMemorySpaceAttr::get(ctx), /*mutableMemory=*/true,
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
             auto ctaLayout = self.getChecked<ttg::CTALayoutAttr>(
                 ctx, ctasPerCga, ctaSplitNum, ctaOrder);
             return self.getChecked<ttg::BlockedEncodingAttr>(
                 ctx, sizePerThread, threadsPerWarp, warpsPerCta, order,
                 ctaLayout);
           })
      .def("get_slice_layout",
           [](GluonOpBuilder &self, unsigned dim,
              Attribute parent) -> Attribute {
             auto ctx = self.getContext();
             auto dist = cast<ttg::DistributedEncodingTrait>(parent);
             return self.getChecked<ttg::SliceEncodingAttr>(ctx, dim, dist);
           })
      .def("get_distributed_linear_layout",
           [](GluonOpBuilder &self, std::vector<std::vector<int>> regBases,
              std::vector<std::vector<int>> laneBases,
              std::vector<std::vector<int>> warpBases,
              std::vector<std::vector<int>> blockBases,
              std::vector<int64_t> shape) -> Attribute {
             auto ctx = self.getContext();
             auto kReg = mlir::StringAttr::get(ctx, "register");
             auto kLane = mlir::StringAttr::get(ctx, "lane");
             auto kWarp = mlir::StringAttr::get(ctx, "warp");
             auto kBlock = mlir::StringAttr::get(ctx, "block");
             auto outDims = tt::standardOutDimPairs(ctx, shape);
             auto ll = tt::LinearLayout({{kReg, regBases},
                                         {kLane, laneBases},
                                         {kWarp, warpBases},
                                         {kBlock, blockBases}},
                                        outDims,
                                        /*requiresSurjective=*/true);
             return ttg::LinearEncodingAttr::get(ctx, ll);
           })
      .def("get_mma_layout",
           [](GluonOpBuilder &self, std::vector<unsigned> &version,
              std::vector<unsigned> &warpsPerCta,
              std::vector<unsigned> &ctasPerCga,
              std::vector<unsigned> &ctaSplitNum,
              std::vector<unsigned> &ctaOrder,
              std::vector<unsigned> &instrShape) -> Attribute {
             auto ctx = self.getContext();
             auto ctaLayout = self.getChecked<ttg::CTALayoutAttr>(
                 ctx, ctasPerCga, ctaSplitNum, ctaOrder);
             return self.getChecked<ttg::NvidiaMmaEncodingAttr>(
                 ctx, version[0], version[1], warpsPerCta, ctaLayout,
                 instrShape);
           })
      .def("get_nvmma_shared_layout",
           [](GluonOpBuilder &self, unsigned swizzleByteWidth,
              unsigned elementBitwidth, bool transposed, bool fp4Padded,
              std::vector<unsigned> &ctasPerCga,
              std::vector<unsigned> &ctaSplitNum,
              std::vector<unsigned> &ctaOrder) -> Attribute {
             auto ctx = self.getContext();
             auto ctaLayout = self.getChecked<ttg::CTALayoutAttr>(
                 ctx, ctasPerCga, ctaSplitNum, ctaOrder);
             return self.getChecked<ttg::NVMMASharedEncodingAttr>(
                 ctx, swizzleByteWidth, transposed, elementBitwidth, fp4Padded,
                 ctaLayout);
           })
      .def("get_swizzled_shared_layout",
           [](GluonOpBuilder &self, int vec, int perPhase, int maxPhase,
              std::vector<unsigned> &order, std::vector<unsigned> &ctasPerCga,
              std::vector<unsigned> &ctaSplitNum,
              std::vector<unsigned> &ctaOrder) -> Attribute {
             auto ctx = self.getContext();
             auto ctaLayout = self.getChecked<ttg::CTALayoutAttr>(
                 ctx, ctasPerCga, ctaSplitNum, ctaOrder);
             return self.getChecked<ttg::SwizzledSharedEncodingAttr>(
                 ctx, vec, perPhase, maxPhase, order, ctaLayout);
           })
      .def("get_tensor_memory_layout",
           [](GluonOpBuilder &self, std::vector<unsigned> &block, bool unpacked,
              std::vector<unsigned> &ctaSplitNum) -> Attribute {
             auto ctx = self.getContext();
             assert(block.size() == 2);
             assert(ctaSplitNum.size() == 2);
             return self.getChecked<ttng::TensorMemoryEncodingAttr>(
                 ctx, block[0], block[1], unpacked, ctaSplitNum[0],
                 ctaSplitNum[1]);
           })
      .def("get_gluon_layout_from_tensor",
           [](GluonOpBuilder &self, Value tensor) -> py::object {
             auto ty = dyn_cast<RankedTensorType>(tensor.getType());
             assert(ty.getEncoding());
             return layoutToGluon(ty.getEncoding());
           })
      .def("get_gluon_layout_from_memdesc",
           [](GluonOpBuilder &self, Value memdesc) -> py::object {
             auto ty = dyn_cast<ttg::MemDescType>(memdesc.getType());
             assert(ty.getEncoding());
             return layoutToGluon(ty.getEncoding());
           })
      .def("get_tensor_descriptor_layout_type",
           [](GluonOpBuilder &self, Type blockType, bool isSigned,
              Attribute layout) -> Type {
             auto ctx = self.getContext();
             auto blockTy = cast<RankedTensorType>(blockType);
             auto blockTyLayout = RankedTensorType::get(
                 blockTy.getShape(), blockTy.getElementType(), layout);
             return triton::TensorDescType::get(ctx, blockTyLayout, isSigned);
           })
      .def("create_async_copy_global_to_local",
           [](GluonOpBuilder &self, Value smem, Value pointer, Value mask,
              tt::CacheModifier cacheModifier,
              tt::EvictionPolicy evictionPolicy, bool isVolatile) {
             self.create<ttg::AsyncCopyGlobalToLocalOp>(
                 pointer, smem, mask, /*other*/ Value{}, cacheModifier,
                 evictionPolicy, isVolatile);
           })
      .def("create_async_copy_mbarrier_arrive",
           [](GluonOpBuilder &self, Value mbarrier, bool incrementCount) {
             self.create<ttng::AsyncCopyMbarrierArriveOp>(mbarrier,
                                                          !incrementCount);
           })
      .def("create_async_commit_group",
           [](GluonOpBuilder &self) {
             ValueRange tokens;
             self.create<ttg::AsyncCommitGroupOp>(tokens);
           })
      .def("create_async_wait_group",
           [](GluonOpBuilder &self, int num) {
             ValueRange tokens;
             self.create<ttg::AsyncWaitOp>(tokens, num);
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

      .def("create_memdesc_subview",
           [](GluonOpBuilder &self, Type resultType, Value src,
              std::vector<Value> &offsets) -> Value {
             return self.create<ttg::MemDescSubviewOp>(resultType, src,
                                                       offsets);
           })
      .def("create_memdesc_trans",
           [](GluonOpBuilder &self, Value src,
              std::vector<int> &order) -> Value {
             return self.create<ttg::MemDescTransOp>(src, order);
           })
      .def("create_memdesc_reshape",
           [](GluonOpBuilder &self, Type resultType, Value src) -> Value {
             return self.create<ttg::MemDescReshapeOp>(resultType, src);
           })
      .def("create_memdesc_reinterpret",
           [](GluonOpBuilder &self, Type resultType, Value src) -> Value {
             return self.create<ttg::MemDescReinterpretOp>(resultType, src);
           })
      .def("create_split",
           [](GluonOpBuilder &self, Value &a) -> py::tuple {
             auto argTy = cast<RankedTensorType>(a.getType());
             auto ctx = argTy.getContext();
             auto enc = ttg::SliceEncodingAttr::get(
                 ctx, argTy.getRank() - 1,
                 cast<ttg::DistributedEncodingTrait>(argTy.getEncoding()));
             auto resTy =
                 RankedTensorType::get(ArrayRef(argTy.getShape()).drop_back(),
                                       argTy.getElementType(), enc);
             auto op = self.create<triton::SplitOp>(TypeRange{resTy, resTy}, a);
             return py::make_tuple(op->getResult(0), op->getResult(1));
           })
      .def("create_warpgroup_mma",
           [](GluonOpBuilder &self, Value a, Value b, Value acc, Value useAcc,
              triton::InputPrecision precision = triton::InputPrecision::IEEE,
              int maxNumImpreciseAcc = 0, bool isAsync = false) -> Value {
             return self.create<ttng::WarpGroupDotOp>(
                 a, b, acc, useAcc, precision, maxNumImpreciseAcc, isAsync);
           })
      .def("create_warpgroup_mma_wait",
           [](GluonOpBuilder &self, std::vector<Value> &deps, int pendings) {
             self.create<ttng::WarpGroupDotWaitOp>(deps, pendings);
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
      .def("create_tcgen05_commit",
           [](GluonOpBuilder &self, Value &barrier) {
             self.create<ttng::TCGen5CommitOp>(barrier);
           })

      .def("create_async_tma_copy_global_to_local",
           [](GluonOpBuilder &self, Value descPtr, std::vector<Value> &coord,
              Value barrier, Value result, Value pred) {
             self.create<ttng::AsyncTMACopyGlobalToLocalOp>(
                 descPtr, coord, barrier, result, pred);
           })
      .def("create_async_tma_copy_local_to_global",
           [](GluonOpBuilder &self, Value descPtr, std::vector<Value> &coord,
              Value src) {
             self.create<ttng::AsyncTMACopyLocalToGlobalOp>(descPtr, coord,
                                                            src);
           })
      .def("create_async_tma_reduce",
           [](GluonOpBuilder &self, triton::DescriptorReduceKind kind,
              Value descPtr, std::vector<Value> &coord, Value src) {
             self.create<ttng::AsyncTMAReduceOp>(kind, descPtr, coord, src);
           })
      .def("create_async_tma_store_wait",
           [](GluonOpBuilder &self, int pendings) {
             self.create<ttng::TMAStoreWaitOp>(pendings);
           })
      .def("create_async_tma_gather",
           [](GluonOpBuilder &self, Value descPtr, Value xOffsets,
              Value yOffset, Value barrier, Value result, Value pred) {
             self.create<ttng::AsyncTMAGatherOp>(descPtr, xOffsets, yOffset,
                                                 barrier, result, pred);
           })
      .def("create_async_tma_scatter",
           [](GluonOpBuilder &self, Value descPtr, Value xOffsets,
              Value yOffset, Value src) {
             self.create<ttng::AsyncTMAScatterOp>(descPtr, xOffsets, yOffset,
                                                  src);
           })
      .def("create_fence_async_shared",
           [](GluonOpBuilder &self, bool bCluster) -> OpState {
             return self.create<ttng::FenceAsyncSharedOp>(bCluster);
           })

      .def("create_broadcast",
           [](TritonOpBuilder &self, Value &arg, Type retTy) -> Value {
             return self.create<tt::BroadcastOp>(retTy, arg);
           })
      .def(
          "create_expand_dims",
          [](TritonOpBuilder &self, Value &arg, int axis, Type retTy) -> Value {
            return self.create<tt::ExpandDimsOp>(retTy, arg, axis);
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
