#include "ir.h"
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Types.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace py = pybind11;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
namespace gluon = mlir::triton::gluon;
namespace ttag = mlir::triton::amdgpu;

static ttg::CGAEncodingAttr
buildCgaLayoutAttr(MLIRContext *ctx,
                   const std::vector<std::vector<int32_t>> &layout,
                   unsigned rank) {
  auto kBlock = StringAttr::get(ctx, "block");
  tt::LinearLayout::BasesT bases;
  bases[kBlock] = layout;
  auto outDims = tt::standardOutDimNames(ctx, rank);
  tt::LinearLayout ll(std::move(bases), outDims);
  return ttg::CGAEncodingAttr::get(ctx, std::move(ll));
}

static std::vector<std::vector<int32_t>>
getCgaLayoutBases(ttg::CGAEncodingAttr layout) {
  std::vector<std::vector<int32_t>> result;
  auto ctx = layout.getContext();
  auto block = StringAttr::get(ctx, "block");
  const auto &basesMap = layout.getLinearLayout().getBases();
  auto it = basesMap.find(block);
  assert(it != basesMap.end());
  return it->second;
}

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
  using TritonOpBuilder::TritonOpBuilder;
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
            [&] { return mlir::emitError(getLastLoc()); }, args...)))
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
  py::handle AutoLayout;
  py::handle CoalescedLayout;
  py::handle BlockedLayout;
  py::handle SliceLayout;
  py::handle DistributedLinearLayout;
  py::handle DotOperandLayout;
  py::handle NVMMADistributedLayout;
  py::handle TensorMemoryScalesLayout;
  py::handle TensorMemoryLayout;
  py::handle NVMMASharedLayout;
  py::handle SwizzledSharedLayout;
  py::handle SharedLinearLayout;
  py::handle AMDMFMALayout;
  py::handle AMDWMMALayout;
  py::handle PaddedSharedLayout;

  GluonLayouts() {
    auto layouts =
        py::module::import("triton.experimental.gluon.language._layouts");
    auto amdLayouts =
        py::module::import("triton.experimental.gluon.language.amd._layouts");
    auto blackwellLayouts = py::module::import(
        "triton.experimental.gluon.language.nvidia.blackwell");
    AutoLayout = py::object(layouts.attr("AutoLayout")).release();
    CoalescedLayout = py::object(layouts.attr("CoalescedLayout")).release();
    BlockedLayout = py::object(layouts.attr("BlockedLayout")).release();
    SliceLayout = py::object(layouts.attr("SliceLayout")).release();
    DistributedLinearLayout =
        py::object(layouts.attr("DistributedLinearLayout")).release();
    DotOperandLayout = py::object(layouts.attr("DotOperandLayout")).release();
    NVMMADistributedLayout =
        py::object(layouts.attr("NVMMADistributedLayout")).release();
    TensorMemoryScalesLayout =
        py::object(blackwellLayouts.attr("TensorMemoryScalesLayout")).release();
    TensorMemoryLayout =
        py::object(blackwellLayouts.attr("TensorMemoryLayout")).release();
    NVMMASharedLayout = py::object(layouts.attr("NVMMASharedLayout")).release();
    SwizzledSharedLayout =
        py::object(layouts.attr("SwizzledSharedLayout")).release();
    SharedLinearLayout =
        py::object(layouts.attr("SharedLinearLayout")).release();
    AMDMFMALayout = py::object(amdLayouts.attr("AMDMFMALayout")).release();
    AMDWMMALayout = py::object(amdLayouts.attr("AMDWMMALayout")).release();
    PaddedSharedLayout =
        py::object(layouts.attr("PaddedSharedLayout")).release();

    auto core = py::module::import("triton.language.core");
  }
};

static bool isConvertLayoutTrivial(RankedTensorType dstTy, Value value) {
  auto srcTy = cast<RankedTensorType>(value.getType());
  if (srcTy.getEncoding() == dstTy.getEncoding())
    return true;
  // Fail safe on unresolved layouts.
  if (isa<gluon::AutoEncodingAttr>(srcTy.getEncoding()))
    return false;
  if (isa<gluon::AutoEncodingAttr>(dstTy.getEncoding()))
    return false;

  // Check concrete layouts.
  triton::LinearLayout cvt = minimalCvtLayout(srcTy, dstTy);
  auto dims = llvm::to_vector(cvt.getInDimNames());
  return dims.empty() || (dims.size() == 1 && dims.front() == "register");
}

template <typename R>
std::vector<llvm::ValueTypeFromRangeType<R>> toStdVector(R &&range) {
  return {range.begin(), range.end()};
}

py::object layoutToGluon(Attribute layout) {
  static GluonLayouts layouts;
  if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(layout)) {
    auto cgaBases = getCgaLayoutBases(blocked.getCGALayout());
    return layouts.BlockedLayout(toStdVector(blocked.getSizePerThread()),
                                 toStdVector(blocked.getThreadsPerWarp()),
                                 toStdVector(blocked.getWarpsPerCTA()),
                                 toStdVector(blocked.getOrder()), cgaBases);
  } else if (auto sliced = dyn_cast<ttg::SliceEncodingAttr>(layout)) {
    return layouts.SliceLayout(sliced.getDim(),
                               layoutToGluon(sliced.getParent()));
  } else if (auto linear = dyn_cast<ttg::LinearEncodingAttr>(layout)) {
    const auto &ll = linear.getLinearLayout();
    auto ctx = layout.getContext();
    auto kReg = mlir::StringAttr::get(ctx, "register");
    auto kLane = mlir::StringAttr::get(ctx, "lane");
    auto kWarp = mlir::StringAttr::get(ctx, "warp");
    auto kBlock = mlir::StringAttr::get(ctx, "block");
    return layouts.DistributedLinearLayout(
        ll.getBases().lookup(kReg), ll.getBases().lookup(kLane),
        ll.getBases().lookup(kWarp), ll.getBases().lookup(kBlock),
        toStdVector(ll.getOutDimSizes()));
  } else if (auto dotOp = dyn_cast<ttg::DotOperandEncodingAttr>(layout)) {
    return layouts.DotOperandLayout(
        dotOp.getOpIdx(), layoutToGluon(dotOp.getParent()), dotOp.getKWidth());
  } else if (auto mma = dyn_cast<ttg::NvidiaMmaEncodingAttr>(layout)) {
    auto cgaBases = getCgaLayoutBases(mma.getCGALayout());
    return layouts.NVMMADistributedLayout(
        std::vector<unsigned>{mma.getVersionMajor(), mma.getVersionMinor()},
        toStdVector(mma.getWarpsPerCTA()), toStdVector(mma.getInstrShape()),
        cgaBases);
  } else if (auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(layout)) {
    auto cgaLayout = nvmma.getCGALayout();
    auto cgaBases = getCgaLayoutBases(cgaLayout);
    return layouts.NVMMASharedLayout(nvmma.getSwizzlingByteWidth(),
                                     nvmma.getElementBitWidth(),
                                     cgaLayout.getRank(), nvmma.getTransposed(),
                                     nvmma.getFp4Padded(), cgaBases);
  } else if (auto swizzled =
                 dyn_cast<ttg::SwizzledSharedEncodingAttr>(layout)) {
    auto cgaBases = getCgaLayoutBases(swizzled.getCGALayout());
    return layouts.SwizzledSharedLayout(
        swizzled.getVec(), swizzled.getPerPhase(), swizzled.getMaxPhase(),
        toStdVector(swizzled.getOrder()), cgaBases);
  } else if (auto sharedLl = dyn_cast<ttg::SharedLinearEncodingAttr>(layout)) {
    const auto &ll = sharedLl.getLinearLayout();
    auto ctx = layout.getContext();
    auto kOffset = mlir::StringAttr::get(ctx, "offset");
    auto kBlock = mlir::StringAttr::get(ctx, "block");
    return layouts.SharedLinearLayout(
        toStdVector(ll.getBases().lookup(kOffset)),
        toStdVector(ll.getBases().lookup(kBlock)), sharedLl.getAlignment());
  } else if (auto autoEnc = dyn_cast<gluon::AutoEncodingAttr>(layout)) {
    return layouts.AutoLayout();
  } else if (auto autoEnc = dyn_cast<gluon::CoalescedEncodingAttr>(layout)) {
    return layouts.CoalescedLayout();
  } else if (auto amdMfma = dyn_cast<ttg::AMDMfmaEncodingAttr>(layout)) {
    auto cgaBases = getCgaLayoutBases(amdMfma.getCGALayout());
    return layouts.AMDMFMALayout(
        amdMfma.getVersion(), toStdVector(amdMfma.getInstrShape()),
        amdMfma.getIsTransposed(), toStdVector(amdMfma.getWarpsPerCTA()),
        amdMfma.getElementBitWidth(), toStdVector(amdMfma.getTilesPerWarp()),
        cgaBases);
  } else if (auto amdWmma = dyn_cast<ttg::AMDWmmaEncodingAttr>(layout)) {
    auto cgaBases = getCgaLayoutBases(amdWmma.getCGALayout());
    return layouts.AMDWMMALayout(
        amdWmma.getVersion(), amdWmma.getIsTransposed(),
        toStdVector(amdWmma.getWarpsPerCTA()),
        toStdVector(amdWmma.getInstrShape()),
        toStdVector(amdWmma.getTilesPerWarp()), cgaBases);
  } else if (auto paddedShared =
                 dyn_cast<ttg::PaddedSharedEncodingAttr>(layout)) {
    auto *ctx = paddedShared.getContext();
    std::vector<std::pair<unsigned, unsigned>> intervalPaddingPairs;
    for (auto [interval, padding] :
         llvm::zip(paddedShared.getIntervals(), paddedShared.getPaddings())) {
      intervalPaddingPairs.push_back({interval, padding});
    }
    auto kOffset = mlir::StringAttr::get(ctx, "offset");
    auto kBlock = mlir::StringAttr::get(ctx, "block");
    const auto &ll = paddedShared.getLinearComponent();
    auto shape = toStdVector(ll.getOutDimSizes());
    return layouts.PaddedSharedLayout(intervalPaddingPairs,
                                      ll.getBases().lookup(kOffset),
                                      ll.getBases().lookup(kBlock), shape);
  } else if (auto tmemScales =
                 dyn_cast<ttng::TensorMemoryScalesEncodingAttr>(layout)) {
    return layouts.TensorMemoryScalesLayout(std::vector<unsigned>{
        tmemScales.getCTASplitM(), tmemScales.getCTASplitN()});
  } else if (auto tmem = dyn_cast<ttng::TensorMemoryEncodingAttr>(layout)) {
    return layouts.TensorMemoryLayout(
        std::vector<unsigned>{tmem.getBlockM(), tmem.getBlockN()},
        tmem.getColStride(),
        std::vector<unsigned>{tmem.getCTASplitM(), tmem.getCTASplitN()});
  }

  throw py::value_error("Unhandled encoding encountered");
}

template <typename CondT> static void check(CondT &&cond, const char *msg) {
  if (!std::forward<CondT>(cond))
    throw py::value_error(msg);
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
                 ttg::SharedMemorySpaceAttr::get(ctx),
                 /*mutableMemory=*/true,
                 /*allocShape=*/allocShape);
           })
      .def("get_tensor_mem_desc_ty",
           [](GluonOpBuilder &self, Type &elementType,
              std::vector<int64_t> &shape, Attribute layout,
              std::vector<int64_t> &allocShape) -> Type {
             auto ctx = self.getContext();
             return self.getChecked<ttg::MemDescType>(
                 shape, elementType, layout,
                 ttng::TensorMemorySpaceAttr::get(ctx),
                 /*mutableMemory=*/true,
                 /*allocShape=*/allocShape);
           })
      .def("get_blocked_layout",
           [](GluonOpBuilder &self, std::vector<unsigned> &sizePerThread,
              std::vector<unsigned> &threadsPerWarp,
              std::vector<unsigned> &warpsPerCta, std::vector<unsigned> &order,
              std::vector<std::vector<int32_t>> &cgaBases) -> Attribute {
             auto ctx = self.getContext();
             unsigned rank = order.size();
             auto cgaLayout = buildCgaLayoutAttr(ctx, cgaBases, rank);
             return self.getChecked<ttg::BlockedEncodingAttr>(
                 ctx, sizePerThread, threadsPerWarp, warpsPerCta, order,
                 cgaLayout);
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
             return ttg::LinearEncodingAttr::get(ctx, std::move(ll));
           })
      .def("to_linear_layout",
           [](GluonOpBuilder &self, Attribute layout,
              std::vector<int64_t> &shape) -> py::object {
             auto ctx = self.getContext();
             auto linearLayout = ttg::toLinearLayout(shape, layout);

             if (isa<ttg::DistributedEncodingTrait>(layout)) {
               auto attr =
                   ttg::LinearEncodingAttr::get(ctx, std::move(linearLayout));
               return layoutToGluon(attr);
             }
             if (isa<ttg::SharedEncodingTrait>(layout)) {
               auto alignment =
                   cast<ttg::SharedEncodingTrait>(layout).getAlignment();
               auto attr = ttg::SharedLinearEncodingAttr::get(
                   ctx, std::move(linearLayout), alignment);
               return layoutToGluon(attr);
             }

             // TensorMemory encodings: keep the LinearLayout but wrap as
             // print-only Python object carrying row/col bases -> dim0/dim1.
             auto inNamesRange = linearLayout.getInDimNames();
             auto inNames = llvm::to_vector(inNamesRange);
             bool isTmemLayout =
                 (inNames.size() == 2 && inNames[0].str() == "row" &&
                  inNames[1].str() == "col");
             if (!isTmemLayout)
               throw std::invalid_argument(
                   "Unsupported layout in to_linear_layout");

             // Build Py _TensorMemoryLinearLayout(row_bases, col_bases, shape,
             // repr)
             py::object tmemCls =
                 py::module::import(
                     "triton.experimental.gluon.language.nvidia.blackwell")
                     .attr("_TensorMemoryLinearLayout");
             auto bases = linearLayout.getBases();
             auto rowBases = bases[mlir::StringAttr::get(ctx, "row")];
             auto colBases = bases[mlir::StringAttr::get(ctx, "col")];
             auto outDims = linearLayout.getOutDims();
             std::vector<int> shapeVec;
             for (auto &od : outDims)
               shapeVec.push_back(od.second);

             py::object pyObj = tmemCls(py::cast(rowBases), py::cast(colBases),
                                        py::cast(shapeVec));
             return pyObj;
           })
      .def("get_dot_operand_layout",
           [](GluonOpBuilder &self, unsigned opIdx, Attribute parent,
              unsigned kWidth) -> Attribute {
             return self.getChecked<ttg::DotOperandEncodingAttr>(
                 self.getContext(), opIdx, parent, kWidth);
           })
      .def("get_mma_layout",
           [](GluonOpBuilder &self, std::vector<unsigned> &version,
              std::vector<unsigned> &warpsPerCta,
              std::vector<std::vector<int32_t>> &cgaBases,
              std::vector<unsigned> &instrShape) -> Attribute {
             auto ctx = self.getContext();
             unsigned rank = warpsPerCta.size();
             auto cgaLayout = buildCgaLayoutAttr(ctx, cgaBases, rank);
             return self.getChecked<ttg::NvidiaMmaEncodingAttr>(
                 ctx, version[0], version[1], warpsPerCta, cgaLayout,
                 instrShape);
           })
      .def("get_amd_mfma_layout",
           [](GluonOpBuilder &self, unsigned version,
              std::vector<unsigned> &warpsPerCta,
              std::vector<unsigned> &instrShape, bool transposed,
              std::vector<std::vector<int32_t>> &cgaBases,
              std::vector<unsigned> &tilesPerWarp,
              unsigned elementBitWidth) -> Attribute {
             auto ctx = self.getContext();
             unsigned rank = warpsPerCta.size();
             auto cgaLayout = buildCgaLayoutAttr(ctx, cgaBases, rank);
             return ttg::AMDMfmaEncodingAttr::get(
                 ctx, version, warpsPerCta, instrShape, transposed, cgaLayout,
                 tilesPerWarp, elementBitWidth);
           })
      .def("get_amd_wmma_layout",
           [](GluonOpBuilder &self, unsigned version, bool transposed,
              std::vector<unsigned> &warpsPerCta,
              std::vector<unsigned> &tilesPerWarp,
              std::vector<std::vector<int32_t>> &cgaBases,
              std::vector<unsigned> &instrShape) -> Attribute {
             auto ctx = self.getContext();
             unsigned rank = warpsPerCta.size();
             auto cgaLayout = buildCgaLayoutAttr(ctx, cgaBases, rank);
             return ttg::AMDWmmaEncodingAttr::get(ctx, version, transposed,
                                                  warpsPerCta, tilesPerWarp,
                                                  cgaLayout, instrShape);
           })
      .def("get_padded_shared_layout",
           [](GluonOpBuilder &self, std::vector<unsigned> &intervals,
              std::vector<unsigned> &paddings,
              std::vector<std::vector<int>> &offsetBases,
              std::vector<std::vector<int>> &blockBases,
              std::vector<int64_t> &shape) -> Attribute {
             auto ctx = self.getContext();
             auto rank = shape.size();
             auto kOffset = mlir::StringAttr::get(ctx, "offset");
             auto kBlock = mlir::StringAttr::get(ctx, "block");
             auto ll = tt::LinearLayout(
                 {{kOffset, offsetBases}, {kBlock, blockBases}},
                 tt::standardOutDimNames(ctx, rank));
             return ttg::PaddedSharedEncodingAttr::get(ctx, intervals, paddings,
                                                       std::move(ll));
           })
      .def("get_shared_linear_layout",
           [](GluonOpBuilder &self, std::vector<std::vector<int>> &offsetBases,
              std::vector<std::vector<int>> &blockBases,
              unsigned alignment) -> Attribute {
             auto ctx = self.getContext();
             auto kOffset = mlir::StringAttr::get(ctx, "offset");
             auto kBlock = mlir::StringAttr::get(ctx, "block");
             auto outDims = tt::standardOutDimNames(ctx, offsetBases[0].size());
             auto ll = tt::LinearLayout(
                 {{kOffset, offsetBases}, {kBlock, blockBases}}, outDims);
             return self.getChecked<ttg::SharedLinearEncodingAttr>(
                 ctx, std::move(ll), alignment);
           })
      .def("get_nvmma_shared_layout",
           [](GluonOpBuilder &self, unsigned swizzleByteWidth,
              unsigned elementBitwidth, bool transposed, bool fp4Padded,
              std::vector<std::vector<int32_t>> &cgaBases,
              unsigned rank) -> Attribute {
             auto ctx = self.getContext();
             auto cgaLayout = buildCgaLayoutAttr(ctx, cgaBases, rank);
             return self.getChecked<ttg::NVMMASharedEncodingAttr>(
                 ctx, swizzleByteWidth, transposed, elementBitwidth, fp4Padded,
                 cgaLayout);
           })
      .def("get_auto_layout",
           [](GluonOpBuilder &self) -> Attribute {
             return self.getChecked<gluon::AutoEncodingAttr>(self.getContext());
           })
      .def("get_coalesced_layout",
           [](GluonOpBuilder &self) -> Attribute {
             return self.getChecked<gluon::CoalescedEncodingAttr>(
                 self.getContext());
           })
      .def("get_swizzled_shared_layout",
           [](GluonOpBuilder &self, int vec, int perPhase, int maxPhase,
              std::vector<unsigned> &order,
              std::vector<std::vector<int32_t>> &cgaBases) -> Attribute {
             auto ctx = self.getContext();
             unsigned rank = order.size();
             auto cgaLayout = buildCgaLayoutAttr(ctx, cgaBases, rank);
             return self.getChecked<ttg::SwizzledSharedEncodingAttr>(
                 ctx, vec, perPhase, maxPhase, order, cgaLayout);
           })
      .def("get_tensor_memory_layout",
           [](GluonOpBuilder &self, std::vector<unsigned> &block,
              unsigned colStride, std::vector<unsigned> &ctaSplitNum,
              bool twoCTAs) -> Attribute {
             auto ctx = self.getContext();
             check(block.size() == 2, "expected a 2D block");
             check(ctaSplitNum.size() == 2, "expected 2D CTA dimensions");
             return self.getChecked<ttng::TensorMemoryEncodingAttr>(
                 ctx, block[0], block[1], colStride, ctaSplitNum[0],
                 ctaSplitNum[1], twoCTAs);
           })
      .def("get_tensor_memory_scales_layout",
           [](GluonOpBuilder &self,
              std::vector<unsigned> &ctaSplitNum) -> Attribute {
             auto ctx = self.getContext();
             check(ctaSplitNum.size() == 2, "expected 2D CTA dimensions");
             return self.getChecked<ttng::TensorMemoryScalesEncodingAttr>(
                 ctx, ctaSplitNum[0], ctaSplitNum[1]);
           })
      .def("get_gluon_layout_from_tensor",
           [](GluonOpBuilder &self, Value tensor) -> py::object {
             auto ty = dyn_cast<RankedTensorType>(tensor.getType());
             check(ty.getEncoding(), "expected a tensor with an encoding");
             return layoutToGluon(ty.getEncoding());
           })
      .def("get_gluon_layout_from_memdesc",
           [](GluonOpBuilder &self, Value memdesc) -> py::object {
             auto ty = dyn_cast<ttg::MemDescType>(memdesc.getType());
             check(ty.getEncoding(), "expected a memdesc with an encoding");
             return layoutToGluon(ty.getEncoding());
           })
      .def("get_tensor_descriptor_layout_type",
           [](GluonOpBuilder &self, Type blockType, bool isSigned,
              Attribute layout) -> Type {
             auto ctx = self.getContext();
             auto blockTy = cast<RankedTensorType>(blockType);
             auto blockTyLayout = blockTy.cloneWithEncoding(layout);
             return triton::TensorDescType::get(ctx, blockTyLayout, isSigned);
           })
      .def("is_convert_layout_trivial",
           [](GluonOpBuilder &self, Type resultTy, Value value) -> bool {
             auto dstTy = cast<RankedTensorType>(resultTy);
             return isConvertLayoutTrivial(dstTy, value);
           })
      .def("create_histogram",
           [](GluonOpBuilder &self, Value operand, int numBins,
              std::optional<Value> mask, Attribute layout) -> Value {
             auto *ctx = self.getContext();
             auto resultTy =
                 RankedTensorType::get({static_cast<int64_t>(numBins)},
                                       IntegerType::get(ctx, 32), layout);
             if (!mask) {
               return self.create<triton::HistogramOp>(resultTy, operand);
             } else {
               return self.create<triton::HistogramOp>(resultTy, operand,
                                                       *mask);
             }
           })
      .def("create_cat",
           [](GluonOpBuilder &self, Value &lhs, Value &rhs,
              Type retType) -> Value {
             return self.create<triton::CatOp>(retType, lhs, rhs);
           })
      .def("create_fp4_to_fp",
           [](GluonOpBuilder &self, Value src, Type elemType,
              int axis) -> Value {
             return self.create<ttg::Fp4ToFpOp>(
                 cast<TypedValue<RankedTensorType>>(src), elemType, axis);
           })
      .def("create_async_copy_global_to_local",
           [](GluonOpBuilder &self, Value smem, Value pointer, Value mask,
              Value other, tt::CacheModifier cacheModifier,
              tt::EvictionPolicy evictionPolicy, bool isVolatile) {
             self.create<ttg::AsyncCopyGlobalToLocalOp>(
                 pointer, smem, mask, other, cacheModifier, evictionPolicy,
                 isVolatile);
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
      .def("get_shared_bank_conflicts",
           [](GluonOpBuilder &self, Attribute regLayoutAttr,
              Attribute sharedLayoutAttr, std::vector<int64_t> &shape,
              int bitwidth) -> int {
             auto regLayout = ttg::toLinearLayout(shape, regLayoutAttr);
             auto smemLayout = ttg::toLinearLayout(shape, sharedLayoutAttr);
             return ttg::bankConflictsMemDesc(regLayout, smemLayout, bitwidth);
           })
      .def("create_local_dealloc",
           [](GluonOpBuilder &self, Value memDesc) -> Operation * {
             return self.create<ttg::LocalDeallocOp>(memDesc);
           })

      .def("create_memdesc_index",
           [](GluonOpBuilder &self, Type resultType, Value src,
              Value index) -> Value {
             return self.create<ttg::MemDescIndexOp>(resultType, src, index);
           })
      .def("create_memdesc_subslice",
           [](GluonOpBuilder &self, Type resultType, Value src,
              std::vector<int32_t> &offsets) -> Value {
             return self.create<ttg::MemDescSubsliceOp>(resultType, src,
                                                        offsets);
           })
      .def("create_memdesc_trans",
           [](GluonOpBuilder &self, Value src,
              std::vector<int> &order) -> Value {
             return self.create<ttg::MemDescTransOp>(src, order);
           })
      .def("create_memdesc_reshape",
           [](GluonOpBuilder &self, Value src,
              std::vector<int64_t> &shape) -> Value {
             return self.create<ttg::MemDescReshapeOp>(src, shape);
           })
      .def("create_memdesc_reinterpret",
           [](GluonOpBuilder &self, Type resultType, Value src) -> Value {
             return self.create<ttg::MemDescReinterpretOp>(resultType, src);
           })
      .def("create_set_auto_layout",
           [](GluonOpBuilder &self, Attribute layout, Value value) -> Value {
             return self.create<gluon::SetAutoLayoutOp>(layout, value);
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
             std::vector<Value> results;
             auto wait = self.create<ttng::WarpGroupDotWaitOp>(deps, pendings);
             llvm::append_range(results, wait.getResults());
             return results;
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
      .def("create_tmem_copy",
           [](GluonOpBuilder &self, Value src, Value dst) {
             self.create<ttng::TMEMCopyOp>(src, dst, /*barrier=*/Value());
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
              std::vector<Value> &mbarrier_preds, bool two_ctas) {
             Value accDep;
             auto tokType = self.getBuilder().getType<ttg::AsyncTokenType>();
             self.create<ttng::TCGen5MMAOp>(tokType, a, b, acc, accDep, useAcc,
                                            pred, two_ctas, mbarriers,
                                            mbarrier_preds);
           })
      .def("create_tcgen05_mma_scaled",
           [](GluonOpBuilder &self, Value a, Value b, Value acc, Value aScale,
              Value bScale, tt::ScaleDotElemType aType,
              tt::ScaleDotElemType bType, Value useAcc, Value pred,
              std::vector<Value> &mbarriers,
              std::vector<Value> &mbarrier_preds) {
             Value accDep;
             auto tokType = self.getBuilder().getType<ttg::AsyncTokenType>();
             self.create<ttng::TCGen5MMAScaledOp>(
                 tokType, a, b, acc, accDep, aScale, bScale, aType, bType,
                 useAcc, pred, mbarriers, mbarrier_preds);
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
      .def("create_warp_specialize",
           [](GluonOpBuilder &self, std::vector<Type> &resultTypes,
              std::vector<Value> &explicitCaptures,
              std::vector<int> &partitionNumWarps) {
             return self.create<ttg::WarpSpecializeOp>(
                 resultTypes, explicitCaptures, partitionNumWarps);
           })
      .def("create_buffer_load",
           [](GluonOpBuilder &self, Type resultType, Value ptr, Value offsets,
              Value mask, Value other, tt::CacheModifier cache) -> Value {
             return self.create<ttag::BufferLoadOp>(resultType, ptr, offsets,
                                                    Value() /*stride*/, cache,
                                                    mask, other);
           })
      .def("create_buffer_store",
           [](GluonOpBuilder &self, Value storedValue, Value ptr, Value offsets,
              Value mask, tt::CacheModifier cache) {
             self.create<ttag::BufferStoreOp>(storedValue, ptr, offsets,
                                              Value() /*stride*/, cache, mask);
           })
      .def("create_buffer_atomic_rmw",
           [](GluonOpBuilder &self, tt::RMWOp op, Value ptr, Value offsets,
              Value value, tt::MemSemantic sem, tt::MemSyncScope scope,
              Value mask) -> Value {
             return self.create<ttag::BufferAtomicRMWOp>(
                 value.getType(), op, ptr, offsets, value, Value() /*stride*/,
                 sem, scope, mask);
           })
      .def("create_buffer_load_to_local",
           [](GluonOpBuilder &self, Value dest, Value ptr, Value offsets,
              Value mask, Value other, Value stride,
              tt::CacheModifier cacheModifier) {
             self.create<ttag::BufferLoadToLocalOp>(
                 dest, ptr, offsets, mask, other, stride, cacheModifier);
           })
      .def("create_make_tensor_descriptor",
           [](TritonOpBuilder &self, Type resultTy, Value &base,
              std::vector<Value> &shape, std::vector<Value> &strides,
              tt::PaddingOption paddingOption) -> Value {
             return self.create<tt::MakeTensorDescOp>(resultTy, base, shape,
                                                      strides, paddingOption);
           })
      .def("create_async_tdm_copy_global_to_local",
           [](GluonOpBuilder &self, Value descPtr, std::vector<Value> &indices,
              Value result, Value pred, Value barrier) {
             self.create<ttag::AsyncTDMCopyGlobalToLocalOp>(
                 descPtr, indices, result, pred, barrier);
           })
      .def("create_async_tdm_copy_local_to_global",
           [](GluonOpBuilder &self, Value descPtr, std::vector<Value> &indices,
              Value src, Value barrier) {
             self.create<ttag::AsyncTDMCopyLocalToGlobalOp>(descPtr, indices,
                                                            src, barrier);
           })
      .def("create_async_tdm_wait",
           [](GluonOpBuilder &self, int num) {
             ValueRange tokens;
             self.create<ttag::AsyncTDMWait>(tokens, num);
           })
      .def("create_async_copy_lds_barrier_arrive",
           [](GluonOpBuilder &self, Value mbarrier) {
             self.create<ttag::AsyncCopyMbarrierArriveOp>(mbarrier);
           })
      .def("create_lds_barrier_init",
           [](GluonOpBuilder &self, Value memDesc, int count) {
             self.create<ttag::InitBarrierOp>(memDesc, count);
           })
      .def("create_lds_barrier_wait",
           [](GluonOpBuilder &self, Value memDesc, Value phase) {
             self.create<ttag::WaitBarrierOp>(memDesc, phase);
           })
      .def("create_lds_barrier_arrive",
           [](GluonOpBuilder &self, Value memDesc, int count) -> Value {
             return self.create<ttag::ArriveBarrierOp>(memDesc, count);
           })
      .def("create_warp_pipeline_border",
           [](GluonOpBuilder &self, const std::string &marker) {
             auto border = self.create<ROCDL::SchedBarrier>(0);
             auto ctx = self.getContext();
             border->setAttr("triton.warp_pipeline.border",
                             StringAttr::get(ctx, marker));
           });

  m.def(
      "compute_tmem_reg_layout",
      [](py::object elementTyObj, std::vector<int64_t> shape,
         py::object layoutObj, unsigned numWarps, const std::string &atomName,
         std::vector<std::vector<int32_t>> cgaBases) -> py::object {
        DialectRegistry registry;
        registry.insert<triton::TritonDialect, ttg::TritonGPUDialect,
                        ttng::TritonNvidiaGPUDialect, gluon::GluonDialect>();
        MLIRContext context(MLIRContext::Threading::DISABLED);
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();

        GluonOpBuilder builder(&context);
        auto builderObj =
            py::cast(&builder, py::return_value_policy::reference);

        auto elementType = elementTyObj.attr("to_ir")(builderObj).cast<Type>();
        auto layoutAttr =
            layoutObj.attr("_to_ir")(builderObj).cast<Attribute>();
        auto allocShape = shape;

        auto ctx = builder.getContext();
        unsigned rank = shape.size();
        auto memDescTy = builder.getChecked<ttg::MemDescType>(
            shape, elementType, layoutAttr,
            ttng::TensorMemorySpaceAttr::get(ctx),
            /*mutableMemory=*/true, allocShape);
        auto ctaLayoutAttr = buildCgaLayoutAttr(ctx, cgaBases, rank);

        auto maybeAtom =
            llvm::StringSwitch<std::optional<ttng::TMemAccessAtom>>(atomName)
                .Case("32x32b", ttng::TMemAccessAtom::I32x32b)
                .Case("16x64b", ttng::TMemAccessAtom::I16x64b)
                .Case("16x128b", ttng::TMemAccessAtom::I16x128b)
                .Case("16x256b", ttng::TMemAccessAtom::I16x256b)
                .Case("16x32bx2", ttng::TMemAccessAtom::I16x32bx2)
                .Default(std::nullopt);
        if (!maybeAtom)
          throw std::invalid_argument("unknown TMEM access atom: " + atomName);
        auto atom = *maybeAtom;
        if (atom == ttng::TMemAccessAtom::I16x32bx2)
          throw std::invalid_argument(
              "Atom 16x32bx2 is inferred implicitly and cannot be requested "
              "explicitly");
        if (numWarps < 4 || !llvm::isPowerOf2_32(numWarps))
          throw std::invalid_argument(
              "numWarps must be a power of two and >= 4");

        auto layout = ttng::getDistributedLayoutForTmemLdSt(
            memDescTy, atom, numWarps, ctaLayoutAttr);
        if (!layout)
          return py::none();

        auto attr = ttg::LinearEncodingAttr::get(ctx, std::move(*layout));
        return layoutToGluon(attr);
      });

  m.def(
      "make_cga_layout",
      [](std::vector<unsigned> ctasPerCga, std::vector<unsigned> ctaSplitNum,
         std::vector<unsigned> ctaOrder) -> std::vector<std::vector<int32_t>> {
        DialectRegistry registry;
        registry.insert<triton::TritonDialect, ttg::TritonGPUDialect>();
        MLIRContext ctx(MLIRContext::Threading::DISABLED);
        ctx.appendDialectRegistry(registry);
        ctx.loadAllAvailableDialects();
        auto attr = ttg::CGAEncodingAttr::fromSplitParams(
            &ctx, ctasPerCga, ctaSplitNum, ctaOrder);
        return getCgaLayoutBases(attr);
      });

  m.def("get_amd_mfma_scale_layout",
        [](unsigned opIdx, std::vector<int64_t> &shape, unsigned mfmaMDim,
           std::vector<unsigned> &tilesPerWarp,
           std::vector<unsigned> &warpsPerCTA) -> py::object {
          DialectRegistry registry;
          registry.insert<triton::TritonDialect, ttg::TritonGPUDialect,
                          ttng::TritonNvidiaGPUDialect, gluon::GluonDialect>();
          MLIRContext ctx(MLIRContext::Threading::DISABLED);
          ctx.appendDialectRegistry(registry);
          ctx.loadAllAvailableDialects();

          auto ll = ttg::chooseScaledMfmaScaleLayout(
              &ctx, opIdx, shape, mfmaMDim, tilesPerWarp, warpsPerCTA);
          auto attr = ttg::LinearEncodingAttr::get(&ctx, std::move(ll));
          return layoutToGluon(attr);
        });

  m.def("get_amd_wmma_scale_layout",
        [](unsigned opIdx, std::vector<int64_t> &shape, unsigned wmmaMDim,
           std::vector<unsigned> &tilesPerWarp,
           std::vector<unsigned> &warpsPerCTA) -> py::object {
          DialectRegistry registry;
          registry.insert<triton::TritonDialect, ttg::TritonGPUDialect,
                          ttng::TritonNvidiaGPUDialect, gluon::GluonDialect>();
          MLIRContext ctx(MLIRContext::Threading::DISABLED);
          ctx.appendDialectRegistry(registry);
          ctx.loadAllAvailableDialects();

          auto ll = ttg::chooseScaledWmmaScaleLayout(
              &ctx, opIdx, shape, wmmaMDim, tilesPerWarp, warpsPerCTA);
          auto attr = ttg::LinearEncodingAttr::get(&ctx, std::move(ll));
          return layoutToGluon(attr);
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
