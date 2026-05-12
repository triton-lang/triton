#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-convert-buffer-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using ::mlir::LLVM::AMD::getVectorSize;
using mlir::triton::amdgpu::TargetFeatures;

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCONVERTTOBUFFEROPS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// Return true iff the given value v is a tensor splatting from 1 (int).
// The usefulness of this func stems from the fact than if a buffer-op's mask
// operand is a all-1-tensor, it does not need to take this operand.
bool isSplatOneConstTensor(const Value v) {
  auto constantOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return false;

  if (auto denseAttr =
          dyn_cast<DenseIntElementsAttr>(constantOp.getValueAttr()))
    return denseAttr.isSplat() && denseAttr.getSplatValue<APInt>().isOne();

  return false;
}

bool isByteOffsetSmallerThan2GB(triton::AddPtrOp addPtrOp,
                                std::shared_ptr<DataFlowSolver> solver) {
  Value elemIdx = addPtrOp.getOffset();
  LDBG("Determining value-range of element-index: " << elemIdx);

  // step 1: Get the value range of the element index
  const auto *lattice =
      solver->lookupState<dataflow::IntegerValueRangeLattice>(elemIdx);
  if (!lattice) {
    // Note that it is not always able to get lattice, e.g. the element-index
    // is defined by a tt.load.
    LDBG("Cannot get lattice");
    return false;
  }

  const mlir::IntegerValueRange &vr = lattice->getValue();
  if (vr.isUninitialized() || AMD::isEmptyInitializedRange(vr.getValue())) {
    LDBG("Cannot get value range of the offset");
    return false;
  };

  const auto &smin = vr.getValue().smin();
  const auto &smax = vr.getValue().smax();

  LDBG("Element-index value-range: " << smin << " : " << smax);
  if (smin.isNegative() || smax.isNegative())
    return false;

  // step 2: Get element type and size.
  // e.g. addPtrOp.getType is tensor<64x64x!tt.ptr<f16>, then elemTy is
  // !tt.ptr<f16>, and dereferencing elemTy gets f16.
  // TODO: Not sure if we need to keep dereferencing in a loop.
  Type elemTy = getElementTypeOrSelf(addPtrOp.getType());
  while (auto ptrTy = dyn_cast<triton::PointerType>(elemTy))
    elemTy = ptrTy.getPointeeType();

  if (!elemTy || !elemTy.isIntOrFloat()) {
    LDBG("unknown element type: " << elemTy);
    return false;
  }

  // step 3: check of byte-offset is within 2G
  int64_t elemBitSz = elemTy.getIntOrFloatBitWidth();
  int64_t elemMaxIdx = smax.getSExtValue();
  int64_t byteOfst = (elemBitSz * elemMaxIdx + elemBitSz + 7) / 8;
  int64_t szLimit2GB = (1L << 31) - 1;

  LDBG("element bit sz:" << elemBitSz << ", max byte offset:" << byteOfst
                         << ((szLimit2GB > byteOfst) ? ", out of range"
                                                     : ", in range"));

  return byteOfst <= szLimit2GB;
}

bool isFuncArgWith32bitPtrRange(mlir::Value value) {
  if (value.getDefiningOp())
    return false;

  mlir::BlockArgument blockArg = mlir::cast<mlir::BlockArgument>(value);
  auto blk = blockArg.getOwner();
  auto funcOp = dyn_cast_or_null<tt::FuncOp>(blk->getParentOp());

  if (funcOp && blk == &funcOp->getRegion(0).front()) {
    for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
      if (arg != value)
        continue;
      auto attr = funcOp.getArgAttrOfType<IntegerAttr>(idx, "tt.pointer_range");
      return attr && attr.getInt() <= 32;
    }
  }

  return false;
}

// Pure query: check whether the pointer can be lowered to buffer ops.
// This function must not modify IR. The actual offset truncation (i64 -> i32)
// is handled separately by truncateOffsetToI32().
bool canUseBufferOps(Value ptr,
                     const DenseMap<Value, SetVector<Operation *>> &assumptions,
                     std::shared_ptr<DataFlowSolver> solver,
                     bool analyzeSmallTensorOfst) {
  // 1. Check if the pointer is uniform: i.e., if it comes from a uniform
  // pointer(splatted) and non-uniform offset addition.
  LDBG("Buffer op checks for: " << ptr);
  auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
  if (!addPtrOp)
    return false;

  auto maybeSplatOp = addPtrOp.getPtr().getDefiningOp<triton::SplatOp>();
  if (!maybeSplatOp)
    return false;
  LDBG("Pattern matched");

  // 2. Check offset bit width. Buffer ops support i32 offsets natively;
  // i64 offsets are truncated later if proven safe.
  Value offset = addPtrOp.getOffset();
  auto ofstBit =
      cast<RankedTensorType>(offset.getType()).getElementTypeBitWidth();
  LLVM_DEBUG(llvm::dbgs() << "offset bits:" << ofstBit << "\n");

  if (ofstBit != 32 && ofstBit != 64)
    return false;

  // 3. Determine if buffer op conversion is safe via pointer_range attribute
  // or range analysis.
  bool isSafe = false;
  if (!analyzeSmallTensorOfst &&
      isFuncArgWith32bitPtrRange(maybeSplatOp.getSrc())) {
    LDBG("base-ptr has tt.pointer_range=32 attribute");
    isSafe = true;
  } else {
    isSafe = isByteOffsetSmallerThan2GB(addPtrOp, std::move(solver));
  }

  return isSafe;
}

// Buffer ops require i32 offsets. If the offset is already i32, return it
// as-is. If it is i64, insert an arith.trunci right before insertBefore.
// The caller's insertion point is saved and restored automatically.
Value truncateOffsetToI32(Value origOffset, OpBuilder &builder, Location loc,
                          Operation *insertBefore) {
  auto offsetTy = cast<RankedTensorType>(origOffset.getType());
  if (offsetTy.getElementTypeBitWidth() == 32)
    return origOffset;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(insertBefore);
  auto i32Ty = RankedTensorType::get(offsetTy.getShape(), builder.getI32Type(),
                                     offsetTy.getEncoding());
  return arith::TruncIOp::create(builder, loc, i32Ty, origOffset);
}

// Extract stride of the blocked offset of LD/ST ops.
Value getBlockStride(Location loc, Value offset, PatternRewriter &rewriter) {
  // Buffer ops take an i32 offset; `truncateOffsetToI32` may insert
  // `arith.trunci` from i64. That op sits in front of the offset chain that
  // `getBlockStride` pattern-matches, so peel it. Any `trunci` here is from
  // that helper (same pass); checking the result is i32 matches it.
  if (auto truncOp = offset.getDefiningOp<arith::TruncIOp>()) {
    if (getElementTypeOrSelf(truncOp.getResult().getType()).isInteger(32))
      offset = truncOp.getIn();
  }
  // canonicalize pointer pass sets block stride via
  // `offset:add-broadcast-muli-splat`, backtrace that pattern to reach the
  // stride.
  if (auto maybeAdd = offset.getDefiningOp<arith::AddIOp>())
    for (auto addOpr : maybeAdd.getOperands())
      if (auto maybeBC = addOpr.getDefiningOp<tt::BroadcastOp>()) {
        auto bcSrc = maybeBC.getSrc();
        if (auto maybeMul = bcSrc.getDefiningOp<arith::MulIOp>())
          for (auto mulOpr : maybeMul.getOperands())
            if (auto maybeSplat = mulOpr.getDefiningOp<tt::SplatOp>())
              return maybeSplat.getSrc();
      }
  return nullptr;
}

// Buffer ops take Optional<I32> stride. getBlockStride walks the offset chain
// and returns the splat's scalar source, which may be i64 when the kernel uses
// i64 indices (e.g. row * stride + col with stride in i64).
static Value maybeTruncateStrideToI32(Value stride, PatternRewriter &rewriter,
                                      Location loc, Operation *insertBefore) {
  if (!stride)
    return stride;
  auto intTy = dyn_cast<IntegerType>(stride.getType());
  if (!intTy)
    return stride;
  if (intTy.getWidth() == 32)
    return stride;
  if (intTy.getWidth() == 64) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(insertBefore);
    return arith::TruncIOp::create(rewriter, loc, rewriter.getI32Type(),
                                   stride);
  }
  return stride;
}

// /*-----------------AtomicCAS-------------------*/

struct ConvertTritonAtomicCASOpToBufferAtomicCAS
    : public mlir::OpRewritePattern<triton::AtomicCASOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonAtomicCASOpToBufferAtomicCAS(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      std::shared_ptr<DataFlowSolver> solver, bool analyzeSmallTensorOfst_)
      : mlir::OpRewritePattern<triton::AtomicCASOp>(context),
        assumptions(assumptions), axisAnalysisPass(axisAnalysisPass),
        solver(std::move(solver)),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_) {}

  mlir::LogicalResult
  matchAndRewrite(triton::AtomicCASOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();
    auto sem = op.getSem();
    auto scope = op.getScope();

    if (!canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst)) {
      return rewriter.notifyMatchFailure(op, "canUseBufferOps check failed");
    }

    switch (scope) {
    case MemSyncScope::GPU:
    case MemSyncScope::CTA:
      break;
    default:
      return rewriter.notifyMatchFailure(op, "CAS with unsupported scope");
    }
    LDBG("CAS supported scope");

    switch (sem) {
    case MemSemantic::RELAXED:
    case MemSemantic::RELEASE:
    case MemSemantic::ACQUIRE:
    case MemSemantic::ACQUIRE_RELEASE:
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "CAS with unsupported memory ordering");
    }

    // Buffer atomic CAS only supports i32/i64
    auto checkType = getElementTypeOrSelf(op.getVal());
    bool isSupportedType = checkType.isInteger(32) || checkType.isInteger(64);
    if (!isSupportedType) {
      return rewriter.notifyMatchFailure(op, "AtomicCAS with unsupported type");
    }
    LDBG("AtomicCAS supported type");

    // All checks passed; now safe to modify IR.
    auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
    Value tensorPtr = addPtrOp.getPtr();
    Value offset = addPtrOp.getOffset();
    Value tensorOffset =
        truncateOffsetToI32(offset, rewriter, op->getLoc(), op);
    auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
    Value basePtr = splatOp.getSrc();

    // Buffer atomics support 32 and 64-bit operations, so inputs must be at
    // least 32-bits. Otherwise, fall back to the existing path for atomics
    auto opValueType = op.getVal().getType();
    auto opBitWidth = 0;
    if (auto tensorType = dyn_cast<RankedTensorType>(opValueType)) {
      auto elemBitWidth = tensorType.getElementTypeBitWidth();
      opBitWidth =
          getVectorSize(basePtr, tensorOffset, axisAnalysisPass) * elemBitWidth;
    } else {
      opBitWidth = opValueType.getIntOrFloatBitWidth();
    }

    if (opBitWidth < 32) {
      return rewriter.notifyMatchFailure(
          op, "BufferAtomicCAS requires opBitWidth >= 32");
    }
    Value blockStride = maybeTruncateStrideToI32(
        getBlockStride(op->getLoc(), tensorOffset, rewriter), rewriter,
        op->getLoc(), op);
    rewriter.replaceOpWithNewOp<triton::amdgpu::BufferAtomicCASOp>(
        op, op.getVal().getType(), basePtr, tensorOffset, op.getCmp(),
        op.getVal(), blockStride, sem, scope);
    return success();
  }

private:
  // Assumptions collected through the function
  const DenseMap<Value, SetVector<Operation *>> &assumptions;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  std::shared_ptr<DataFlowSolver> solver;
  bool analyzeSmallTensorOfst;
};

struct ConvertTritonAtomicRMWOpToBufferAtomicRMW
    : public mlir::OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonAtomicRMWOpToBufferAtomicRMW(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      std::shared_ptr<DataFlowSolver> solver,
      const TargetFeatures &targetFeatures, bool analyzeSmallTensorOfst_)
      : mlir::OpRewritePattern<triton::AtomicRMWOp>(context),
        assumptions(assumptions), axisAnalysisPass(axisAnalysisPass),
        solver(std::move(solver)), targetFeatures(targetFeatures),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_) {}

  mlir::LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();
    auto atomicRmwOp = op.getAtomicRmwOp();
    auto sem = op.getSem();
    auto scope = op.getScope();

    // In addition to the `canUseBufferOps` check, we should ensure that
    // 1. Perform the canUseBufferOps check
    if (!canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst)) {
      return rewriter.notifyMatchFailure(op, "canUseBufferOps check failed");
    }

    // 2. Check the scope. We support GPU and CTA for now (SYSTEM scope is not
    // supported yet)
    switch (scope) {
    case MemSyncScope::GPU:
    case MemSyncScope::CTA:
      break;
    default:
      return rewriter.notifyMatchFailure(op, "RMW with unsupported scope");
    }
    LDBG("RMW supported scope");

    // 3. Check the memory ordering.
    //    TODO: support monotonic
    switch (sem) {
    case MemSemantic::RELAXED:
    case MemSemantic::RELEASE:
    case MemSemantic::ACQUIRE:
    case MemSemantic::ACQUIRE_RELEASE:
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "RMW with unsupported memory ordering");
    }

    // 4. Buffer atomic RMW does not support FP8 ops
    //    easier to just check what we support
    auto checkType = getElementTypeOrSelf(op.getVal());
    bool isSupportedType = checkType.isF16() || checkType.isBF16() ||
                           checkType.isF32() || checkType.isF64() ||
                           checkType.isInteger(32) || checkType.isInteger(64);
    if (!isSupportedType) {
      return rewriter.notifyMatchFailure(op, "RMW with unsupported type");
    }
    LDBG("RMW supported type");

    if (atomicRmwOp == RMWOp::FADD &&
        !targetFeatures.supportsBufferAtomicFadd(checkType)) {
      return rewriter.notifyMatchFailure(
          op, "RMW FADD unsupported for this type on target");
    }
    LDBG("RMW FADD supported type");

    auto vecSize = getVectorSize(ptr, axisAnalysisPass);
    if (auto mask = op.getMask()) {
      vecSize = std::min(vecSize, axisAnalysisPass.getMaskAlignment(mask));
    }
    // f16/bf16 dtypes could only be efficiently calculated using instructions
    // that pack 2 elements (e.g. @llvm.amdgcn.raw.buffer.atomic.fadd.v2f16)
    if (vecSize % 2 != 0 && (checkType.isF16() || checkType.isBF16())) {
      return rewriter.notifyMatchFailure(
          op, "RMW float 16 dtypes must be aligned by 2");
    }
    LDBG("RMW passed alignment check");

    // 5. Check if the RMWOp is supported
    switch (atomicRmwOp) {
    case RMWOp::AND:
    case RMWOp::OR:
    case RMWOp::XOR:
    case RMWOp::ADD:
    case RMWOp::FADD:
    case RMWOp::UMAX:
    case RMWOp::UMIN:
    case RMWOp::XCHG:
      break;
    case RMWOp::MAX:
    case RMWOp::MIN:
      // TODO: It likely means smax/smin, for now intrinsic
      // llvm.amdgcn.raw.ptr.buffer.atomic.{min|max} is emitted, and llvm get
      // confused as how to deal with {f|s|u}{min|max}.
      if (!checkType.isInteger())
        break;
      // else fall through
    default:
      auto rmwOpStr = stringifyRMWOp(atomicRmwOp).str();
      return rewriter.notifyMatchFailure(op, "RMW with unsupported op: " +
                                                 rmwOpStr);
    }
    LDBG("RMW supported Op");

    // 6. Buffer atomics support 32 and 64-bit operations, so inputs must be at
    //    least 32-bits. Otherwise, fall back to the existing path for atomics
    auto opValueType = op.getVal().getType();
    auto opBitWidth = 0;
    if (auto tensorType = dyn_cast<RankedTensorType>(opValueType)) {
      // We can't just compute the opBitWidth using the numElements *
      // elemBitWidth here. In cases such as tensor<2xf16...>, if the elements
      // are contiguous we can emit the buffer op. Otherwise, the buffer ops
      // lowering will try to emit individual (unsupported) f16/bf16 ops.
      auto elemBitWidth = tensorType.getElementTypeBitWidth();
      opBitWidth = vecSize * elemBitWidth;
    } else {
      opBitWidth = opValueType.getIntOrFloatBitWidth();
    }

    if (opBitWidth < 32) {
      return rewriter.notifyMatchFailure(op, "RMW requires opBitWidth >= 32");
    }

    // All checks passed; now safe to modify IR.
    auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
    Value tensorPtr = addPtrOp.getPtr();
    Value offset = addPtrOp.getOffset();
    Value tensorOffset =
        truncateOffsetToI32(offset, rewriter, op->getLoc(), op);
    auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
    Value basePtr = splatOp.getSrc();

    Value maybeMask{};
    if (op.getMask() && !isSplatOneConstTensor(op.getMask()))
      maybeMask = op.getMask();
    Value blockStride = maybeTruncateStrideToI32(
        getBlockStride(op->getLoc(), tensorOffset, rewriter), rewriter,
        op->getLoc(), op);
    rewriter.replaceOpWithNewOp<triton::amdgpu::BufferAtomicRMWOp>(
        op, op.getVal().getType(), atomicRmwOp, basePtr, tensorOffset,
        op.getVal(), blockStride, sem, scope, maybeMask);

    return success();
  }

private:
  // Assumptions collected through the function
  DenseMap<Value, SetVector<Operation *>> assumptions;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  std::shared_ptr<DataFlowSolver> solver;
  TargetFeatures targetFeatures;
  bool analyzeSmallTensorOfst;
};

// Workaround to allow static_assert(false) on older compilers as it was
// ill-formed before defect report CWG2518
// (https://cplusplus.github.io/CWG/issues/2518.html)
template <typename T> struct always_false : std::false_type {};

template <typename SourceOp>
struct ConvertTritonLoadToBufferLoad : public mlir::OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  ConvertTritonLoadToBufferLoad(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      std::shared_ptr<DataFlowSolver> solver, bool analyzeSmallTensorOfst_)
      : mlir::OpRewritePattern<SourceOp>(context), assumptions(assumptions),
        axisAnalysisPass(axisAnalysisPass), solver(std::move(solver)),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_) {}
  mlir::LogicalResult
  matchAndRewrite(SourceOp op, PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getOperand(0);

    if (canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value offset = addPtrOp.getOffset();
      Value tensorOffset =
          truncateOffsetToI32(offset, rewriter, op->getLoc(), op);
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeOther{};
      if (op.getOther() && !isZeroConst(op.getOther()))
        maybeOther = op.getOther();
      Value maybeMask{};
      if (op.getMask() && !isSplatOneConstTensor(op.getMask()))
        maybeMask = op.getMask();
      Value blockStride = maybeTruncateStrideToI32(
          getBlockStride(op->getLoc(), tensorOffset, rewriter), rewriter,
          op->getLoc(), op);

      auto bufferLoadOp = [&]() {
        if constexpr (std::is_same_v<SourceOp, triton::LoadOp>) {
          unsigned contig = getVectorSize(ptr, axisAnalysisPass);
          if (maybeMask)
            contig = std::min<unsigned>(
                contig, axisAnalysisPass.getMaskAlignment(maybeMask));
          return triton::amdgpu::BufferLoadOp::create(
              rewriter, op->getLoc(), op.getType(), basePtr, tensorOffset,
              blockStride, op.getCache(), maybeMask, maybeOther, contig);
        } else if constexpr (std::is_same_v<
                                 SourceOp,
                                 triton::gpu::AsyncCopyGlobalToLocalOp>) {
          return triton::amdgpu::BufferLoadToLocalOp::create(
              rewriter, op->getLoc(), op.getType(), op.getResult(), basePtr,
              tensorOffset, maybeMask, maybeOther, blockStride, op.getCache(),
              op.getContiguity());
        } else {
          static_assert(always_false<SourceOp>::value,
                        "Unsupported type in ConvertTritonLoadToBufferLoad");
        }
      }();

      assert(bufferLoadOp);

      rewriter.replaceOp(op, bufferLoadOp);
      return success();
    }

    LDBG("Failed to convert: " << op);
    return rewriter.notifyMatchFailure(op, "Failed to convert LoadOp");
  }

private:
  // Assumptions collected through the function
  DenseMap<Value, SetVector<Operation *>> assumptions;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  std::shared_ptr<DataFlowSolver> solver;
  bool analyzeSmallTensorOfst;
};

struct ConvertTritonStoreToBufferStore
    : public mlir::OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonStoreToBufferStore(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      std::shared_ptr<DataFlowSolver> solver, bool analyzeSmallTensorOfst_)
      : mlir::OpRewritePattern<triton::StoreOp>(context),
        assumptions(assumptions), axisAnalysisPass(axisAnalysisPass),
        solver(std::move(solver)),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_) {}

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();

    if (canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value offset = addPtrOp.getOffset();
      Value tensorOffset =
          truncateOffsetToI32(offset, rewriter, op->getLoc(), op);
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeMask{};
      unsigned contig = getVectorSize(ptr, axisAnalysisPass);
      if (op.getMask() && !isSplatOneConstTensor(op.getMask())) {
        maybeMask = op.getMask();
        contig = std::min<unsigned>(
            contig, axisAnalysisPass.getMaskAlignment(maybeMask));
      }
      Value blockStride = maybeTruncateStrideToI32(
          getBlockStride(op->getLoc(), tensorOffset, rewriter), rewriter,
          op->getLoc(), op);

      rewriter.replaceOpWithNewOp<triton::amdgpu::BufferStoreOp>(
          op, op.getValue(), basePtr, tensorOffset, blockStride, op.getCache(),
          maybeMask, contig);
      return success();
    }
    LDBG("Failed to convert: " << op);
    return rewriter.notifyMatchFailure(op, "Failed to convert StoreOp");
  }

private:
  // Assumptions collected through the function
  DenseMap<Value, SetVector<Operation *>> assumptions;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  std::shared_ptr<DataFlowSolver> solver;
  bool analyzeSmallTensorOfst;
};

} // anonymous namespace

struct TritonAMDGPUConvertToBufferOpsPass
    : impl::TritonAMDGPUConvertToBufferOpsBase<
          TritonAMDGPUConvertToBufferOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();
    TargetFeatures targetFeatures{llvm::StringRef(gfxArch)};

    // Collect assumptions in the function
    DenseMap<Value, SetVector<Operation *>> assumptions =
        AMD::TritonIntegerRangeAnalysis::collectAssumptions(getOperation());
    std::shared_ptr<DataFlowSolver> solver = createDataFlowSolver();

    AMD::TritonIntegerRangeAnalysis *rangeAnalysis =
        solver->load<AMD::TritonIntegerRangeAnalysis>(
            assumptions, &getAnalysis<DominanceInfo>());
    AMD::initializeFuncOps(mod, rangeAnalysis);
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    patterns.add<ConvertTritonLoadToBufferLoad<tt::LoadOp>,
                 ConvertTritonStoreToBufferStore>(context, assumptions,
                                                  axisInfoAnalysis, solver,
                                                  this->analyzeSmallTensorOfst);
    if (targetFeatures.supportsBufferLoadToLocal()) {
      patterns
          .add<ConvertTritonLoadToBufferLoad<ttg::AsyncCopyGlobalToLocalOp>>(
              context, assumptions, axisInfoAnalysis, solver,
              this->analyzeSmallTensorOfst);
    }

    if (this->allowBufferAtomics && targetFeatures.supportsBufferAtomicRMW())
      patterns.add<ConvertTritonAtomicRMWOpToBufferAtomicRMW>(
          context, assumptions, axisInfoAnalysis, solver, targetFeatures,
          this->analyzeSmallTensorOfst);
    patterns.add<ConvertTritonAtomicCASOpToBufferAtomicCAS>(
        context, assumptions, axisInfoAnalysis, solver,
        this->analyzeSmallTensorOfst);

    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
