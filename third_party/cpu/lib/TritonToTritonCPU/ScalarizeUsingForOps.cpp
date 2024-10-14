#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "cpu/include/ScalarizePass/ScalarizeInterface.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_SCALARIZEUSINGFOROP
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

template <typename OpTy>
struct ScalarizeOpConversion : public OpRewritePattern<OpTy> {

  ScalarizeOpConversion(ModuleAxisInfoAnalysis &axisInfoAnalysis,
                        MLIRContext *context, bool skipGatherScatter)
      : OpRewritePattern<OpTy>(context), axisAnalysis(axisInfoAnalysis) {
    this->skipGatherScatter = skipGatherScatter;
  }

  Value createAlloca(Location loc, MemRefType ty, Operation *before,
                     PatternRewriter &rewriter) const {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(before);
    return rewriter.create<memref::AllocaOp>(
        loc, ty, rewriter.getIntegerAttr(rewriter.getI64Type(), 64));
  }

  // If tensor is not null and its element cannot be recomputed in a scalar
  // loop, then store it to a temporary buffer.
  Value storeIfNonScalarizable(Location loc, Value vals, Value zeroIdx,
                               Operation *allocaPoint,
                               PatternRewriter &rewriter) const {
    // To skip optional values and scalarizable value, that can be computed
    // inside loop
    if (!vals || canComputeScalarValue(vals))
      return nullptr;

    auto tensor = vals;
    auto tensorTy = cast<RankedTensorType>(vals.getType());
    auto elemTy = tensorTy.getElementType();
    if (isa<triton::PointerType>(elemTy)) {
      elemTy = IntegerType::get(elemTy.getContext(), 64);
    }
    // Memref of i1 assumes one element per byte when we load/store element,
    // but vector store (through transfer write) would write 1 bit per element.
    if (elemTy.isInteger(1)) {
      elemTy = rewriter.getI8Type();
      tensor = rewriter.create<arith::ExtUIOp>(
          loc,
          RankedTensorType::get(tensorTy.getShape(), elemTy,
                                tensorTy.getEncoding()),
          tensor);
    }
    auto memRefTy = MemRefType::get(tensorTy.getShape(), elemTy);
    Value memRef = createAlloca(vals.getLoc(), memRefTy, allocaPoint, rewriter);
    SmallVector<Value> indices(tensorTy.getRank(), zeroIdx);
    rewriter.create<triton::cpu::StoreOp>(vals.getLoc(), tensor, memRef);
    return memRef;
  }

  // Load scalar element from a temporary buffer or recompute it if the
  // buffer doesn't exist.
  Value loadOrComputeScalarValue(Value vals, Value tmpVals, ValueRange indices,
                                 PatternRewriter &rewriter) const {
    // Allow null value for easier handling of optional arguments.
    if (!vals)
      return nullptr;

    // If nothing loaded, value should be scalar computable
    if (!tmpVals) {
      if (!canComputeScalarValue(vals)) {
        llvm::errs()
            << "Passed value was not loaded and can't be computed as scalar: "
            << vals << "\n";
        llvm::report_fatal_error("Cannot proceed such value");
        return nullptr;
      }
      return computeScalarValue(vals.getDefiningOp(), vals, indices, rewriter);
    }

    // Load value from a temp buffer if any.
    Value val =
        rewriter.create<memref::LoadOp>(vals.getLoc(), tmpVals, indices);
    // If we load a pointer then additional cast is needed because tensor of
    // pointers is transformed into a vector of integers.
    auto elemTy = dyn_cast<RankedTensorType>(vals.getType()).getElementType();
    if (isa<PointerType>(elemTy))
      val = rewriter.create<IntToPtrOp>(vals.getLoc(), elemTy, val);
    // We need to transform loaded i8 back to i1.
    else if (elemTy.isInteger(1))
      val = rewriter.create<arith::TruncIOp>(val.getLoc(), rewriter.getI1Type(),
                                             val);
    return val;
  }

  // This is core methods that generates SCF::For
  // We are checking arguments and results of operation
  // to scalarize them if possible and load/store if they are dynamical
  LogicalResult scalarizeWithLoop(OpTy scalarizeOp,
                                  PatternRewriter &rewriter) const {
    llvm_unreachable("nope");
    return failure();
  }

  // Method that describes how to check arguments and results of operation
  // for scalarization
  bool shouldScalarizeOp(OpTy scalarizeOp) const {
    llvm_unreachable("nope");
    return false;
  }

  // code for Memory Ops, as requires getPtr method
  bool shouldScalarizeOpGeneric(OpTy scalarizeOp) const {

    auto ptr = scalarizeOp.getPtr();
    if (triton::isTensorPointerType(ptr.getType())) {
      return false;
    }

    auto axisInfo = axisAnalysis.getAxisInfo(ptr);
    if (isContiguousRowMajorAccess(axisInfo, scalarizeOp)) {
      return false;
    }

    auto [basePtr, offset] = getMemoryBaseOffset(scalarizeOp);
    if (skipGatherScatter && basePtr && offset) {
      return false;
    }

    // Scalar memory ops and boundary checks are not expected.
    if (!scalarizeOp.getBoundaryCheck().empty()) {
      return false;
    }

    return ScalarizeOpConversion<OpTy>::shouldScalarizeOp(scalarizeOp);
  }

  LogicalResult matchAndRewrite(OpTy scalarOp,
                                PatternRewriter &rewriter) const override {

    // We want to avoid a code explosion when scalarize loads of big vectors,
    // so try to build a scalar loop.
    if (shouldScalarizeOpGeneric(scalarOp) &&
        succeeded(scalarizeWithLoop(scalarOp, rewriter)))
      return success();
    return failure();
  }

protected:
  ModuleAxisInfoAnalysis &axisAnalysis;
  bool skipGatherScatter;
};

template <>
LogicalResult ScalarizeOpConversion<triton::StoreOp>::scalarizeWithLoop(
    triton::StoreOp storeOp, PatternRewriter &rewriter) const {
  auto loc = storeOp.getLoc();

  auto ptrs = storeOp.getPtr();
  auto mask = storeOp.getMask();
  auto vals = storeOp.getValue();
  auto cache = storeOp.getCache();
  auto evict = storeOp.getEvict();

  auto tensorTy = cast<RankedTensorType>(vals.getType());

  // Create some reused constants.
  Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  // Alloca is inserted similar to the load case.
  Operation *allocaPoint = storeOp;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();

  // Store a tensor of pointers, mask, and values into a temp buf if we can't
  // compute them in a loop.
  Value tmpPtrs =
      storeIfNonScalarizable(loc, ptrs, zeroIdx, allocaPoint, rewriter);
  Value tmpMask =
      storeIfNonScalarizable(loc, mask, zeroIdx, allocaPoint, rewriter);
  Value tmpVals =
      storeIfNonScalarizable(loc, vals, zeroIdx, allocaPoint, rewriter);

  // Create for-loops to iterate through all vector dimensions.
  SmallVector<scf::ForOp> forOps;
  SmallVector<Value> ivs;
  for (int64_t i = 0; i < tensorTy.getRank(); ++i) {
    Value upperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, tensorTy.getShape()[i]);
    auto forOp = rewriter.create<scf::ForOp>(loc, zeroIdx, upperBound, oneIdx);
    forOps.push_back(forOp);
    ivs.push_back(forOp.getInductionVar());
    rewriter.setInsertionPointToStart(forOp.getBody());
  }

  // Compute or load scalar args.
  Value scalarPtr = loadOrComputeScalarValue(ptrs, tmpPtrs, ivs, rewriter);
  Value scalarMask = loadOrComputeScalarValue(mask, tmpMask, ivs, rewriter);
  Value scalarVal = loadOrComputeScalarValue(vals, tmpVals, ivs, rewriter);

  if (!mask) {
    // Regular store case.
    auto store_op = rewriter.create<triton::StoreOp>(loc, scalarPtr, scalarVal,
                                                     cache, evict);
  } else {
    // Conditional store case
    rewriter.create<scf::IfOp>(loc, scalarMask,
                               [&](OpBuilder &builder, Location loc) {
                                 builder.create<triton::StoreOp>(
                                     loc, scalarPtr, scalarVal, cache, evict);
                                 builder.create<scf::YieldOp>(loc);
                               });
  }

  rewriter.eraseOp(storeOp);
  return success();
}

template <>
bool ScalarizeOpConversion<triton::StoreOp>::shouldScalarizeOp(
    triton::StoreOp scalarOp) const {

  if (!isa<RankedTensorType>(scalarOp.getValue().getType())) {
    return false;
  }

  auto tensorTy = cast<RankedTensorType>(scalarOp.getPtr().getType());
  return tensorTy.getNumElements() >= 16;
}

template <>
LogicalResult ScalarizeOpConversion<triton::LoadOp>::scalarizeWithLoop(
    triton::LoadOp loadOp, PatternRewriter &rewriter) const {
  auto loc = loadOp.getLoc();
  auto tensorTy = cast<RankedTensorType>(loadOp.getType());

  auto ptrs = loadOp.getPtr();
  auto mask = loadOp.getMask();
  auto other = loadOp.getOther();
  auto cache = loadOp.getCache();
  auto evict = loadOp.getEvict();
  auto isVolatile = loadOp.getIsVolatile();

  // Create some reused constants.
  Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  // There is alloca_scope operation to control alloca scopes. But its usage
  // in combination with nested SCF and multi-dimensional vectors make it
  // impossible to lower scopes to LLVM using existing MLIR passes. For now,
  // simply allocate temp memory in the function's region.
  // TODO: Use alloc for big buffers and revisit alloca scoping.
  Operation *allocaPoint = loadOp;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();

  // Allocate temp buffer for the result. Write the other value there if
  // we cannot write it in a loop.
  auto resMemRefTy =
      MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
  Value resMemRef = createAlloca(loc, resMemRefTy, allocaPoint, rewriter);
  bool storeOtherInLoop = static_cast<bool>(mask);
  if (other && !canComputeScalarValue(other)) {
    rewriter.create<triton::cpu::StoreOp>(loc, other, resMemRef);
    storeOtherInLoop = false;
  }

  // Store a tensor of pointers and mask into a temp buf if we can't
  // compute them in a loop.
  Value tmpPtrs =
      storeIfNonScalarizable(loc, ptrs, zeroIdx, allocaPoint, rewriter);
  Value tmpMask =
      storeIfNonScalarizable(loc, mask, zeroIdx, allocaPoint, rewriter);

  // Create for-loops to iterate through all vector dimensions.
  SmallVector<scf::ForOp> forOps;
  SmallVector<Value> ivs;
  for (int64_t i = 0; i < tensorTy.getRank(); ++i) {
    Value upperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, tensorTy.getShape()[i]);
    auto forOp = rewriter.create<scf::ForOp>(loc, zeroIdx, upperBound, oneIdx);
    forOps.push_back(forOp);
    ivs.push_back(forOp.getInductionVar());
    rewriter.setInsertionPointToStart(forOp.getBody());
  }

  // Compute or load a scalar arguments.
  Value scalarPtr = loadOrComputeScalarValue(ptrs, tmpPtrs, ivs, rewriter);
  Value scalarMask = loadOrComputeScalarValue(mask, tmpMask, ivs, rewriter);
  Value scalarOther;
  if (storeOtherInLoop) {
    if (other) {
      scalarOther =
          computeScalarValue(other.getDefiningOp(), other, ivs, rewriter);
    } else {
      scalarOther = rewriter.create<arith::ConstantOp>(
          loc, tensorTy.getElementType(),
          rewriter.getZeroAttr(tensorTy.getElementType()));
    }
  }

  if (!mask) {
    // Regular load case.
    Value val = rewriter.create<triton::LoadOp>(loc, scalarPtr, cache, evict,
                                                isVolatile);
    rewriter.create<memref::StoreOp>(loc, val, resMemRef, ivs);
  } else {
    // Conditional load case
    rewriter.create<scf::IfOp>(
        loc, scalarMask,
        [&](OpBuilder &builder, Location loc) {
          Value val = builder.create<triton::LoadOp>(loc, scalarPtr, cache,
                                                     evict, isVolatile);
          builder.create<memref::StoreOp>(loc, val, resMemRef, ivs);
          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          if (storeOtherInLoop)
            builder.create<memref::StoreOp>(loc, scalarOther, resMemRef, ivs);
          builder.create<scf::YieldOp>(loc);
        });
  }

  // Load vector from the temp storage and return it from alloca scope.
  rewriter.setInsertionPointAfter(forOps.front());
  SmallVector<Value> indices(tensorTy.getRank(), zeroIdx);
  Value res = rewriter.create<triton::cpu::LoadOp>(loc, tensorTy, resMemRef);
  rewriter.replaceOp(loadOp, res);
  return success();
}

template <>
bool ScalarizeOpConversion<triton::LoadOp>::shouldScalarizeOp(
    triton::LoadOp scalarOp) const {
  if (!isa<RankedTensorType>(scalarOp.getType())) {
    return false;
  }
  auto tensorTy = cast<RankedTensorType>(scalarOp.getType());
  return tensorTy.getNumElements() >= 16;
}

struct ScalarizeUsingForOpPass
    : public triton::cpu::impl::ScalarizeUsingForOpBase<
          ScalarizeUsingForOpPass> {
  using ScalarizeUsingForOpBase::ScalarizeUsingForOpBase;

  ScalarizeUsingForOpPass() : ScalarizeUsingForOpBase() {}

  ScalarizeUsingForOpPass(bool skipGatherScatter) : ScalarizeUsingForOpBase() {
    this->skipGatherScatter = skipGatherScatter;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    RewritePatternSet patterns(context);
    patterns.add<ScalarizeOpConversion<triton::LoadOp>,
                 ScalarizeOpConversion<triton::StoreOp>>(
        axisInfoAnalysis, context, skipGatherScatter);

    if (applyPatternsAndFoldGreedily(mod, std::move(patterns)).failed()) {
      return signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createScalarizeUsingForOpPass() {
  return std::make_unique<ScalarizeUsingForOpPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createScalarizeUsingForOpPass(bool skipGatherScatter) {
  return std::make_unique<ScalarizeUsingForOpPass>(skipGatherScatter);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
