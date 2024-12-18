#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <utility>

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

#define DEBUG_TYPE "tritonamdgpu-canonicalize-pointers"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

// -----------------------------------------------------------------------------
// Pointer canonicalizer utility class
// -----------------------------------------------------------------------------
// This class iterates through the argument of the `funcOp`, if the argument is
// a pointer, starts a walk through its transitive uses to build a in-memory
// data structure to record the current offset to that pointer. Only when the
// pointer is really loaded/stored we materialize the base pointer with the
// offset.
//
// Let's suppose that `arg0` is a pointer. The algorithm works like that:
//
// a) At the beginning the offset is a tensor initialized to zero, and we
//    associate with `%arg0` a `FatPtr{basePtr=%arg0, offset=0}`. Through the
//    algorithm `FatPtr.basePtr` represents the scalar base pointer (all the
//    uniform updates will go into that) and `FatPtr.offset` represents the
//    tensor offset (all the non-uniform updates will go into that)
//
//
// b) Follow the pointer through the IR. When we meet:
//    `%ptr = tt.addptr(%arg0, %offset)`
//
//    Isolate the uniform and the non-uniform contributions of %offset =
//    (%u_offset, %nu_offset) and update the scalar pointer and the tensor
//    offset
//    ```
//    %s_ptr = addi(%fatPoniters[ptr].basePtr, %u_offset)
//    %t_offset = addi(%fatPoniters[ptr].offset, %nu_offset)
//    %fatPointers[%ptr0] = FatPtr{base=%s_ptr, offset=%t_offset}
//    ```
// c) When we meet the `tt.load(%ptr)` or `tt.store(%ptr)` instructions,
//    replace that instruction with:
//    `%t_ptr = tt.splat(%fatPointers[%ptr].basePtr)
//    `%fat_ptr = tt.addptr(%t_ptr, %fatPointers[ptr].offset)`
//    `%data = tt.load(%fat_ptr)`
//
// Please note that `%offset` might be a 32bit or 64bit integer. If
// we can, we would like to use 32 bit integers. This can happen under
// certain conditions:
//
// a) We can determine that the offset cannot overflow. In this case, we can
//    downcast the pointer just before emitting the load
// b) We know that the underlying memory size can be expressed as a 32 bit
//    value. In this case we can simply start with a 32bit offset and downcast
//    if we ever meet 64 bit operations (because we know that the offset can be
//    contained in 32 bits)
//
namespace {

// Extend a 32bit `offset` into 64bit using a arith.extsi operation
static Value createExtend32bitOffsetTo64Bits(RewriterBase &rewriter,
                                             Location loc, Value offset) {
  if (auto tensorType = dyn_cast<RankedTensorType>(offset.getType())) {
    auto shape = tensorType.getShape();
    auto newTensorType = RankedTensorType::get(shape, rewriter.getI64Type(),
                                               tensorType.getEncoding());
    return rewriter.create<arith::ExtSIOp>(loc, newTensorType, offset);
  }
  return rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), offset);
}

// Narrow a 64bit `offset` into 32bit using a arith.trunci operation
static Value createNarrow64bitOffsetTo32bits(RewriterBase &rewriter,
                                             Location loc, Value offset) {
  Type elementType = getElementTypeOrSelf(offset);
  if (elementType.isInteger(32))
    return offset;

  if (auto tensorType = dyn_cast<RankedTensorType>(offset.getType())) {
    auto shape = tensorType.getShape();
    auto newTensorType = RankedTensorType::get(shape, rewriter.getI32Type(),
                                               tensorType.getEncoding());
    return rewriter.create<arith::TruncIOp>(loc, newTensorType, offset);
  }
  return rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), offset);
}

// Helper function to determine if the given `op` is a constant tensor and in
// that case return the scalar value.
std::optional<Value> maybeGetOrCreateScalarConstant(RewriterBase &rewriter,
                                                    Location loc, Value expr) {
  Operation *op = expr.getDefiningOp();

  // Check for splatness
  if (auto splatOp = dyn_cast_or_null<triton::SplatOp>(op))
    return splatOp.getSrc();

  // Check for constant
  DenseIntElementsAttr constVal;
  if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(op)) {
    Value val = constOp.getResult();
    if (matchPattern(val, m_Constant(&constVal)) && constVal.isSplat())
      return rewriter.create<arith::ConstantOp>(
          loc, constVal.getSplatValue<IntegerAttr>());
  }

  // Check for block arguments
  if (auto blockArg = dyn_cast_or_null<BlockArgument>(expr)) {
    Type type = blockArg.getType();
    if (!isa<RankedTensorType>(type))
      return blockArg;
  }

  return {};
}

// Narrowing logic
// For now we allow to narrow down to 32 bits only in the following case:
// - `baseOffset` is 32-bits and `addOffset`(64-bits) is zero
// TODO(max): is this correct?
bool canNarrowOffset(Value baseOffset, Value addOffset) {
  Type addOffsetType = getElementTypeOrSelf(addOffset);
  auto baseSplatOp = baseOffset.getDefiningOp<triton::SplatOp>();
  return baseSplatOp && addOffsetType.isInteger(32);
}

// Create a zero tensor with a given `type`
Value createTensorZero(RewriterBase &rw, Location loc, RankedTensorType type) {
  mlir::Attribute zeroAttr = rw.getZeroAttr(type.getElementType());
  auto zeroDenseAttr = DenseElementsAttr::get(type, zeroAttr);
  return rw.create<arith::ConstantOp>(loc, zeroDenseAttr);
}

} // namespace

std::pair<Value, Value> createDecomposeOffsetFromExpr(RewriterBase &rewriter,
                                                      Location loc, Value expr,
                                                      int64_t bitness);
// Offset extraction logic for an addition op:
// decompose(A+B) = {U(A)+U(B), NU(A)+NU(B)}
std::pair<Value, Value> createDecomposeOffsetFromAdd(RewriterBase &rewriter,
                                                     Location loc, Value expr,
                                                     int64_t bitness) {
  auto addOp = expr.getDefiningOp<arith::AddIOp>();
  auto [uniformOffsetL, nonUniformOffsetL] =
      createDecomposeOffsetFromExpr(rewriter, loc, addOp.getLhs(), bitness);
  auto [uniformOffsetR, nonUniformOffsetR] =
      createDecomposeOffsetFromExpr(rewriter, loc, addOp.getRhs(), bitness);
  Value uniformAdd =
      rewriter.create<arith::AddIOp>(loc, uniformOffsetL, uniformOffsetR);
  Value nonUniformAdd =
      rewriter.create<arith::AddIOp>(loc, nonUniformOffsetL, nonUniformOffsetR);
  return {uniformAdd, nonUniformAdd};
}

// Offset extraction logic for a multiplication op:
// decompose(A*B) = {U(A)*U(B), NU(A)*NU(B)+NU(B)*U(A)+U(A)*NU(B)}
std::pair<Value, Value> createDecomposeOffsetFromMul(RewriterBase &rewriter,
                                                     Location loc, Value expr,
                                                     int64_t bitness) {
  auto mulOp = expr.getDefiningOp<arith::MulIOp>();
  auto [uniformOffsetL, nonUniformOffsetL] =
      createDecomposeOffsetFromExpr(rewriter, loc, mulOp.getLhs(), bitness);
  auto [uniformOffsetR, nonUniformOffsetR] =
      createDecomposeOffsetFromExpr(rewriter, loc, mulOp.getRhs(), bitness);
  Value uniformMul =
      rewriter.create<arith::MulIOp>(loc, uniformOffsetL, uniformOffsetR);

  Value uniformOffsetLSplat = rewriter.create<triton::SplatOp>(
      loc, nonUniformOffsetL.getType(), uniformOffsetL);
  Value uniformOffsetRSplat = rewriter.create<triton::SplatOp>(
      loc, nonUniformOffsetR.getType(), uniformOffsetR);

  Value nonUNonU =
      rewriter.create<arith::MulIOp>(loc, nonUniformOffsetL, nonUniformOffsetR);
  Value nonUU = rewriter.create<arith::MulIOp>(loc, uniformOffsetLSplat,
                                               nonUniformOffsetR);
  Value uNonU = rewriter.create<arith::MulIOp>(loc, nonUniformOffsetL,
                                               uniformOffsetRSplat);

  Value tmp = rewriter.create<arith::AddIOp>(loc, nonUNonU, nonUU);
  Value nonUniformMul = rewriter.create<arith::AddIOp>(loc, tmp, uNonU);
  return {uniformMul, nonUniformMul};
}

std::pair<Value, Value> createDecomposeOffsetFromExpr(RewriterBase &rewriter,
                                                      Location loc, Value expr,
                                                      int64_t bitness) {

  // Base case 1: it is a splat. Return the scalar constant as the uniform part
  if (auto scalarConst = maybeGetOrCreateScalarConstant(rewriter, loc, expr)) {
    auto tensorZero =
        createTensorZero(rewriter, loc, cast<RankedTensorType>(expr.getType()));
    return {*scalarConst, tensorZero};
  }

  // Base case 2: block argument. Since it is not a scalar constant, it must be
  // a tensor. Note that this means we won't be able to decompose across loop
  // boundaries (TODO: giuseros).
  if (llvm::isa<BlockArgument>(expr)) {
    Value scalarZero = rewriter.create<arith::ConstantIntOp>(loc, 0, bitness);
    return {scalarZero, expr};
  }

  auto offsets =
      llvm::TypeSwitch<Operation *, std::pair<Value, Value>>(
          expr.getDefiningOp())
          .Case<triton::BroadcastOp>([&](auto broadcastOp) {
            auto [uniform, nonUniform] = createDecomposeOffsetFromExpr(
                rewriter, loc, broadcastOp.getSrc(), bitness);
            auto broadcastNonUniform = rewriter.create<triton::BroadcastOp>(
                loc, broadcastOp.getType(), nonUniform);
            return std::make_pair(uniform, broadcastNonUniform);
          })
          .Case<triton::ExpandDimsOp>([&](auto expandOp) {
            auto [uniform, nonUniform] = createDecomposeOffsetFromExpr(
                rewriter, loc, expandOp.getSrc(), bitness);
            auto expandNonUniform = rewriter.create<triton::ExpandDimsOp>(
                loc, nonUniform, expandOp.getAxis());
            return std::make_pair(uniform, expandNonUniform);
          })
          .Case<arith::AddIOp>([&](Operation *op) {
            return createDecomposeOffsetFromAdd(rewriter, loc, expr, bitness);
          })
          .Case<arith::MulIOp>([&](Operation *op) {
            return createDecomposeOffsetFromMul(rewriter, loc, expr, bitness);
          })
          .Default([&](Operation *op) {
            // Base case 3: it is not a supported operation. We assume no
            // uniform part
            Value scalarZero =
                rewriter.create<arith::ConstantIntOp>(loc, 0, bitness);
            return std::make_pair(scalarZero, expr);
          });

  return offsets;
}

static const std::string kPtrCanonPrefix = "__amdpointercanonicalize.";
static const std::string kLegalAttr = kPtrCanonPrefix + "legal__";
static const std::string kRewrittenAttr = kPtrCanonPrefix + "rewritten__";
static const std::string kSCFThenRewrittenAttr =
    kPtrCanonPrefix + "scf-then-rewritten__";
static const std::string kSCFElseRewrittenAttr =
    kPtrCanonPrefix + "scf-else-rewritten__";
static const std::string kSCFIfOpYieldFatPtrOffsets =
    kPtrCanonPrefix + "scf-if-yield-fatptr-offsets__";

static void setLegalAttr(RewriterBase &rewriter, Operation *newOp) {
  rewriter.modifyOpInPlace(newOp, [&] {
    newOp->setDiscardableAttr(kLegalAttr, rewriter.getUnitAttr());
  });
}

static void setRewrittenAttr(RewriterBase &rewriter, Operation *origOp) {
  rewriter.modifyOpInPlace(origOp, [&] {
    origOp->setDiscardableAttr(kRewrittenAttr, rewriter.getUnitAttr());
  });
}

static void setRewrittenLegalAttrs(RewriterBase &rewriter, Operation *origOp,
                                   Operation *newOp) {
  setRewrittenAttr(rewriter, origOp);
  setLegalAttr(rewriter, newOp);
}

Value createTensorPointer(
    RewriterBase &rewriter, Value basePtr, Value offset, Location loc,
    bool canNarrow,
    const llvm::SmallDenseMap<StringAttr, Attribute> &attributes) {
  auto tensorType = dyn_cast<RankedTensorType>(offset.getType());

  // Scalar case: we only need to `tt.addptr %basePtr, %offset`
  if (!tensorType) {
    auto addPtrOp = rewriter.create<triton::AddPtrOp>(loc, basePtr.getType(),
                                                      basePtr, offset);
    for (auto attribute : attributes)
      addPtrOp->setAttr(attribute.getFirst(), attribute.getSecond());
    return addPtrOp.getResult();
  }

  // Tensor case: splat the scalar pointer and add the (tensor) offset:
  // ```
  //    %tensorBasePtr = tt.splat %basePtr
  //    %tensorPtr = tt.addptr %tensorBasePtr, %offset
  // ```
  ArrayRef<int64_t> offsetShape = tensorType.getShape();
  auto tensorPtrType = RankedTensorType::get(offsetShape, basePtr.getType(),
                                             tensorType.getEncoding());
  if (canNarrow)
    offset = createNarrow64bitOffsetTo32bits(rewriter, loc, offset);

  triton::SplatOp tensorPtr =
      rewriter.create<triton::SplatOp>(loc, tensorPtrType, basePtr);
  setLegalAttr(rewriter, tensorPtr);
  triton::AddPtrOp addPtrOp =
      rewriter.create<triton::AddPtrOp>(loc, tensorPtrType, tensorPtr, offset);
  setLegalAttr(rewriter, addPtrOp);

  for (auto attribute : attributes)
    addPtrOp->setAttr(attribute.getFirst(), attribute.getSecond());
  return addPtrOp.getResult();
}

class TritonAMDGPUCanonicalizePointersPass
    : public TritonAMDGPUCanonicalizePointersBase<
          TritonAMDGPUCanonicalizePointersPass> {
public:
  TritonAMDGPUCanonicalizePointersPass() = default;

  void runOnOperation() override;
};

struct FatPointers {
  struct FatPtrAttrs {
    FatPtrAttrs(const FatPtrAttrs &other) = default;
    FatPtrAttrs &operator=(const FatPtrAttrs &other) = default;
    // for map default insert
    FatPtrAttrs() = default;
    bool canNarrow = false;
    llvm::SmallDenseMap<StringAttr, Attribute> attributes;
    friend bool operator==(const FatPtrAttrs &lhs, const FatPtrAttrs &rhs) {
      return lhs.canNarrow == rhs.canNarrow && lhs.attributes == rhs.attributes;
    }
    friend bool operator!=(const FatPtrAttrs &lhs, const FatPtrAttrs &rhs) {
      return !(lhs == rhs);
    }
  };
  using KeyT = std::pair<Value, Value>;
  using ValueT = FatPtrAttrs;
  using DenseMapT = DenseMap<KeyT, ValueT>;
  ValueT &operator[](const KeyT &k) { return pointers[k]; }
  ValueT &operator[](KeyT &&k) { return pointers[k]; }
  template <typename T>
  using const_arg_type_t = typename llvm::const_pointer_or_const_ref<T>::type;
  const ValueT &at(const_arg_type_t<KeyT> k) const { return pointers.at(k); }
  const bool contains(const KeyT &k) { return pointers.contains(k); }

private:
  DenseMapT pointers;
};

std::optional<UnrealizedConversionCastOp> getFatPtrCastOp(Value base,
                                                          Value offset) {
  std::optional<UnrealizedConversionCastOp> maybeCastOp;
  for (Operation *user : base.getUsers()) {
    if (auto castOp = llvm::dyn_cast<UnrealizedConversionCastOp>(user)) {
      if (castOp.getNumOperands() == 2 && castOp.getOperand(0) == base &&
          castOp.getOperand(1) == offset) {
        maybeCastOp = castOp;
      }
    }
  }
#ifndef NDEBUG
  for (Operation *user : offset.getUsers()) {
    if (auto castOp = llvm::dyn_cast<UnrealizedConversionCastOp>(user)) {
      if (castOp.getNumOperands() == 2 && castOp.getOperand(0) == base &&
          castOp.getOperand(1) == offset) {
        assert(
            castOp == *maybeCastOp &&
            "expected castop through base and castop through offset to match");
      }
    }
  }
#endif
  return maybeCastOp;
}

std::optional<UnrealizedConversionCastOp> getFatPtrCastOp(OpOperand &operand) {
  Value operandVal = operand.get();
  for (Operation *user : operandVal.getUsers()) {
    if (auto castOp = llvm::dyn_cast<UnrealizedConversionCastOp>(user)) {
      if (castOp.getNumOperands() == 2 &&
          (castOp.getOperand(0) == operandVal ||
           castOp.getOperand(1) == operandVal) &&
          castOp.getNumResults() == 1 &&
          std::distance(castOp->getUsers().begin(), castOp->getUsers().end()) ==
              1 &&
          *castOp->getUsers().begin() == operand.getOwner()) {
        return castOp;
      }
    }
  }
  return {};
}

/// Flatten the given value ranges into a single vector of values.
static SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> result;
  for (const ValueRange &vals : values)
    llvm::append_range(result, vals);
  return result;
}

/// Assert that the given value range contains a single value and return it.
static Value getSingleValue(ValueRange values) {
  assert(values.size() == 1 && "expected single value");
  return values.front();
}

template <typename SourceOp>
struct PointerCanonicalizationPattern : OpConversionPattern<SourceOp> {
  PointerCanonicalizationPattern(MLIRContext *context, FatPointers &fatPtrs,
                                 PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(context, benefit), fatPtrs(fatPtrs) {}
  FatPointers &fatPtrs;
};

/// splat integer offset, keep base
class ConvertSplatOp : public PointerCanonicalizationPattern<triton::SplatOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp splatOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange remappedOperands = adaptor.getSrc();
    if (remappedOperands.size() != 2)
      return rewriter.notifyMatchFailure(
          splatOp, "expected SplatOp src to have already been remapped");
    Value fatPtrBase = remappedOperands[0];
    Value fatPtrOffset = remappedOperands[1];
    if (!llvm::isa<triton::PointerType>(fatPtrBase.getType()))
      return rewriter.notifyMatchFailure(splatOp,
                                         "non tt.ptr base unimplemented");
    if (!llvm::isa<IntegerType>(fatPtrOffset.getType()))
      return rewriter.notifyMatchFailure(splatOp,
                                         "non-integer offset unimplemented");

    RankedTensorType outType = splatOp.getResult().getType();
    auto newOffsetType = RankedTensorType::get(
        outType.getShape(), fatPtrOffset.getType(), outType.getEncoding());
    triton::SplatOp offset = rewriter.create<triton::SplatOp>(
        splatOp.getLoc(), newOffsetType, fatPtrOffset);
    setRewrittenLegalAttrs(rewriter, splatOp, offset);
    rewriter.replaceOpWithMultiple(splatOp, {{fatPtrBase, offset}});
    fatPtrs[{fatPtrBase, offset}] = fatPtrs[{fatPtrBase, fatPtrOffset}];
    return success();
  }
};

/// Broadcast offset, keep base.
class ConvertBroadcastOp
    : public PointerCanonicalizationPattern<triton::BroadcastOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp broadcastOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange remappedOperands = adaptor.getSrc();
    if (remappedOperands.size() != 2)
      return rewriter.notifyMatchFailure(
          broadcastOp,
          "expected BroadcastOp src to have already been remapped");

    Value fatPtrBase = remappedOperands[0];
    Value fatPtrOffset = remappedOperands[1];
    if (!llvm::isa<triton::PointerType>(fatPtrBase.getType()))
      return rewriter.notifyMatchFailure(broadcastOp,
                                         "non tt.ptr base unimplemented");
    auto offsetType = dyn_cast<RankedTensorType>(fatPtrOffset.getType());
    return rewriter.notifyMatchFailure(broadcastOp,
                                       "non-tensor offset unimplemented");

    auto outType =
        dyn_cast<RankedTensorType>(broadcastOp.getResult().getType());
    auto newOffsetType = RankedTensorType::get(
        outType.getShape(), offsetType.getElementType(), outType.getEncoding());
    triton::BroadcastOp newOffset = rewriter.create<triton::BroadcastOp>(
        broadcastOp.getLoc(), newOffsetType, fatPtrOffset);
    setRewrittenLegalAttrs(rewriter, broadcastOp, newOffset);
    rewriter.replaceOpWithMultiple(broadcastOp, {{fatPtrBase, newOffset}});
    fatPtrs[{fatPtrBase, newOffset}] = fatPtrs[{fatPtrBase, fatPtrOffset}];
    return success();
  }
};

/// Three cases:
/// 1. If it is a scalar pointer update -> bump only the base pointer;
/// 2. Constant tensor offset -> bump only the offset
/// 3. Non-constant tensor offset -> decompose parent(offset) into uniform and
/// non-uniform comop
class ConvertAddPtrOp
    : public PointerCanonicalizationPattern<triton::AddPtrOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp addPtrOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange remappedPtr = adaptor.getPtr();
    if (remappedPtr.size() != 2)
      return rewriter.notifyMatchFailure(
          addPtrOp, "expected AddPtrOp Ptr to have already been remapped");
    ValueRange nonRemappedOffset = adaptor.getOffset();
    if (nonRemappedOffset.size() != 1)
      return rewriter.notifyMatchFailure(
          addPtrOp, "expected AddPtrOp Offset to have not have been remapped");
    Value fatPtrBase = remappedPtr[0];
    Value fatPtrOffset = remappedPtr[1];
    Value origOffset = nonRemappedOffset[0];
    Location curLoc = addPtrOp.getLoc();

    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(addPtrOp);

    // If it is a scalar pointer update, simply bump the base pointer
    if (llvm::isa<triton::PointerType>(addPtrOp.getPtr().getType())) {
      assert(llvm::isa<IntegerType>(origOffset.getType()) &&
             "expected offset to be integer type");
      auto newAddPtrOp = rewriter.create<triton::AddPtrOp>(
          curLoc, fatPtrBase.getType(), fatPtrBase, origOffset);
      setRewrittenLegalAttrs(rewriter, addPtrOp, newAddPtrOp);
      rewriter.replaceOpWithMultiple(addPtrOp, {{newAddPtrOp, fatPtrOffset}});
      fatPtrs[{newAddPtrOp, fatPtrOffset}].canNarrow =
          fatPtrs[{fatPtrBase, fatPtrOffset}].canNarrow;
      return success();
    }

    assert(llvm::isa<RankedTensorType>(addPtrOp.getPtr().getType()) &&
           "expected Ptr to be RankedTensorType type");

    // Early exit for the case of a constant tensor
    if (auto scalarConst =
            maybeGetOrCreateScalarConstant(rewriter, curLoc, origOffset)) {
      triton::AddPtrOp newAddPtrOp = rewriter.create<triton::AddPtrOp>(
          curLoc, fatPtrBase.getType(), fatPtrBase, *scalarConst);
      setRewrittenLegalAttrs(rewriter, addPtrOp, newAddPtrOp);
      rewriter.replaceOpWithMultiple(addPtrOp, {{newAddPtrOp, fatPtrOffset}});
      // If we are updating the tensor pointer with a constant value, we can
      // propagate the attributes of the tensor pointer to the fat pointer.
      fatPtrs[{newAddPtrOp, fatPtrOffset}].canNarrow =
          fatPtrs[{fatPtrBase, fatPtrOffset}].canNarrow;
      return success();
    }

    int64_t bitness = llvm::cast<RankedTensorType>(origOffset.getType())
                          .getElementTypeBitWidth();
    auto [uniformOffset, nonUniformOffset] =
        createDecomposeOffsetFromExpr(rewriter, curLoc, origOffset, bitness);

    auto newAddPtrOp = rewriter.create<triton::AddPtrOp>(
        curLoc, fatPtrBase.getType(), fatPtrBase, uniformOffset);

    // Vector offset update (if any): bump the tensor offset
    bool canNarrow = fatPtrs[{fatPtrBase, fatPtrOffset}].canNarrow;
    bool propagateAtrs = true;
    Value newOffset = fatPtrOffset;
    if (!isZeroConst(nonUniformOffset)) {
      Type addPtrOffsetType = getElementTypeOrSelf(nonUniformOffset);
      Type fatPtrOffsetType = getElementTypeOrSelf(fatPtrOffset);
      // TODO(max): why is this inside this condition?
      canNarrow = canNarrow && canNarrowOffset(fatPtrOffset, nonUniformOffset);
      // Upcast or downcast the offset accordingly
      if (addPtrOffsetType.isInteger(32) && fatPtrOffsetType.isInteger(64))
        nonUniformOffset =
            createExtend32bitOffsetTo64Bits(rewriter, curLoc, nonUniformOffset);
      else if (addPtrOffsetType.isInteger(64) && fatPtrOffsetType.isInteger(32))
        nonUniformOffset =
            createNarrow64bitOffsetTo32bits(rewriter, curLoc, nonUniformOffset);

      newOffset = rewriter.create<arith::AddIOp>(curLoc, nonUniformOffset,
                                                 fatPtrOffset);
      propagateAtrs = false;
    }

    setRewrittenLegalAttrs(rewriter, addPtrOp, newAddPtrOp);
    rewriter.replaceOpWithMultiple(addPtrOp, {{newAddPtrOp, newOffset}});
    auto nextFatPtr = std::pair{newAddPtrOp.getResult(), newOffset};
    fatPtrs[nextFatPtr].canNarrow = canNarrow;
    if (propagateAtrs)
      fatPtrs[nextFatPtr].attributes =
          fatPtrs.at({fatPtrBase, fatPtrOffset}).attributes;

    return success();
  }
};

/// Rewrite init args and result type and bb args.
class ConvertSCFForOp : public PointerCanonicalizationPattern<scf::ForOp> {
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

public:
  LogicalResult
  matchAndRewrite(scf::ForOp forOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<size_t> valRangeLens;
    ArrayRef<ValueRange> remappedInits = adaptor.getInitArgs();
    for (ValueRange remappedInit : remappedInits)
      valRangeLens.push_back(remappedInit.size());

    // rewrite the body bb args
    unsigned inputNo = 0;
    TypeConverter localTypeConverter;
    localTypeConverter.addConversion(
        [&inputNo, remappedInits = adaptor.getInitArgs()](
            Type inputType, SmallVectorImpl<Type> &types) {
          // handle the 0th iv
          if (inputNo == 0) {
            types.append({inputType});
          } else {
            SmallVector<Type> remappedInitTypes =
                llvm::to_vector(remappedInits[inputNo - 1].getTypes());
            types.append(remappedInitTypes);
          }
          inputNo++;
          return success();
        });
    std::optional<TypeConverter::SignatureConversion> conversion =
        localTypeConverter.convertBlockSignature(forOp.getBody());
    if (!conversion)
      return failure();
    auto newBodyBlock = rewriter.applySignatureConversion(
        forOp.getBody(), *conversion, &localTypeConverter);

    // propagate canNarrow to bb arg fatPtrs in for body bb
    // skip iv at index 0
    int offset = 1;
    for (auto operands : remappedInits) {
      if (operands.size() == 2) {
        assert(fatPtrs.contains({operands[0], operands[1]}) &&
               "expected fatPtrs to contain remapped fat pointer");
        fatPtrs[{newBodyBlock->getArgument(offset),
                 newBodyBlock->getArgument(offset + 1)}]
            .canNarrow = fatPtrs[{operands[0], operands[1]}].canNarrow;
      }
      offset += operands.size();
    }

    SmallVector<Value> initArgs = flattenValues(adaptor.getInitArgs());
    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), getSingleValue(adaptor.getLowerBound()),
        getSingleValue(adaptor.getUpperBound()),
        getSingleValue(adaptor.getStep()), initArgs);

    newForOp->setAttrs(forOp->getAttrs());
    rewriter.eraseBlock(newForOp.getBody());
    rewriter.inlineRegionBefore(forOp.getRegion(), newForOp.getRegion(),
                                newForOp.getRegion().end());

    SmallVector<ValueRange> packedRets;
    for (unsigned i = 0, offset = 0; i < valRangeLens.size(); i++) {
      size_t len = valRangeLens[i];
      assert(offset < newForOp->getNumResults() &&
             "expected offset to be within bounds of results");
      ValueRange mappedValue = newForOp->getResults().slice(offset, len);
      packedRets.push_back(mappedValue);
      offset += len;
    }

    setRewrittenLegalAttrs(rewriter, forOp, newForOp);
    rewriter.replaceOpWithMultiple(forOp, packedRets);

    return success();
  }
};

/// Rewrite with new remapped operands but also if the scf.yield is inside of
/// scf.if (possibly) annotate the scf.if.
class ConvertSCFYieldOp : public PointerCanonicalizationPattern<scf::YieldOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp yieldOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ArrayRef<ValueRange> remappedYields = adaptor.getOperands();
    SmallVector<Value> newYieldedValues = flattenValues(remappedYields);
    // have to mutate here because otherwise scf.if, scf.for, and scf.while will
    // get confused about which yield is the "correct" yield (since there will
    // be two of them before the rewriter DCEs)
    rewriter.modifyOpInPlace(yieldOp, [&]() {
      yieldOp.getResultsMutable().clear();
      yieldOp.getResultsMutable().append(newYieldedValues);
    });

    // rewriting a parent op from a child op isn't a great idea but there's no
    // other to indicate to the parent IfOp that the result type can now be
    // rewritten and not before.
    if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
      if (ifOp.thenBlock() == yieldOp->getBlock()) {
        rewriter.modifyOpInPlace(ifOp, [&] {
          ifOp->setDiscardableAttr(kSCFThenRewrittenAttr,
                                   rewriter.getUnitAttr());
        });
      } else {
        rewriter.modifyOpInPlace(ifOp, [&] {
          ifOp->setDiscardableAttr(kSCFElseRewrittenAttr,
                                   rewriter.getUnitAttr());
        });
      }
      // set indices of fatPtrs so that IfOp can propagate canNarrow to
      // result users
      int offset = 0;
      SmallVector<int64_t> fatPtrOffsets;
      for (auto operands : remappedYields) {
        if (operands.size() == 2) {
          assert(fatPtrs.contains({operands[0], operands[1]}) &&
                 "expected fatPtrs to contain remapped fat pointer");
          fatPtrOffsets.push_back(offset);
        }
        offset += operands.size();
      }
      if (!fatPtrOffsets.empty())
        yieldOp->setDiscardableAttr(
            kSCFIfOpYieldFatPtrOffsets,
            rewriter.getDenseI64ArrayAttr(fatPtrOffsets));
    }

    setLegalAttr(rewriter, yieldOp);
    return success();
  }
};

/// Rewrite init_args, result type, before region bb args, after region bb args.
class ConvertSCFWhileOp : public PointerCanonicalizationPattern<scf::WhileOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;
  LogicalResult
  matchAndRewrite(scf::WhileOp whileOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<size_t> valRangeLens;
    ArrayRef<ValueRange> remappedInits = adaptor.getInits();
    for (ValueRange remappedInit : remappedInits)
      valRangeLens.push_back(remappedInit.size());

    // rewrite the "before" block bb args
    unsigned inputNo = 0;
    TypeConverter localTypeConverter;
    localTypeConverter.addConversion(
        [&inputNo, remappedInits](Type _inputType,
                                  SmallVectorImpl<Type> &types) {
          SmallVector<Type> remappedInitTypes =
              llvm::to_vector(remappedInits[inputNo].getTypes());
          types.append(remappedInitTypes);
          inputNo++;
          return success();
        });
    std::optional<TypeConverter::SignatureConversion> conversion =
        localTypeConverter.convertBlockSignature(whileOp.getBeforeBody());
    if (!conversion)
      return failure();
    auto newBeforeBodyBlock = rewriter.applySignatureConversion(
        whileOp.getBeforeBody(), *conversion, &localTypeConverter);

    auto propagateCanNarrowToBlock = [remappedInits, this](Block *block) {
      int offset = 0;
      for (auto operands : remappedInits) {
        if (operands.size() == 2) {
          assert(fatPtrs.contains({operands[0], operands[1]}) &&
                 "expected fatPtrs to contain remapped fat pointer");
          fatPtrs[{block->getArgument(offset), block->getArgument(offset + 1)}]
              .canNarrow = fatPtrs[{operands[0], operands[1]}].canNarrow;
        }
        offset += operands.size();
      }
    };

    // propagate canNarrow to bb arg fatPtrs in before bb
    propagateCanNarrowToBlock(newBeforeBodyBlock);

    // rewrite the "after" block bb args
    conversion =
        localTypeConverter.convertBlockSignature(whileOp.getAfterBody());
    if (!conversion)
      return failure();
    auto newAfterBodyBlock = rewriter.applySignatureConversion(
        whileOp.getAfterBody(), *conversion, &localTypeConverter);

    // propagate canNarrow to bb arg fatPtrs in after bb
    propagateCanNarrowToBlock(newAfterBodyBlock);

    SmallVector<Value> initArgs = flattenValues(remappedInits);
    SmallVector<Type> resultTypes =
        llvm::map_to_vector(initArgs, [](Value v) { return v.getType(); });
    auto newWhileOp =
        rewriter.create<scf::WhileOp>(whileOp.getLoc(), resultTypes, initArgs);

    newWhileOp->setAttrs(whileOp->getAttrs());
    rewriter.inlineRegionBefore(whileOp.getBefore(), newWhileOp.getBefore(),
                                newWhileOp.getBefore().end());
    rewriter.inlineRegionBefore(whileOp.getAfter(), newWhileOp.getAfter(),
                                newWhileOp.getAfter().end());

    SmallVector<ValueRange> packedRets;
    for (unsigned i = 0, offset = 0; i < valRangeLens.size(); i++) {
      size_t len = valRangeLens[i];
      assert(offset < newWhileOp->getNumResults() &&
             "expected offset to be within bounds of results");
      ValueRange mappedValue = newWhileOp->getResults().slice(offset, len);
      packedRets.push_back(mappedValue);
      offset += len;
    }

    setRewrittenLegalAttrs(rewriter, whileOp, newWhileOp);
    rewriter.replaceOpWithMultiple(whileOp, packedRets);

    return success();
  }
};

/// Rewrite with new operands.
class ConvertSCFConditionOp
    : public PointerCanonicalizationPattern<scf::ConditionOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;
  LogicalResult
  matchAndRewrite(scf::ConditionOp condOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> newArgs = flattenValues(adaptor.getArgs());
    // have to mutate here because otherwise scf.while will
    // get confused about which condition is the "correct" condition (since
    // there will be two of them before the rewriter DCEs)
    rewriter.modifyOpInPlace(condOp, [&]() {
      condOp.getArgsMutable().clear();
      condOp.getArgsMutable().append(newArgs);
    });
    setLegalAttr(rewriter, condOp);
    return success();
  }
};

/// Rewrite operands for both true dest and false dest.
class ConvertCFCondBranch
    : public PointerCanonicalizationPattern<cf::CondBranchOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;
  LogicalResult
  matchAndRewrite(cf::CondBranchOp branchOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ArrayRef<ValueRange> remappedTrueOperands = adaptor.getTrueDestOperands();
    ArrayRef<ValueRange> remappedFalseOperands = adaptor.getFalseDestOperands();
    SmallVector<Value> trueOperands = flattenValues(remappedTrueOperands);
    SmallVector<Value> falseOperands = flattenValues(remappedFalseOperands);

    setRewrittenAttr(rewriter, branchOp);
    auto newBrancOp = rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        branchOp, branchOp.getCondition(), branchOp.getTrueDest(), trueOperands,
        branchOp.getFalseDest(), falseOperands);

    // can't put inputNo inside because of limited lifetime (it'll be popped
    // from stack memory after lambda returns...)
    auto makeTypeConv = [](unsigned &inputNo,
                           ArrayRef<ValueRange> remappedOperands) {
      return [&inputNo, remappedOperands](Type inputType,
                                          SmallVectorImpl<Type> &types) {
        SmallVector<Type> remappedInitTypes =
            llvm::to_vector(remappedOperands[inputNo].getTypes());
        types.append(remappedInitTypes);
        inputNo++;
        return success();
      };
    };

    auto propagateCanNarrowToBlock = [this](Block *block,
                                            ArrayRef<ValueRange>
                                                remappedOperands) {
      int offset = 0;
      for (auto operands : remappedOperands) {
        if (operands.size() == 2) {
          assert(fatPtrs.contains({operands[0], operands[1]}) &&
                 "expected fatPtrs to contain remapped fat pointer");
          fatPtrs[{block->getArgument(offset), block->getArgument(offset + 1)}]
              .canNarrow = fatPtrs[{operands[0], operands[1]}].canNarrow;
        }
        offset += operands.size();
      }
    };

    // convert the type signature of the true dest bb
    TypeConverter localTypeConverterTrueDest;
    unsigned inputNo = 0;
    localTypeConverterTrueDest.addConversion(
        makeTypeConv(inputNo, remappedTrueOperands));
    std::optional<TypeConverter::SignatureConversion> conversion =
        localTypeConverterTrueDest.convertBlockSignature(
            branchOp.getTrueDest());
    if (!conversion)
      return failure();
    auto newTrueBlock = rewriter.applySignatureConversion(
        branchOp.getTrueDest(), *conversion, &localTypeConverterTrueDest);

    // propagate canNarrow to bb arg fatPtrs in true bb
    propagateCanNarrowToBlock(newTrueBlock, remappedTrueOperands);

    // convert the type signature of the false dest bb
    inputNo = 0;
    TypeConverter localTypeConverterFalseDest;
    localTypeConverterFalseDest.addConversion(
        makeTypeConv(inputNo, remappedFalseOperands));
    conversion = localTypeConverterFalseDest.convertBlockSignature(
        branchOp.getFalseDest());
    if (!conversion)
      return failure();
    auto newFalseBlock = rewriter.applySignatureConversion(
        branchOp.getFalseDest(), *conversion, &localTypeConverterFalseDest);

    // propagate canNarrow to bb arg fatPtrs in false bb
    propagateCanNarrowToBlock(newFalseBlock, remappedFalseOperands);

    setLegalAttr(rewriter, newBrancOp);
    return success();
  }
};

/// Rewrite both operands. Note, this should only be reached after both
/// operands have already been rewritten because DialectConversion walks
/// PreOrder in order ForwardDominance order: see
/// https://github.com/llvm/llvm-project/blob/58389b220a9354ed6c34bdb9310a35165579c5e3/mlir/lib/Transforms/Utils/DialectConversion.cpp#L2702
class ConvertArithSelectOp
    : public PointerCanonicalizationPattern<arith::SelectOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;
  LogicalResult
  matchAndRewrite(arith::SelectOp selectOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ArrayRef<ValueRange> remappedOperands = adaptor.getOperands();
    if (remappedOperands[1].size() != 2 || remappedOperands[2].size() != 2)
      return rewriter.notifyMatchFailure(
          selectOp, "expected adaptor to have had both true and false operands "
                    "already remapped");
    // If both have been traversed, then we can rewrite select of pointers as a
    // select of base and offset
    ValueRange fatPtrTrue = remappedOperands[1];
    ValueRange fatPtrFalse = remappedOperands[2];
    // Simple case of a scalar select: update the base pointer
    if (!isa<RankedTensorType>(selectOp.getType())) {
      auto newSelectOp = rewriter.create<arith::SelectOp>(
          selectOp.getLoc(), selectOp.getType(),
          // TODO(max): why fatPtrTrue here?
          selectOp.getCondition(), fatPtrTrue[0], selectOp.getFalseValue());
      setRewrittenLegalAttrs(rewriter, selectOp, newSelectOp);
      rewriter.replaceOpWithMultiple(selectOp, {{newSelectOp, fatPtrTrue[1]}});
      return success();
    }

    // Rewrite to select(fatBaseT, fatBaseF) and select(fatOffsetT, fatOffsetF)
    auto newBase = rewriter.create<arith::SelectOp>(
        selectOp.getLoc(), selectOp.getCondition(), fatPtrTrue[0],
        fatPtrFalse[0]);
    auto newOffset = rewriter.create<arith::SelectOp>(
        selectOp.getLoc(), selectOp.getCondition(), fatPtrTrue[1],
        fatPtrFalse[1]);

    assert((fatPtrs[{fatPtrTrue[0], fatPtrTrue[1]}].canNarrow ==
            fatPtrs[{fatPtrFalse[0], fatPtrFalse[1]}].canNarrow) &&
           "expected can narrow to be the same for both fatPtrT and fatPtrF");

    setRewrittenLegalAttrs(rewriter, selectOp, newBase);
    setRewrittenLegalAttrs(rewriter, selectOp, newOffset);
    rewriter.replaceOpWithMultiple(selectOp, {{newBase, newOffset}});
    fatPtrs[{newBase, newOffset}].canNarrow =
        fatPtrs[{fatPtrTrue[0], fatPtrTrue[1]}].canNarrow;

    return success();
  }
};

/// Rewrite result type only after both arms have been visited.
/// We contrive this to happen, even though DialectConversion does a PreOrder
/// walk, by checking for two attributes in the ConversionTarget
/// ("then_rewritten", and "else_rewritten").
class ConvertSCFIfOp : public PointerCanonicalizationPattern<scf::IfOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;
  LogicalResult
  matchAndRewrite(scf::IfOp ifOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(ifOp.thenYield()->hasAttr(kSCFIfOpYieldFatPtrOffsets) &&
           "expected then yield to report fat ptr indices");

    bool withElseRegion = ifOp.getNumRegions() > 1;

#ifndef NDEBUG
    if (withElseRegion) {
      assert(ifOp.thenYield().getOperandTypes() ==
                 ifOp.elseYield().getOperandTypes() &&
             "ifOp types must match in both arms");
      assert(ifOp.elseYield()->hasAttr(kSCFIfOpYieldFatPtrOffsets) &&
             "expected then yield to report fat ptr indices");
      if (auto thenFatPtrIndxs = ifOp.thenYield()->getDiscardableAttr(
              kSCFIfOpYieldFatPtrOffsets)) {
        auto elseFatPtrIndx =
            ifOp.elseYield()->getDiscardableAttr(kSCFIfOpYieldFatPtrOffsets);
        assert(elseFatPtrIndx &&
               "expected else fat ptr indices as well as then fat ptr indices");
        for (auto [i, j] : llvm::zip(
                 llvm::cast<DenseI64ArrayAttr>(thenFatPtrIndxs).asArrayRef(),
                 llvm::cast<DenseI64ArrayAttr>(elseFatPtrIndx).asArrayRef())) {
          assert(i == j &&
                 "expected thenFatPtrIndxs and elseFatPtrIndxs to agree");
          assert(i < ifOp.thenYield().getNumOperands() &&
                 i + 1 < ifOp.thenYield().getNumOperands() &&
                 "expected idx to be within bounds of IfOp's results");
          Value thenFatPtrBase = ifOp.thenYield().getOperand(i);
          Value thenFatPtrOffset = ifOp.thenYield().getOperand(i + 1);
          Value elseFatPtrBase = ifOp.elseYield().getOperand(i);
          Value elseFatPtrOffset = ifOp.elseYield().getOperand(i + 1);
          assert((fatPtrs[{thenFatPtrBase, thenFatPtrOffset}].canNarrow ==
                  fatPtrs[{elseFatPtrBase, elseFatPtrOffset}].canNarrow) &&
                 "expected then fat ptr canNarrow and else fat ptr canNarrow "
                 "to be equal");
        }
      }
    }
#endif

    auto newIfOp = rewriter.create<scf::IfOp>(
        ifOp.getLoc(), ifOp.thenYield().getOperandTypes(), ifOp.getCondition(),
        withElseRegion);
    rewriter.inlineBlockBefore(ifOp.thenBlock(), newIfOp.thenBlock(),
                               newIfOp.thenBlock()->begin());
    if (withElseRegion)
      rewriter.inlineBlockBefore(ifOp.elseBlock(), newIfOp.elseBlock(),
                                 newIfOp.elseBlock()->begin());

    setRewrittenLegalAttrs(rewriter, ifOp, newIfOp);
    rewriter.replaceOpWithMultiple(ifOp, {newIfOp.getResults()});

    for (int64_t idx :
         llvm::cast<DenseI64ArrayAttr>(newIfOp.thenYield()->getDiscardableAttr(
                                           kSCFIfOpYieldFatPtrOffsets))
             .asArrayRef()) {
      Value thenFatPtrBase = newIfOp.thenYield().getOperand(idx);
      Value thenFatPtrOffset = newIfOp.thenYield().getOperand(idx + 1);
      fatPtrs[{newIfOp.getResult(idx), newIfOp.getResult(idx + 1)}].canNarrow =
          fatPtrs[{thenFatPtrBase, thenFatPtrOffset}].canNarrow;
    }

    return success();
  }
};

/// Rewrite the non-cond operands and the signature of the dest bb.
class ConvertCFBranch : public PointerCanonicalizationPattern<cf::BranchOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;
  LogicalResult
  matchAndRewrite(cf::BranchOp branchOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ArrayRef<ValueRange> remappedDestOperands = adaptor.getDestOperands();
    SmallVector<Value> trueOperands = flattenValues(remappedDestOperands);

    setRewrittenAttr(rewriter, branchOp);
    auto newBrancOp = rewriter.replaceOpWithNewOp<cf::BranchOp>(
        branchOp, branchOp.getDest(), trueOperands);

    unsigned inputNo = 0;
    TypeConverter localTypeConverterTrueDest;
    localTypeConverterTrueDest.addConversion(
        [&inputNo, remappedDestOperands](Type _inputType,
                                         SmallVectorImpl<Type> &types) {
          SmallVector<Type> remappedInitTypes =
              llvm::to_vector(remappedDestOperands[inputNo].getTypes());
          types.append(remappedInitTypes);
          inputNo++;
          return success();
        });
    std::optional<TypeConverter::SignatureConversion> conversion =
        localTypeConverterTrueDest.convertBlockSignature(branchOp.getDest());
    if (!conversion)
      return failure();
    auto newDestBlock = rewriter.applySignatureConversion(
        branchOp.getDest(), *conversion, &localTypeConverterTrueDest);

    int offset = 0;
    for (auto operands : remappedDestOperands) {
      if (operands.size() == 2) {
        assert(fatPtrs.contains({operands[0], operands[1]}) &&
               "expected fatPtrs to contain remapped fat pointer");
        fatPtrs[{newDestBlock->getArgument(offset),
                 newDestBlock->getArgument(offset + 1)}]
            .canNarrow = fatPtrs[{operands[0], operands[1]}].canNarrow;
      }
      offset += operands.size();
    }

    setLegalAttr(rewriter, newBrancOp);
    return success();
  }
};

class ConvertLoadOp : public PointerCanonicalizationPattern<triton::LoadOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp loadOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange fatPtr = adaptor.getPtr();
    if (fatPtr.size() != 2)
      return rewriter.notifyMatchFailure(
          loadOp, "expected LoadOp ptr to have already been remapped");
    Value fatPtrBase = fatPtr[0];
    Value fatPtrOffset = fatPtr[1];
    Location curLoc = loadOp.getLoc();

    llvm::SmallDenseMap<StringAttr, Attribute> attributes{
        {rewriter.getStringAttr(kLegalAttr), rewriter.getUnitAttr()}};
    Value newPtr = fatPtrBase;
    if (llvm::isa<RankedTensorType>(loadOp.getPtr().getType()))
      newPtr = createTensorPointer(
          rewriter, fatPtrBase, fatPtrOffset, curLoc,
          fatPtrs[{fatPtrBase, fatPtrOffset}].canNarrow, attributes);
    SmallVector<Value> operands =
        loadOp.getOperands().take_back(loadOp.getNumOperands() - 1);
    operands.insert(operands.begin(), newPtr);
    SmallVector<NamedAttribute> attrs = llvm::to_vector(loadOp->getAttrs());
    attrs.append({rewriter.getNamedAttr(kLegalAttr, rewriter.getUnitAttr())});
    auto newLoadPtrOp =
        rewriter.replaceOpWithNewOp<triton::LoadOp>(loadOp, operands, attrs);
    setLegalAttr(rewriter, newLoadPtrOp);
    return success();
  }
};

class ConvertStoreOp : public PointerCanonicalizationPattern<triton::StoreOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp storeOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange fatPtr = adaptor.getPtr();
    if (fatPtr.size() != 2)
      return rewriter.notifyMatchFailure(
          storeOp, "expected StoreOp ptr to have already been remapped");
    Value fatPtrBase = fatPtr[0];
    Value fatPtrOffset = fatPtr[1];
    Location curLoc = storeOp.getLoc();

    llvm::SmallDenseMap<StringAttr, Attribute> attributes{
        {rewriter.getStringAttr(kLegalAttr), rewriter.getUnitAttr()}};

    Value newPtr = fatPtrBase;
    if (llvm::isa<RankedTensorType>(storeOp.getPtr().getType()))
      newPtr = createTensorPointer(
          rewriter, fatPtrBase, fatPtrOffset, curLoc,
          fatPtrs[{fatPtrBase, fatPtrOffset}].canNarrow, attributes);
    SmallVector<Value> operands =
        storeOp.getOperands().take_back(storeOp.getNumOperands() - 1);
    operands.insert(operands.begin(), newPtr);
    SmallVector<NamedAttribute> attrs = llvm::to_vector(storeOp->getAttrs());
    attrs.append({rewriter.getNamedAttr(kLegalAttr, rewriter.getUnitAttr())});
    auto newStoreOp = rewriter.replaceOpWithNewOp<triton::StoreOp>(
        storeOp, TypeRange{}, ValueRange{operands}, attrs);
    setLegalAttr(rewriter, newStoreOp);
    return success();
  }
};

/// tt.func gets rewritten differently from all of the other ops - the op itself
/// is not rewritten but all tt.ptr args are rewritten (all uses) to be
/// %1 = unrealize_cast(%arg0: tt.ptr, c0: i32) -> tt.ptr.
/// This unrealized_cast remains through out the first pass of the dialect
/// conversion and is then materialized in the second pass
/// (ConvertUnrealizedConversionCastOp).
class ConvertFuncOp : public PointerCanonicalizationPattern<triton::FuncOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int64_t bitness = 64;
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    rewriter.modifyOpInPlace(funcOp, [&] {
      for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
        // The pointer argument needs to be a scalar
        if (!isa<triton::PointerType>(arg.getType()))
          continue;
        if (auto pointerRangeAttr =
                funcOp.getArgAttrOfType<IntegerAttr>(idx, "tt.pointer_range"))
          bitness = pointerRangeAttr.getInt();
        Value zeroOffset =
            rewriter.create<arith::ConstantIntOp>(funcOp.getLoc(), 0, bitness);
        auto dummyCast = rewriter.create<UnrealizedConversionCastOp>(
            arg.getLoc(), TypeRange{arg.getType()}, ValueRange{arg});
        rewriter.replaceUsesOfBlockArgument(arg, dummyCast.getResult(0));
        // TODO(max): why is this true?
        fatPtrs[{arg, zeroOffset}].canNarrow = true;
        rewriter.replaceOpWithMultiple(dummyCast, {{arg, zeroOffset}});
      }
    });
    setRewrittenAttr(rewriter, funcOp);

    return success();
  }
};

/// Rewrite %1 = unrealize_cast(%arg0: tt.ptr, c0: i32) -> tt.ptr inserted by
/// ConvertFuncOp to be just %arg0: tt.ptr.
class ConvertUnrealizedConversionCastOp
    : public PointerCanonicalizationPattern<UnrealizedConversionCastOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(std::distance(castOp->getUses().begin(), castOp->getUses().end()) >
               0 &&
           "expected at least 1 use of unrealized_cast");
    // dunno why but i get -Wdangling here...
    ArrayRef<ValueRange> remappedOperands = adaptor.getOperands();
    if (remappedOperands.size() != 1 || remappedOperands[0].size() != 2)
      return rewriter.notifyMatchFailure(
          castOp, "expected CastOp to have already been remapped");
    Value fatPtrBase = remappedOperands[0][0];
    Value fatPtrOffset = remappedOperands[0][1];
    if (!llvm::isa<triton::PointerType>(fatPtrBase.getType()))
      return rewriter.notifyMatchFailure(castOp,
                                         "non tt.ptr base unimplemented");
    if (!llvm::isa<IntegerType>(fatPtrOffset.getType()))
      return rewriter.notifyMatchFailure(castOp,
                                         "non-integer offset unimplemented");
    OpFoldResult maybeScalar = getAsOpFoldResult(fatPtrOffset);
    auto integerAttr = llvm::dyn_cast<mlir::Attribute>(maybeScalar);
    if (!integerAttr || !llvm::isa<IntegerAttr>(integerAttr) ||
        llvm::cast<IntegerAttr>(integerAttr).getValue() != 0)
      return rewriter.notifyMatchFailure(
          castOp, "CastOp should have been inserted by ConvertFuncOp and "
                  "should have constant integer offset=0");

    rewriter.replaceAllUsesWith(castOp.getResult(0), fatPtrBase);
    rewriter.eraseOp(castOp);
    return success();
  }
};

void TritonAMDGPUCanonicalizePointersPass::runOnOperation() {
  ModuleOp module = getOperation();
  mlir::MLIRContext *context = &getContext();
  ConversionTarget target(*context);
  RewritePatternSet patterns(context);
  auto isLegal = [](Operation *op) {
    if (op->hasAttr(kRewrittenAttr) || op->hasAttr(kLegalAttr))
      return true;
    for (OpOperand &operand : op->getOpOperands()) {
      if (auto arg = llvm::dyn_cast<BlockArgument>(operand.get())) {
        if (!llvm::isa<triton::PointerType>(getElementTypeOrSelf(arg)))
          continue;
        return false;
      }
      if (operand.get().getDefiningOp()->hasAttr(kRewrittenAttr))
        return false;
    }
    return true;
  };
  target.addDynamicallyLegalDialect<triton::TritonDialect>(
      [&isLegal](Operation *op) {
        if (llvm::isa<triton::FuncOp>(op) && !op->hasAttr(kRewrittenAttr))
          return false;
        return isLegal(op);
      });
  target.addDynamicallyLegalDialect<scf::SCFDialect>([&isLegal](Operation *op) {
    if (auto ifOp = llvm::dyn_cast<scf::IfOp>(op))
      return !(ifOp->hasAttr(kSCFThenRewrittenAttr) and
               ifOp->hasAttr(kSCFElseRewrittenAttr));
    if (llvm::isa<scf::ConditionOp>(op) && !op->hasAttr(kLegalAttr))
      return false;
    return isLegal(op);
  });
  target.addDynamicallyLegalDialect<cf::ControlFlowDialect>(
      [&isLegal](Operation *op) { return isLegal(op); });
  target.addDynamicallyLegalDialect<arith::ArithDialect>(
      [&isLegal](Operation *op) {
        if (llvm::isa<arith::SelectOp>(op))
          return isLegal(op);
        return true;
      });

  FatPointers fatPrs;

  patterns
      .add<ConvertFuncOp, ConvertBroadcastOp, ConvertSplatOp, ConvertAddPtrOp,
           ConvertLoadOp, ConvertStoreOp, ConvertSCFForOp, ConvertSCFYieldOp,
           ConvertSCFIfOp, ConvertSCFConditionOp, ConvertSCFWhileOp,
           ConvertCFCondBranch, ConvertCFBranch, ConvertArithSelectOp>(
          patterns.getContext(), fatPrs);
  ConversionConfig config;
  config.buildMaterializations = false;
  if (failed(
          applyPartialConversion(module, target, std::move(patterns), config)))
    return signalPassFailure();

  patterns.clear();
  target.addIllegalOp<UnrealizedConversionCastOp>();
  patterns.add<ConvertUnrealizedConversionCastOp>(patterns.getContext(),
                                                  fatPrs);
  if (failed(
          applyPartialConversion(module, target, std::move(patterns), config)))
    return signalPassFailure();

  module.walk<WalkOrder::PreOrder>([](Operation *op) {
    for (auto attr : op->getDiscardableAttrs()) {
      if (attr.getName().strref().starts_with(kPtrCanonPrefix))
        op->removeDiscardableAttr(attr.getName());
    }
  });
}

std::unique_ptr<Pass> mlir::createTritonAMDGPUCanonicalizePointersPass() {
  return std::make_unique<TritonAMDGPUCanonicalizePointersPass>();
}
