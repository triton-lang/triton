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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
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
class PointerCanonicalizer {
public:
  explicit PointerCanonicalizer(ModuleOp moduleOp)
      : rewriter(moduleOp.getContext()), mod(moduleOp) {}

  // Propagate fat pointers in all the functions of the module
  LogicalResult run();

  // A fat pointer is represented as `basePtr + offset` internally.
  struct FatPtr {
    // Scalar base pointer. Needs to be `tt.splat`ed before used
    Value basePtr;
    // Tensor offset
    Value offset;
    // Flag to express if we can narrow the uses of the offset down to 32 bits
    bool canNarrow = false;
    // Collection of attributes that need to be applied to the pointer
    llvm::SmallDenseMap<StringAttr, Attribute> attributes{};

    // Utility copy functions
    FatPtr copy(Value newBasePtr, Value newOffset) {
      return FatPtr{newBasePtr, newOffset, canNarrow};
    }
    FatPtr copyWithBase(Value newOffset) {
      return FatPtr{basePtr, newOffset, canNarrow};
    }
    FatPtr copyWithOffset(Value newBase) {
      return FatPtr{newBase, offset, canNarrow};
    }
    // Attribute functions
    void setAttr(StringAttr name, Attribute value) {
      attributes.insert({name, value});
    }
    void setAttr(NamedAttribute attr) {
      attributes.insert({attr.getName(), attr.getValue()});
    }
    void setAttrs(ArrayRef<NamedAttribute> attrs) {
      for (auto attr : attrs)
        attributes.insert({attr.getName(), attr.getValue()});
    }
  };

  // Rewrite any operation that needs a pointer
  LogicalResult materializeFatPointer(Operation *op, Location loc, Value ptr);

  // Start from an argument of a function and propagate its fat pointers
  LogicalResult rewritePointer(Value argPtr);

  // Create a tensor pointer from a fat pointer `fatPtr`. The tensor pointer is
  // obtained by splatting the `fatPtr.basePtr` using the `fatPtr.offset` shape
  // and adding the offset to it.

  // Push the attributes of the given operation `op` to the fat pointer
  // corresponding to `val`
  void collectFatPointerAttributes(Operation *op, Value val);

  // Rewrite a given function, canonicalizing the different pointer arguments of
  // the region
  LogicalResult rewriteFunction(triton::FuncOp funcOp);

  // Rewriters for different operation a pointer can walk into
  LogicalResult rewriteSplatOp(triton::SplatOp splatOp, Location curLoc,
                               Value &nextPtr);
  LogicalResult rewriteBroadcastOp(triton::BroadcastOp broadcastOp,
                                   Location curLoc, Value &nextPtr);
  LogicalResult rewriteAddPtrOp(triton::AddPtrOp addPtrOp, Location curLoc,
                                Value &nextPtr);
  LogicalResult rewriteForOp(scf::ForOp forOp, Location curLoc,
                             OpOperand *operand, Value &nextPtr);
  LogicalResult rewriteYieldOp(scf::YieldOp yieldOp, Location curLoc,
                               OpOperand *operand, Value &nextPtr);
  LogicalResult rewriteWhileOp(scf::WhileOp whileOp, Location curLoc,
                               OpOperand *operand, Value &nextPtr);
  LogicalResult rewriteConditionOp(scf::ConditionOp conditionOp,
                                   Location curLoc, OpOperand *operand,
                                   Value &nextPtr);
  LogicalResult rewriteCondBranchOp(cf::CondBranchOp condBrOp, Location curLoc,
                                    OpOperand *operand, Value &nextPtr);
  LogicalResult rewriteSelectOp(arith::SelectOp selectOp, Location curLoc,
                                OpOperand *operand, Value &nextPtr);
  LogicalResult rewriteBranchOp(cf::BranchOp branchOp, Location curLoc,
                                OpOperand *operand, Value &nextPtr);

  // Perform simplified scalar extraction. An offset can be composed by Unifrom
  // (U) and non-uniform(N) components. A uniform component is basically a
  // tensor constant (or a splat). A NonUniform value is a `make_range` or
  // whatever we multiply with a `make_range` operation. We consider the generic
  // expressions:
  //   offset = (N+U)*(N+U)
  //
  // Where the `uniformOffset=U*U` and the `nonUniformOffset=(N*U+U*N+N*N).
  //
  // We do not consider any expression not involving * and +.
  //
  // The function accepts the `rewriter`, the `location` and start recursing at
  // the given `expr`.
  //
  // We also pass the bitness of the offset.
  //
  // The function returns the two components of the given offset as a
  // std::pair{U, NU}
  // std::pair<Value, Value> decomposeOffsetFromExpr(Location loc, Value expr,
  //                                                 int64_t bitness);
  // std::pair<Value, Value> decomposeOffsetFromAdd(Location loc, Value expr,
  //                                                int64_t bitness);
  // std::pair<Value, Value> decomposeOffsetFromMul(Location loc, Value expr,
  //                                                int64_t bitness);

  // Return either the operation or its rewritten op
  template <typename OpTy>
  OpTy resolveOp(Operation *op,
                 const DenseMap<Operation *, Operation *> &rewriteOpMap) {
    OpTy resolvedOp = dyn_cast<OpTy>(op);
    if (rewriteOpMap.contains(op))
      resolvedOp = dyn_cast<OpTy>(rewriteOpMap.at(op));
    return resolvedOp;
  }

  mlir::IRRewriter rewriter;
  ModuleOp mod;

  // Symbol table: association between pointers and fatPointers
  llvm::MapVector<Value, FatPtr> pointers;

  void clearFunctionState() {
    rewriteOpMap.clear();
    queue.clear();
    opToDelete.clear();
  }

  // This structure is used to point to the right operation during the traversal
  // of a function
  DenseMap<Operation *, Operation *> rewriteOpMap;

  // Queue of operations to visit in the current function
  SmallVector<OpOperand *> queue;

  // List of IR to delete in the current function
  SetVector<Operation *> opToDelete;
};

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
Value getScalarConstant(RewriterBase &rewriter, Location loc, Value expr) {
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

  return Value();
}

// Narrowing logic
// For now we allow to narrow down to 32 bits only in the following case:
// - `baseOffset` is 32-bits and `addOffset`(64-bits) is zero
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

void PointerCanonicalizer::collectFatPointerAttributes(Operation *op,
                                                       Value val) {
  auto addBlockArgumentAttr = [&](BlockArgument arg) {
    // If the value is a block parameter, the operation can specify
    // an attribute for the given parameter by using `tt.property_argi`
    // where `argi` refers to the arg number of the given parameter.
    // So we need to iterate through the property, find the right one
    // and push the property onto the pointers attributes.
    llvm::SmallString<8> scratchStr;
    for (NamedAttribute namedAttr : op->getAttrs()) {
      scratchStr.clear();
      llvm::raw_svector_ostream sstream(scratchStr);
      sstream << "_arg" << arg.getArgNumber();
      StringRef attrName = namedAttr.getName().getValue();
      if (attrName.ends_with(scratchStr)) {
        StringRef newAttrName = attrName.drop_back(scratchStr.size());
        namedAttr.setName(rewriter.getStringAttr(newAttrName));
        pointers[val].setAttr(namedAttr);
        // Propagate the argument to the offset if it is also a block argument
        if (auto offsetArg = dyn_cast<BlockArgument>(pointers[val].offset)) {
          scratchStr.clear();
          sstream << newAttrName << "_arg" << offsetArg.getArgNumber();
          op->setAttr(scratchStr, namedAttr.getValue());
        }
      }
    }
  };

  // If it is the i-th block argument, then look if the operation defined some
  // _argi attribute and add it to the fat pointer attributes
  if (auto arg = dyn_cast<BlockArgument>(val)) {
    addBlockArgumentAttr(arg);
    return;
  }

  // Otherwise add the attributes of the operation to the fat pointer
  for (NamedAttribute attr : op->getAttrs())
    pointers[val].setAttr(attr);
}

std::pair<Value, Value> decomposeOffsetFromExpr(RewriterBase &rewriter,
                                                Location loc, Value expr,
                                                int64_t bitness);
// Offset extraction logic for an addition op:
// decompose(A+B) = {U(A)+U(B), NU(A)+NU(B)}
std::pair<Value, Value> decomposeOffsetFromAdd(RewriterBase &rewriter,
                                               Location loc, Value expr,
                                               int64_t bitness) {
  auto addOp = expr.getDefiningOp<arith::AddIOp>();
  auto [uniformOffsetL, nonUniformOffsetL] =
      decomposeOffsetFromExpr(rewriter, loc, addOp.getLhs(), bitness);
  auto [uniformOffsetR, nonUniformOffsetR] =
      decomposeOffsetFromExpr(rewriter, loc, addOp.getRhs(), bitness);
  Value uniformAdd =
      rewriter.create<arith::AddIOp>(loc, uniformOffsetL, uniformOffsetR);
  Value nonUniformAdd =
      rewriter.create<arith::AddIOp>(loc, nonUniformOffsetL, nonUniformOffsetR);
  return {uniformAdd, nonUniformAdd};
}

// Offset extraction logic for a multiplication op:
// decompose(A*B) = {U(A)*U(B), NU(A)*NU(B)+NU(B)*U(A)+U(A)*NU(B)}
std::pair<Value, Value> decomposeOffsetFromMul(RewriterBase &rewriter,
                                               Location loc, Value expr,
                                               int64_t bitness) {
  auto mulOp = expr.getDefiningOp<arith::MulIOp>();
  auto [uniformOffsetL, nonUniformOffsetL] =
      decomposeOffsetFromExpr(rewriter, loc, mulOp.getLhs(), bitness);
  auto [uniformOffsetR, nonUniformOffsetR] =
      decomposeOffsetFromExpr(rewriter, loc, mulOp.getRhs(), bitness);
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

std::pair<Value, Value> decomposeOffsetFromExpr(RewriterBase &rewriter,
                                                Location loc, Value expr,
                                                int64_t bitness) {

  // RewriterBase::InsertionGuard guard(rewriter);
  // rewriter.setInsertionPointAfterValue(expr);

  // Base case 1: it is a splat. Return the scalar constant as the uniform part
  if (Value scalarConst = getScalarConstant(rewriter, loc, expr)) {
    auto tensorZero =
        createTensorZero(rewriter, loc, cast<RankedTensorType>(expr.getType()));
    return {scalarConst, tensorZero};
  }

  // Base case 2: block argument. Since it is not a scalar constant, it must be
  // a tensor. Note that this means we won't be able to decompose across loop
  // boundaries (TODO: giuseros).
  if (auto blockArg = dyn_cast<BlockArgument>(expr)) {
    Value scalarZero = rewriter.create<arith::ConstantIntOp>(loc, 0, bitness);
    return std::make_pair(scalarZero, expr);
  }

  auto offsets =
      llvm::TypeSwitch<Operation *, std::pair<Value, Value>>(
          expr.getDefiningOp())
          .Case<triton::BroadcastOp>([&](auto broadcastOp) {
            auto [uniform, nonUniform] = decomposeOffsetFromExpr(
                rewriter, loc, broadcastOp.getSrc(), bitness);
            auto broadcastNonUniform = rewriter.create<triton::BroadcastOp>(
                loc, broadcastOp.getType(), nonUniform);
            return std::make_pair(uniform, broadcastNonUniform);
          })
          .Case<triton::ExpandDimsOp>([&](auto expandOp) {
            auto [uniform, nonUniform] = decomposeOffsetFromExpr(
                rewriter, loc, expandOp.getSrc(), bitness);
            auto expandNonUniform = rewriter.create<triton::ExpandDimsOp>(
                loc, nonUniform, expandOp.getAxis());
            return std::make_pair(uniform, expandNonUniform);
          })
          .Case<arith::AddIOp>([&](Operation *op) {
            return decomposeOffsetFromAdd(rewriter, loc, expr, bitness);
          })
          .Case<arith::MulIOp>([&](Operation *op) {
            return decomposeOffsetFromMul(rewriter, loc, expr, bitness);
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

  Value tensorPtr = rewriter.create<triton::SplatOp>(
      loc, TypeRange{tensorPtrType}, ValueRange{basePtr},
      SmallVector{rewriter.getNamedAttr("legal", rewriter.getUnitAttr())});

  auto addPtrOp =
      rewriter.create<triton::AddPtrOp>(loc, tensorPtrType, tensorPtr, offset);

  for (auto attribute : attributes)
    addPtrOp->setAttr(attribute.getFirst(), attribute.getSecond());
  return addPtrOp.getResult();
}

// Rewrite a memory operation
LogicalResult PointerCanonicalizer::materializeFatPointer(Operation *op,
                                                          Location loc,
                                                          Value ptr) {
  auto fatPtr = pointers[ptr];
  Value basePtr = fatPtr.basePtr;
  Value offset = fatPtr.offset;

  // Create the tensor pointer (i.e., splat the base && add the offset)
  Value newPtr = basePtr;
  if (isa<RankedTensorType>(ptr.getType()))
    newPtr = createTensorPointer(rewriter, fatPtr.basePtr, fatPtr.offset, loc,
                                 fatPtr.canNarrow, fatPtr.attributes);

  // Save the fat pointer in the table
  pointers[newPtr] = fatPtr;

  // Map and replace the load
  IRMapping mapper;
  mapper.map(ptr, newPtr);
  Operation *newOp = rewriter.clone(*op, mapper);
  rewriter.replaceAllOpUsesWith(op, newOp);
  opToDelete.insert(op);
  return success();
}

LogicalResult PointerCanonicalizer::rewriteSplatOp(triton::SplatOp splatOp,
                                                   Location curLoc,
                                                   Value &nextPtr) {
  nextPtr = splatOp.getResult();
  auto fatPtr = pointers[splatOp.getSrc()];
  auto outType = splatOp.getResult().getType();
  auto ptrShape = outType.getShape();
  auto newOffsetType = RankedTensorType::get(ptrShape, fatPtr.offset.getType(),
                                             outType.getEncoding());
  Value offset =
      rewriter.create<triton::SplatOp>(curLoc, newOffsetType, fatPtr.offset);
  // The shape of the fat pointer is contained within the offset. We don't
  // need to keep the `splat` operation here.
  opToDelete.insert(splatOp);
  pointers[nextPtr] = fatPtr.copy(splatOp.getSrc(), offset);
  return success();
}

LogicalResult
PointerCanonicalizer::rewriteBroadcastOp(triton::BroadcastOp broadcastOp,
                                         Location curLoc, Value &nextPtr) {
  nextPtr = broadcastOp.getResult();
  auto fatPtr = pointers[broadcastOp.getSrc()];
  auto outType = dyn_cast<RankedTensorType>(broadcastOp.getResult().getType());
  auto ptrShape = outType.getShape();
  auto offsetType = dyn_cast<RankedTensorType>(fatPtr.offset.getType());
  if (!offsetType)
    return failure();

  opToDelete.insert(broadcastOp);

  auto newOffsetType = RankedTensorType::get(
      ptrShape, offsetType.getElementType(), outType.getEncoding());
  Value offset = rewriter.create<triton::BroadcastOp>(curLoc, newOffsetType,
                                                      fatPtr.offset);
  pointers[nextPtr] = fatPtr.copyWithBase(offset);
  return success();
}

LogicalResult PointerCanonicalizer::rewriteAddPtrOp(triton::AddPtrOp addPtrOp,
                                                    Location curLoc,
                                                    Value &nextPtr) {
  nextPtr = addPtrOp.getResult();
  auto fatPtr = pointers[addPtrOp.getPtr()];
  Value newPtr = fatPtr.basePtr;
  // If it is a scalar pointer update, simply bump the base pointer
  if (!isa<RankedTensorType>(addPtrOp.getPtr().getType())) {
    addPtrOp->setOperand(0, newPtr);
    pointers[nextPtr] = fatPtr.copyWithOffset(nextPtr);
    return success();
  }
  Value offset = addPtrOp.getOffset();

  // Early exit for the case of a constant tensor
  if (Value scalarConst = getScalarConstant(rewriter, curLoc, offset)) {
    newPtr = rewriter.create<triton::AddPtrOp>(curLoc, newPtr.getType(), newPtr,
                                               scalarConst);
    pointers[nextPtr] = fatPtr.copyWithOffset(newPtr);
    // If we are updating the tensor pointer with a uniform value, we can
    // propagate the attributes of the tensor pointer to the fat pointer.
    for (auto attribute : fatPtr.attributes)
      pointers[nextPtr].setAttr(attribute.getFirst(), attribute.getSecond());
    opToDelete.insert(addPtrOp);
    return success();
  }

  int64_t bitness =
      cast<RankedTensorType>(offset.getType()).getElementTypeBitWidth();
  auto [uniformOffset, nonUniformOffset] =
      decomposeOffsetFromExpr(rewriter, curLoc, offset, bitness);

  // Scalar pointer update: bump the scalar pointer
  newPtr = rewriter.create<triton::AddPtrOp>(curLoc, newPtr.getType(), newPtr,
                                             uniformOffset);

  // Vector offset update (if any): bump the tensor offset
  Value fatPtrOffset = fatPtr.offset;
  bool canNarrow = fatPtr.canNarrow;
  Value newOffset = fatPtrOffset;
  bool propagateAtrs = true;
  if (!isZeroConst(nonUniformOffset)) {
    Type addPtrOffsetType = getElementTypeOrSelf(nonUniformOffset);
    Type fatPtrOffsetType = getElementTypeOrSelf(fatPtrOffset);
    canNarrow = canNarrow && canNarrowOffset(fatPtrOffset, nonUniformOffset);

    // Upcast or downcast the offset accordingly
    if (addPtrOffsetType.isInteger(32) && fatPtrOffsetType.isInteger(64))
      nonUniformOffset =
          createExtend32bitOffsetTo64Bits(rewriter, curLoc, nonUniformOffset);
    else if (addPtrOffsetType.isInteger(64) && fatPtrOffsetType.isInteger(32))
      nonUniformOffset =
          createNarrow64bitOffsetTo32bits(rewriter, curLoc, nonUniformOffset);

    newOffset =
        rewriter.create<arith::AddIOp>(curLoc, nonUniformOffset, fatPtrOffset);
    propagateAtrs = false;
  }
  opToDelete.insert(addPtrOp);
  pointers[nextPtr] = FatPtr{newPtr, newOffset, canNarrow};

  // If we are updating the tensor pointer with a uniform value, we can
  // propagate the attributes of the tensor pointer to the fat pointer.
  if (propagateAtrs)
    for (auto attribute : fatPtr.attributes)
      pointers[nextPtr].setAttr(attribute.getFirst(), attribute.getSecond());
  return success();
}

LogicalResult PointerCanonicalizer::rewriteForOp(scf::ForOp forOp,
                                                 Location curLoc,
                                                 OpOperand *curOperand,
                                                 Value &nextPtr) {
  size_t operandNum = curOperand->getOperandNumber();
  FatPtr fatPtr = pointers[curOperand->get()];
  Value offset = fatPtr.offset;
  Value basePtr = fatPtr.basePtr;

  // Replace the forOp with two additional argument (i.e., the curOperand's
  // scalar pointer and the offset)
  Value tensorPtr =
      createTensorPointer(rewriter, fatPtr.basePtr, fatPtr.offset, curLoc,
                          fatPtr.canNarrow, fatPtr.attributes);
  auto newForOp =
      replaceForOpWithNewSignature(rewriter, forOp, {basePtr, offset});
  rewriteOpMap[forOp] = newForOp;

  newForOp->setOperand(operandNum, tensorPtr);
  OpOperand *forOperand = &newForOp->getOpOperand(operandNum);
  // This is making sure we propagate the visit from the forOp result
  nextPtr = newForOp.getTiedLoopResult(forOperand);

  // This is making sure we visit the uses within the forOp region
  Value arg = newForOp.getTiedLoopRegionIterArg(forOperand);
  size_t numIterArgs = newForOp.getNumRegionIterArgs();
  pointers[arg] = fatPtr.copy(newForOp.getRegionIterArg(numIterArgs - 2),
                              newForOp.getRegionIterArg(numIterArgs - 1));

  // Collect attributes before continuing the visit
  collectFatPointerAttributes(newForOp, arg);

  for (OpOperand &use : arg.getUses())
    queue.push_back(&use);

  // This is setting the fat pointer for the users of the loop
  // and then propagate the result
  size_t numResults = newForOp->getNumResults();
  pointers[nextPtr] = fatPtr.copy(newForOp->getResult(numResults - 2),
                                  newForOp.getResult(numResults - 1));
  opToDelete.insert(forOp);
  return success();
}

LogicalResult PointerCanonicalizer::rewriteYieldOp(scf::YieldOp yieldOp,
                                                   Location curLoc,
                                                   OpOperand *curOperand,
                                                   Value &nextPtr) {

  // Rewriting the yield op is a bit more complicated, because a
  // yield op can be inside of a ForOp, WhileOp(in the AfterRegion) or
  // IfOp
  size_t operandNum = curOperand->getOperandNumber();
  FatPtr fatPtr = pointers[curOperand->get()];
  yieldOp.getResultsMutable().append(fatPtr.basePtr);
  yieldOp.getResultsMutable().append(fatPtr.offset);

  if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
    yieldOp->setOperand(operandNum, forOp.getRegionIterArg(operandNum));
  } else if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
    // Case 1: the yieldOp is contained within an IfOp. One of the
    // two branches is responsible to rewrite the operation. The other
    // branch only update the yieldOp with the right parameters
    Value tensorPtr =
        createTensorPointer(rewriter, fatPtr.basePtr, fatPtr.offset, curLoc,
                            fatPtr.canNarrow, fatPtr.attributes);
    yieldOp->setOperand(operandNum, tensorPtr);

    if (yieldOp->getBlock() == &ifOp.getThenRegion().front()) {
      auto newIfOp = replaceIfOpWithNewSignature(
          rewriter, ifOp, {fatPtr.basePtr.getType(), fatPtr.offset.getType()});
      nextPtr = newIfOp.getResult(operandNum);
      size_t numResults = newIfOp->getNumResults();
      pointers[nextPtr] = fatPtr.copy(newIfOp->getResult(numResults - 2),
                                      newIfOp.getResult(numResults - 1));
      opToDelete.insert(ifOp);
    }

  } else if (auto whileOp = resolveOp<scf::WhileOp>(yieldOp->getParentOp(),
                                                    rewriteOpMap)) {
    // Case 2: the yieldOp is contained within the AfterRegion of a
    // WhileOp. In this case, we know that the before region should have
    // already been replaced (when we met the WhileOp), hence we can
    // simply replace the WhileOp with a new AfterRegion (and hance a new
    // set of return types)
    auto newWhileOp = replaceWhileOpWithNewSignature(
        rewriter, whileOp, {},
        {fatPtr.basePtr.getType(), fatPtr.offset.getType()});
    nextPtr = newWhileOp.getResult(operandNum);
    size_t numResults = newWhileOp->getNumResults();
    pointers[nextPtr] = fatPtr.copy(newWhileOp->getResult(numResults - 2),
                                    newWhileOp->getResult(numResults - 1));
    rewriteOpMap[whileOp] = newWhileOp;
    opToDelete.insert(whileOp.getOperation());
    yieldOp.setOperand(operandNum, newWhileOp.getAfterArguments()[operandNum]);
  }
  return success();
}

LogicalResult PointerCanonicalizer::rewriteWhileOp(scf::WhileOp whileOp,
                                                   Location curLoc,
                                                   OpOperand *curOperand,
                                                   Value &nextPtr) {
  // WhileOp rewrite happens in two phases: first rewrite the operand list
  // and then rewrite the types when we meet the yieldOp
  size_t operandNum = curOperand->getOperandNumber();
  FatPtr fatPtr = pointers[curOperand->get()];
  Value offset = fatPtr.offset;
  Value basePtr = fatPtr.basePtr;
  // Rewrite the while op with a new set of operands (but with the same
  // set of return types)
  Value tensorPtr =
      createTensorPointer(rewriter, fatPtr.basePtr, fatPtr.offset, curLoc,
                          fatPtr.canNarrow, fatPtr.attributes);
  auto newWhileOp =
      replaceWhileOpWithNewSignature(rewriter, whileOp, {basePtr, offset}, {});
  newWhileOp->setOperand(operandNum, tensorPtr);
  Value arg = newWhileOp.getBeforeBody()->getArgument(operandNum);
  // Propagate inside the BeforeRegion
  size_t numArguments = newWhileOp.getBeforeBody()->getNumArguments();
  pointers[arg] =
      fatPtr.copy(newWhileOp.getBeforeBody()->getArgument(numArguments - 2),
                  newWhileOp.getBeforeBody()->getArgument(numArguments - 1));
  nextPtr = arg;
  rewriteOpMap[whileOp] = newWhileOp;
  opToDelete.insert(whileOp);
  return success();
}

// ConditionOp can only be contained within the BeforeRegion of a
// WhileOp. We already rewrote the WhileOp with the right operands, so
// we need only to add the offset the current operand to be the base
// pointer and continue the walk inside the AfterRegion
LogicalResult
PointerCanonicalizer::rewriteConditionOp(scf::ConditionOp conditionOp,
                                         Location curLoc, OpOperand *curOperand,
                                         Value &nextPtr) {

  size_t operandNum = curOperand->getOperandNumber();
  FatPtr fatPtr = pointers[curOperand->get()];
  Value offset = fatPtr.offset;
  Value basePtr = fatPtr.basePtr;
  auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());

  // Update the condition op
  auto afterBlock = whileOp.getAfterBody();
  conditionOp.getArgsMutable().append({basePtr, offset});

  // Propagate through the after region
  afterBlock->addArgument(basePtr.getType(), curLoc);
  afterBlock->addArgument(offset.getType(), curLoc);
  nextPtr = afterBlock->getArgument(operandNum - 1);
  size_t numArguments = afterBlock->getNumArguments();
  conditionOp.setOperand(operandNum,
                         whileOp.getRegionIterArgs()[operandNum - 1]);
  pointers[nextPtr] = fatPtr.copy(afterBlock->getArgument(numArguments - 2),
                                  afterBlock->getArgument(numArguments - 1));
  return success();
}

LogicalResult PointerCanonicalizer::rewriteCondBranchOp(
    cf::CondBranchOp condBrOp, Location curLoc, OpOperand *curOperand,
    Value &nextPtr) {
  // CondBranchOp is a bit tricky to handle. Because we might be inserting
  // the basePtr+offset as a TrueDestOperand(s), which is not the end of
  // `condBrOp.getOperands()`
  auto falseOperands = llvm::to_vector(condBrOp.getFalseDestOperands());
  auto trueOperands = llvm::to_vector(condBrOp.getTrueOperands());
  auto it = llvm::find(falseOperands, curOperand->get());
  bool isFalseOperand = (it != falseOperands.end());
  size_t operandNum = curOperand->getOperandNumber();

  if (rewriteOpMap.contains(condBrOp)) {
    // If we need to use a different condBrOp, we might also need to
    // update `operandNum`
    auto condBranchReplacement =
        dyn_cast<cf::CondBranchOp>(rewriteOpMap[condBrOp]);
    if (isFalseOperand) {
      // basePtr+offset need to be added if we are on the FalseOperands
      // side, but the true operands have been rewritten
      bool needOffset = (condBranchReplacement.getTrueDestOperands().size() !=
                         condBrOp.getTrueDestOperands().size());
      int maybeOffset = (needOffset ? 2 : 0);
      operandNum += maybeOffset;
      curOperand = &condBranchReplacement->getOpOperand(operandNum);
    }
    // Now we need to recompute the currentOperation and its {true,false}
    // operands
    falseOperands =
        llvm::to_vector(condBranchReplacement.getFalseDestOperands());
    trueOperands = llvm::to_vector(condBranchReplacement.getTrueDestOperands());
    condBrOp = condBranchReplacement;
  }

  // Now we can proceed almost normally
  FatPtr fatPtr = pointers[curOperand->get()];
  Value offset = fatPtr.offset;
  Value basePtr = fatPtr.basePtr;

  Block *falseDest = condBrOp.getFalseDest();
  Block *trueDest = condBrOp.getTrueDest();
  // Walk the destination block only if you don't have visited it yet
  if (isFalseOperand) {
    falseOperands.push_back(basePtr);
    falseOperands.push_back(offset);
    Value falseDestArg =
        falseDest->getArgument(operandNum - condBrOp.getNumTrueOperands() - 1);
    if (!pointers.contains(falseDestArg)) {
      nextPtr = falseDestArg;
      Value basePtrArg = falseDest->addArgument(basePtr.getType(), curLoc);
      Value offsetArg = falseDest->addArgument(offset.getType(), curLoc);
      pointers[nextPtr] = fatPtr.copy(basePtrArg, offsetArg);
    }
  } else {
    trueOperands.push_back(basePtr);
    trueOperands.push_back(offset);
    Value trueDestArg = trueDest->getArgument(operandNum - 1);
    if (!pointers.contains(trueDestArg)) {
      nextPtr = trueDestArg;
      Value basePtrArg = trueDest->addArgument(basePtr.getType(), curLoc);
      Value offsetArg = trueDest->addArgument(offset.getType(), curLoc);
      pointers[nextPtr] = fatPtr.copy(basePtrArg, offsetArg);
    }
  }

  // Create a new condBranch. We cannot simply extend the operands,
  // because this would invalidate other operands pointing at the same
  // cond branch
  Value tensorPtr =
      createTensorPointer(rewriter, fatPtr.basePtr, fatPtr.offset, curLoc,
                          fatPtr.canNarrow, fatPtr.attributes);
  auto newCondBranch = rewriter.create<cf::CondBranchOp>(
      curLoc, condBrOp.getCondition(), trueDest, trueOperands, falseDest,
      falseOperands);

  newCondBranch.setOperand(operandNum, tensorPtr);
  rewriteOpMap[condBrOp] = newCondBranch;
  opToDelete.insert(condBrOp);
  return success();
}

LogicalResult PointerCanonicalizer::rewriteSelectOp(arith::SelectOp selectOp,
                                                    Location curLoc,
                                                    OpOperand *curOperand,
                                                    Value &nextPtr) {
  Value trueVal = selectOp.getTrueValue();
  Value falseVal = selectOp.getFalseValue();
  Value cond = selectOp.getCondition();
  // If we didn't traverse both operands, simply materialize the pointer
  if (!pointers.contains(trueVal) || !pointers.contains(falseVal))
    return materializeFatPointer(selectOp, curLoc, curOperand->get());

  // If both have been traversed, then we can rewrite select of pointers as a
  // select of base and offset
  FatPtr fatPtrT = pointers[trueVal];
  FatPtr fatPtrF = pointers[falseVal];
  nextPtr = selectOp.getResult();

  // Simple case of a scalar select: update the base pointer
  if (!isa<RankedTensorType>(selectOp.getType())) {
    FatPtr fatPtr = pointers[trueVal];
    pointers[nextPtr] = fatPtr.copyWithOffset(nextPtr);
    nextPtr = selectOp.getResult();
    return success();
  }

  // Rewrite `select` for base and offset
  Value newBase = rewriter.create<arith::SelectOp>(
      curLoc, cond, fatPtrT.basePtr, fatPtrF.basePtr);
  Value newOffset = rewriter.create<arith::SelectOp>(
      curLoc, cond, fatPtrT.offset, fatPtrF.offset);
  assert(fatPtrT.canNarrow == fatPtrF.canNarrow);

  pointers[nextPtr] = fatPtrT.copy(newBase, newOffset);
  opToDelete.insert(selectOp);
  return success();
}

LogicalResult PointerCanonicalizer::rewriteBranchOp(cf::BranchOp branchOp,
                                                    Location curLoc,
                                                    OpOperand *curOperand,
                                                    Value &nextPtr) {
  size_t operandNum = curOperand->getOperandNumber();
  FatPtr fatPtr = pointers[curOperand->get()];
  Value offset = fatPtr.offset;
  Value basePtr = fatPtr.basePtr;
  branchOp.getDestOperandsMutable().append({basePtr, fatPtr.offset});
  Value tensorPtr =
      createTensorPointer(rewriter, fatPtr.basePtr, fatPtr.offset, curLoc,
                          fatPtr.canNarrow, fatPtr.attributes);
  branchOp->setOperand(operandNum, tensorPtr);
  Block *dest = branchOp.getDest();

  // Walk the destination block only if you don't have visited it yet
  if (!pointers.contains(dest->getArgument(operandNum))) {
    Value basePtrArg = dest->addArgument(basePtr.getType(), curLoc);
    Value offsetArg = dest->addArgument(offset.getType(), curLoc);
    nextPtr = dest->getArgument(operandNum);
    pointers[nextPtr] = {basePtrArg, offsetArg, fatPtr.canNarrow};
  }
  return success();
}

// Start from an argument of a function and propagate its
// fat pointers
LogicalResult PointerCanonicalizer::rewritePointer(Value argPtr) {
  // Start the visit
  for (OpOperand &use : argPtr.getUses())
    queue.push_back(&use);

  while (!queue.empty()) {
    OpOperand *curOperand = queue.pop_back_val();
    Operation *curOp = curOperand->getOwner();
    Location curLoc = curOp->getLoc();

    rewriter.setInsertionPoint(curOp);
    LogicalResult res = success();
    Value nextPtr;
    // We need to propagate the fat pointer throughout the IR
    llvm::TypeSwitch<Operation *>(curOp)
        .Case<triton::SplatOp>([&](auto splatOp) {
          res = rewriteSplatOp(splatOp, curLoc, nextPtr);
        })
        .Case<triton::BroadcastOp>([&](auto broadcastOp) {
          res = rewriteBroadcastOp(broadcastOp, curLoc, nextPtr);
        })
        .Case<triton::AddPtrOp>([&](auto addPtrOp) {
          res = rewriteAddPtrOp(addPtrOp, curLoc, nextPtr);
        })
        .Case<scf::ForOp>([&](auto forOp) {
          res = rewriteForOp(resolveOp<scf::ForOp>(forOp, rewriteOpMap), curLoc,
                             curOperand, nextPtr);
        })
        .Case<scf::YieldOp>([&](auto yieldOp) {
          res = rewriteYieldOp(yieldOp, curLoc, curOperand, nextPtr);
        })
        .Case<scf::WhileOp>([&](auto whileOp) {
          res = rewriteWhileOp(resolveOp<scf::WhileOp>(whileOp, rewriteOpMap),
                               curLoc, curOperand, nextPtr);
        })
        .Case<scf::ConditionOp>([&](auto conditionOp) {
          res = rewriteConditionOp(conditionOp, curLoc, curOperand, nextPtr);
        })
        .Case<cf::CondBranchOp>([&](auto condBrOp) {
          res = rewriteCondBranchOp(condBrOp, curLoc, curOperand, nextPtr);
        })
        .Case<arith::SelectOp>([&](auto selectOp) {
          res = rewriteSelectOp(selectOp, curLoc, curOperand, nextPtr);
        })
        .Case<cf::BranchOp>([&](auto branchOp) {
          res = rewriteBranchOp(branchOp, curLoc, curOperand, nextPtr);
        })
        .Case<triton::LoadOp, triton::StoreOp, triton::AtomicCASOp,
              triton::AtomicRMWOp, triton::PtrToIntOp>([&](Operation *op) {
          res = materializeFatPointer(curOp, curLoc, op->getOperand(0));
        })
        .Default([&](Operation *op) {
          // If we meet an unsupported operation, materialize the fat pointer
          // and continue.
          LDBG("Unknown op during pointer canonicalization: " << *curOp);
          res = materializeFatPointer(op, curLoc, curOperand->get());
        });

    // Collect the attributes and Keep propagating the fat pointer down the IR
    if (nextPtr) {
      collectFatPointerAttributes(curOp, nextPtr);
      for (OpOperand &use : nextPtr.getUses())
        if (!opToDelete.contains(use.getOwner()))
          queue.push_back(&use);
    }
  }
  return success();
}

LogicalResult PointerCanonicalizer::rewriteFunction(triton::FuncOp funcOp) {
  Region &region = funcOp.getRegion();
  for (auto [idx, arg] : llvm::enumerate(region.getArguments())) {
    // The pointer argument needs to be a scalar
    if (!isa<triton::PointerType>(arg.getType()))
      continue;
    int64_t bitness = 64;
    if (IntegerAttr pointerRangeAttr =
            funcOp.getArgAttrOfType<IntegerAttr>(idx, "tt.pointer_range"))
      bitness = pointerRangeAttr.getInt();

    rewriter.setInsertionPointToStart(&region.front());
    Value zeroOffset =
        rewriter.create<arith::ConstantIntOp>(region.getLoc(), 0, bitness);

    // Start the rewrite
    clearFunctionState();
    pointers[arg] = FatPtr{arg, zeroOffset, true};
    if (failed(rewritePointer(arg)))
      return failure();

    // Clean-up: don't assume the operation to delete are in the correct order,
    // but force dropping the reference of the ops before we delete them
    for (Operation *op : opToDelete) {
      op->dropAllReferences();
      op->dropAllDefinedValueUses();
      rewriter.eraseOp(op);
    }
  }
  return success();
}

LogicalResult PointerCanonicalizer::run() {
  llvm::SmallVector<triton::FuncOp> funcOps;

  // For now we don't cross function boundaries, but we should do that whenever
  // is possible
  mod.walk([&](triton::FuncOp funcOp) { funcOps.push_back(funcOp); });

  for (triton::FuncOp funcOp : funcOps) {
    if (failed(rewriteFunction(funcOp)))
      return failure();
  }
  return success();
}
// This pass is calling the pointer canonicalization utility
// on the given MLIR module
class TritonAMDGPUCanonicalizePointersPass
    : public TritonAMDGPUCanonicalizePointersBase<
          TritonAMDGPUCanonicalizePointersPass> {
public:
  TritonAMDGPUCanonicalizePointersPass() = default;

  void runOnOperation() override;
  void runOnOperationmine();
};

struct FatPointers {
  struct FatPtr {
    bool canNarrow = false;
    llvm::SmallDenseMap<StringAttr, Attribute> attributes;
  };
  using KeyT = std::pair<Value, Value>;
  using ValueT = FatPtr;
  using DenseMapT = DenseMap<KeyT, ValueT>;
  DenseMapT pointers;
  ValueT &operator[](const KeyT &k) { return pointers[k]; }
  ValueT &operator[](KeyT &&k) { return pointers[k]; }
  template <typename T>
  using const_arg_type_t = typename llvm::const_pointer_or_const_ref<T>::type;
  const ValueT &at(const_arg_type_t<KeyT> k) const { return pointers.at(k); }
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
struct PointerCanonPattern : OpConversionPattern<SourceOp> {
  PointerCanonPattern(MLIRContext *context, FatPointers &fatPtrs,
                      PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(context, benefit), fatPtrs(fatPtrs) {}
  FatPointers &fatPtrs;
};

class ConvertAddPtrOp : public PointerCanonPattern<triton::AddPtrOp> {
public:
  using PointerCanonPattern::PointerCanonPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp addPtrOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(addPtrOp);

    ArrayRef<ValueRange> remappedOperands = adaptor.getOperands();
    assert(remappedOperands.size() == 2 && remappedOperands[0].size() == 2 &&
           "expected adaptor to have 2 remapped values");
    Value fatPtrBase = remappedOperands[0][0];
    Value fatPtrOffset = remappedOperands[0][1];
    Value origOffset = remappedOperands[1][0];
    Location curLoc = addPtrOp.getLoc();

    // If it is a scalar pointer update, simply bump the base pointer
    if (!isa<RankedTensorType>(addPtrOp.getPtr().getType())) {
      auto newAddPtrOp = rewriter.create<triton::AddPtrOp>(
          curLoc, TypeRange{fatPtrBase.getType()},
          ValueRange{fatPtrBase, origOffset},
          llvm::ArrayRef{
              rewriter.getNamedAttr("legal", rewriter.getUnitAttr())});
      rewriter.modifyOpInPlace(addPtrOp, [&] {
        addPtrOp->setDiscardableAttr("rewritten", rewriter.getUnitAttr());
      });
      rewriter.replaceOpWithMultiple(addPtrOp, {{newAddPtrOp, fatPtrOffset}});
      return success();
    }

    // Early exit for the case of a constant tensor
    if (Value scalarConst = getScalarConstant(rewriter, curLoc, origOffset)) {
      auto newAddPtrOp = rewriter.create<triton::AddPtrOp>(
          curLoc, TypeRange{fatPtrBase.getType()},
          ValueRange{fatPtrBase, scalarConst},
          llvm::ArrayRef{
              rewriter.getNamedAttr("legal", rewriter.getUnitAttr())});
      rewriter.modifyOpInPlace(addPtrOp, [&] {
        addPtrOp->setDiscardableAttr("rewritten", rewriter.getUnitAttr());
      });
      rewriter.replaceOpWithMultiple(addPtrOp, {{newAddPtrOp, fatPtrOffset}});
      // If we are updating the tensor pointer with a uniform value, we can
      // propagate the attributes of the tensor pointer to the fat pointer.
      fatPtrs[{newAddPtrOp.getResult(), fatPtrOffset}].attributes =
          fatPtrs[{fatPtrBase, fatPtrOffset}].attributes;
      return success();
    }

    int64_t bitness =
        cast<RankedTensorType>(origOffset.getType()).getElementTypeBitWidth();
    auto [uniformOffset, nonUniformOffset] =
        decomposeOffsetFromExpr(rewriter, curLoc, origOffset, bitness);

    // Vector offset update (if any): bump the tensor offset
    bool canNarrow = false;
    bool propagateAtrs = true;
    Value newOffset = fatPtrOffset;
    if (!isZeroConst(nonUniformOffset)) {
      Type addPtrOffsetType = getElementTypeOrSelf(nonUniformOffset);
      Type fatPtrOffsetType = getElementTypeOrSelf(fatPtrOffset);
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

    // Scalar pointer update: bump the scalar pointer
    auto newAddPtrOp = rewriter.create<triton::AddPtrOp>(
        curLoc, TypeRange{fatPtrBase.getType()},
        ValueRange{fatPtrBase, uniformOffset},
        llvm::ArrayRef{rewriter.getNamedAttr("legal", rewriter.getUnitAttr())});
    rewriter.modifyOpInPlace(addPtrOp, [&] {
      addPtrOp->setDiscardableAttr("rewritten", rewriter.getUnitAttr());
    });
    rewriter.replaceOpWithMultiple(addPtrOp, {{newAddPtrOp, newOffset}});
    auto nextFatPtr = std::pair{newAddPtrOp.getResult(), newOffset};
    fatPtrs[nextFatPtr].canNarrow = canNarrow;
    if (propagateAtrs)
      fatPtrs[nextFatPtr].attributes =
          fatPtrs.at({fatPtrBase, fatPtrOffset}).attributes;

    return success();
  }
};

class ConvertSplatOp : public PointerCanonPattern<triton::SplatOp> {
public:
  using PointerCanonPattern::PointerCanonPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp splatOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ArrayRef<ValueRange> remappedOperands = adaptor.getOperands();
    // see
    // https://github.com/llvm/llvm-project/blob/58389b220a9354ed6c34bdb9310a35165579c5e3/mlir/lib/Transforms/Utils/DialectConversion.cpp#L1177
    assert(remappedOperands.size() == 1 && remappedOperands[0].size() == 2 &&
           "expected adaptor to have 2 remapped values");
    Value fatPtrBase = remappedOperands[0][0];
    Value fatPtrOffset = remappedOperands[0][1];
    assert(llvm::isa<triton::PointerType>(fatPtrBase.getType()) &&
           "expected fatPtrBase to be a tt.ptr");
    assert(llvm::isa<IntegerType>(fatPtrOffset.getType()) &&
           "expected fatPtrOffset to be an integer type");

    RankedTensorType outType = splatOp.getResult().getType();
    llvm::ArrayRef<int64_t> ptrShape = outType.getShape();
    auto newOffsetType = RankedTensorType::get(ptrShape, fatPtrOffset.getType(),
                                               outType.getEncoding());
    Value offset = rewriter.create<triton::SplatOp>(
        splatOp.getLoc(), newOffsetType, fatPtrOffset);
    rewriter.modifyOpInPlace(splatOp, [&] {
      splatOp->setDiscardableAttr("rewritten", rewriter.getUnitAttr());
    });
    rewriter.replaceOpWithMultiple(splatOp, {{splatOp.getSrc(), offset}});

    return success();
  }
};

class ConvertLoadOp : public PointerCanonPattern<triton::LoadOp> {
public:
  using PointerCanonPattern::PointerCanonPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp loadOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange fatPtr = *adaptor.getOperands().begin();
    Value fatPtrBase = fatPtr.front();
    Value fatPtrOffset = fatPtr.back();
    Location curLoc = loadOp.getLoc();

    llvm::SmallDenseMap<StringAttr, Attribute> attributes{
        {rewriter.getStringAttr("legal"), rewriter.getUnitAttr()}};
    Value newPtr = createTensorPointer(
        rewriter, fatPtrBase, fatPtrOffset, curLoc,
        fatPtrs[{fatPtrBase, fatPtrOffset}].canNarrow, attributes);
    SmallVector<Value> operands =
        loadOp.getOperands().take_back(loadOp.getNumOperands() - 1);
    operands.insert(operands.begin(), newPtr);
    SmallVector<NamedAttribute> attrs = llvm::to_vector(loadOp->getAttrs());
    attrs.append({rewriter.getNamedAttr("legal", rewriter.getUnitAttr())});
    auto newLoadPtrOp =
        rewriter.replaceOpWithNewOp<triton::LoadOp>(loadOp, operands, attrs);
    return success();
  }
};

class ConvertFuncOp : public PointerCanonPattern<triton::FuncOp> {
public:
  using PointerCanonPattern::PointerCanonPattern;

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
        rewriter.replaceOpWithMultiple(dummyCast, {{arg, zeroOffset}});
      }
      funcOp->setDiscardableAttr("rewritten", rewriter.getUnitAttr());
    });

    return success();
  }
};

class ConvertSCFYieldOp : public PointerCanonPattern<scf::YieldOp> {
public:
  using PointerCanonPattern::PointerCanonPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp yieldOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> newYieldedValues = flattenValues(adaptor.getOperands());

    // Value tensorPtr =
    //     createTensorPointer(rewriter, fatPtr.basePtr, fatPtr.offset, curLoc,
    //                         fatPtr.canNarrow, fatPtr.attributes);

    rewriter.modifyOpInPlace(yieldOp, [&]() {
      yieldOp.getResultsMutable().clear();
      yieldOp.getResultsMutable().append(newYieldedValues);
    });

    // TODO(max): this is bad
    if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
      if (ifOp.thenBlock() == yieldOp->getBlock())
        rewriter.modifyOpInPlace(ifOp, [&] {
          ifOp->setDiscardableAttr("then_rewritten", rewriter.getUnitAttr());
        });
      else
        rewriter.modifyOpInPlace(ifOp, [&] {
          ifOp->setDiscardableAttr("else_rewritten", rewriter.getUnitAttr());
        });
    }

    rewriter.modifyOpInPlace(yieldOp, [&] {
      yieldOp->setDiscardableAttr("legal", rewriter.getUnitAttr());
    });

    return success();
  }
};

class ConvertUnrealizedConversionCastOp
    : public PointerCanonPattern<UnrealizedConversionCastOp> {
public:
  using PointerCanonPattern::PointerCanonPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(std::distance(castOp->getUses().begin(), castOp->getUses().end()) >
               0 &&
           "expected at least 1 use of unrealized_cast");
    // dunno why but i get -Wdangling here...
    ArrayRef<ValueRange> remappedOperands = adaptor.getOperands();
    assert(remappedOperands.size() == 1 && remappedOperands[0].size() == 2 &&
           "expected adaptor to have 2 remapped values");
    Value fatPtrBase = remappedOperands[0][0];
    Value fatPtrOffset = remappedOperands[0][1];
    assert(llvm::isa<triton::PointerType>(fatPtrBase.getType()) &&
           "expected fatPtrBase to be a tt.ptr");
    assert(llvm::isa<IntegerType>(fatPtrOffset.getType()) &&
           "expected fatPtrOffset to be an integer type");
    OpFoldResult maybeScalar = getAsOpFoldResult(fatPtrOffset);
    if (auto attr = llvm::dyn_cast<mlir::Attribute>(maybeScalar)) {
      auto integerAttr = llvm::cast<IntegerAttr>(attr);
      if (integerAttr.getValue() == 0) {
        rewriter.replaceAllUsesWith(castOp.getResult(0), fatPtrBase);
        rewriter.eraseOp(castOp);
        return success();
      }
    }
    return failure();
  }
};

class ConvertSCFForOp : public PointerCanonPattern<scf::ForOp> {
  using PointerCanonPattern::PointerCanonPattern;

public:
  LogicalResult
  matchAndRewrite(scf::ForOp forOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<std::pair<Value, Value>> fatPtrInits;
    SmallVector<size_t> valRangeLens;
    ArrayRef<ValueRange> remappedInits = adaptor.getInitArgs();
    for (ValueRange remappedInit : remappedInits) {
      if (remappedInit.size() == 2) {
        Value fatPtrBase = remappedInit[0];
        Value fatPtrOffset = remappedInit[1];
        fatPtrInits.emplace_back(fatPtrBase, fatPtrOffset);
      }
      valRangeLens.push_back(remappedInit.size());
    }

    TypeConverter hackTypeConverter;
    unsigned inputNo = 0;
    hackTypeConverter.addConversion(
        [&inputNo, &remappedInits = std::as_const(remappedInits)](
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
    if (failed(
            rewriter.convertRegionTypes(&forOp.getRegion(), hackTypeConverter)))
      return failure();
    SmallVector<Value> initArgs = flattenValues(adaptor.getInitArgs());
    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), getSingleValue(adaptor.getLowerBound()),
        getSingleValue(adaptor.getUpperBound()),
        getSingleValue(adaptor.getStep()), initArgs);
    // replaceWithAdditionalYields

    newForOp->setAttrs(forOp->getAttrs());
    rewriter.eraseBlock(newForOp.getBody(0));
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

    rewriter.modifyOpInPlace(forOp, [&] {
      forOp->setDiscardableAttr("rewritten", rewriter.getUnitAttr());
    });
    rewriter.modifyOpInPlace(newForOp, [&] {
      newForOp->setDiscardableAttr("legal", rewriter.getUnitAttr());
    });
    rewriter.replaceOpWithMultiple(forOp, packedRets);

    return success();
  }
};

class ConvertSCFIfOp : public PointerCanonPattern<scf::IfOp> {
public:
  using PointerCanonPattern::PointerCanonPattern;
  // One of the two branches is responsible to rewrite the operation. The other
  // branch only update the yieldOp with the right parameters
  LogicalResult
  matchAndRewrite(scf::IfOp ifOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(ifOp.getNumResults() == 1 &&
           ifOp.thenYield().getOperandTypes().size() == 2 &&
           "only 1 -> 2 supported for scf::IfOp rewrite");
    bool withElseRegion = ifOp.getNumRegions() > 1;
    if (withElseRegion) {
      assert(ifOp.thenYield().getOperandTypes() ==
                 ifOp.elseYield().getOperandTypes() &&
             "ifOp types must match in both arms");
    }

    auto newIfOp = rewriter.create<scf::IfOp>(
        ifOp.getLoc(), ifOp.thenYield().getOperandTypes(), ifOp.getCondition(),
        withElseRegion);
    rewriter.inlineBlockBefore(ifOp.thenBlock(), newIfOp.thenBlock(),
                               newIfOp.thenBlock()->begin());
    if (withElseRegion)
      rewriter.inlineBlockBefore(ifOp.elseBlock(), newIfOp.elseBlock(),
                                 newIfOp.elseBlock()->begin());

    rewriter.modifyOpInPlace(ifOp, [&] {
      ifOp->setDiscardableAttr("rewritten", rewriter.getUnitAttr());
    });
    rewriter.modifyOpInPlace(newIfOp, [&] {
      newIfOp->setDiscardableAttr("legal", rewriter.getUnitAttr());
    });
    rewriter.replaceOpWithMultiple(ifOp, {newIfOp.getResults()});

    return success();
  }
};

void TritonAMDGPUCanonicalizePointersPass::runOnOperation() {
  ModuleOp module = getOperation();
  mlir::MLIRContext *context = &getContext();
  ConversionTarget target(*context);
  RewritePatternSet patterns(context);
  target.addLegalDialect<arith::ArithDialect>();
  auto isLegal = [](Operation *op) {
    if (op->hasAttr("rewritten") || op->hasAttr("legal"))
      return true;
    for (OpOperand &operand : op->getOpOperands()) {
      if (auto arg = llvm::dyn_cast<BlockArgument>(operand.get()))
        return !llvm::isa<triton::PointerType>(getElementTypeOrSelf(arg));
      if (operand.get().getDefiningOp()->hasAttr("rewritten"))
        return false;
    }
    return true;
  };
  target.addDynamicallyLegalDialect<triton::TritonDialect>(
      [&isLegal](Operation *op) {
        if (llvm::isa<triton::FuncOp>(op) && !op->hasAttr("rewritten"))
          return false;
        return isLegal(op);
      });
  target.addDynamicallyLegalDialect<scf::SCFDialect>([&isLegal](Operation *op) {
    if (auto ifOp = llvm::dyn_cast<scf::IfOp>(op))
      return !(ifOp->hasAttr("then_rewritten") and
               ifOp->hasAttr("else_rewritten"));
    return isLegal(op);
  });

  FatPointers fatPrs;

  patterns.add<ConvertFuncOp, ConvertSplatOp, ConvertAddPtrOp, ConvertLoadOp,
               ConvertSCFForOp, ConvertSCFYieldOp, ConvertSCFIfOp>(
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
}

void TritonAMDGPUCanonicalizePointersPass::runOnOperationmine() {
  ModuleOp m = getOperation();
  if (failed(PointerCanonicalizer(m).run()))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createTritonAMDGPUCanonicalizePointersPass() {
  return std::make_unique<TritonAMDGPUCanonicalizePointersPass>();
}
