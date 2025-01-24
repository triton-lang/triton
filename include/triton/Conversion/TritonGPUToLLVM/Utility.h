#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_UTILITY_H

#include <set>

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/StrUtil.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "ttgpu_to_llvm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

// Shortcuts for some commonly used LLVM ops to keep code simple and intuitive
// Operators
#define inttofloat(...) rewriter.create<LLVM::SIToFPOp>(loc, __VA_ARGS__)
#define inttoptr(...) rewriter.create<LLVM::IntToPtrOp>(loc, __VA_ARGS__)
#define ptrtoint(...) rewriter.create<LLVM::PtrToIntOp>(loc, __VA_ARGS__)
#define zext(...) rewriter.create<LLVM::ZExtOp>(loc, __VA_ARGS__)
#define sext(...) rewriter.create<LLVM::SExtOp>(loc, __VA_ARGS__)
#define fpext(...) rewriter.create<LLVM::FPExtOp>(loc, __VA_ARGS__)
#define fptrunc(...) rewriter.create<LLVM::FPTruncOp>(loc, __VA_ARGS__)
#define trunc(...) rewriter.create<LLVM::TruncOp>(loc, __VA_ARGS__)
#define udiv(...) rewriter.create<LLVM::UDivOp>(loc, __VA_ARGS__)
#define sdiv(...) rewriter.create<LLVM::SDivOp>(loc, __VA_ARGS__)
#define urem(...) rewriter.create<LLVM::URemOp>(loc, __VA_ARGS__)
#define add(...) rewriter.create<LLVM::AddOp>(loc, __VA_ARGS__)
#define sub(...) rewriter.create<LLVM::SubOp>(loc, __VA_ARGS__)
#define fadd(...) rewriter.create<LLVM::FAddOp>(loc, __VA_ARGS__)
#define mul(...) rewriter.create<LLVM::MulOp>(loc, __VA_ARGS__)
#define fmul(...) rewriter.create<LLVM::FMulOp>(loc, __VA_ARGS__)
#define fma(...) rewriter.create<LLVM::FMAOp>(loc, __VA_ARGS__)
#define neg(...) rewriter.create<LLVM::FNegOp>(loc, __VA_ARGS__)
#define smax(...) rewriter.create<LLVM::SMaxOp>(loc, __VA_ARGS__)
#define umax(...) rewriter.create<LLVM::UMaxOp>(loc, __VA_ARGS__)
#define fmax(...) rewriter.create<LLVM::MaxNumOp>(loc, __VA_ARGS__)
#define smin(...) rewriter.create<LLVM::SMinOp>(loc, __VA_ARGS__)
#define umin(...) rewriter.create<LLVM::UMinOp>(loc, __VA_ARGS__)
#define fmin(...) rewriter.create<LLVM::MinNumOp>(loc, __VA_ARGS__)
#define shl(...) rewriter.create<LLVM::ShlOp>(loc, __VA_ARGS__)
#define lshr(...) rewriter.create<LLVM::LShrOp>(loc, __VA_ARGS__)
#define ashr(...) rewriter.create<LLVM::AShrOp>(loc, __VA_ARGS__)
#define and_(...) rewriter.create<LLVM::AndOp>(loc, __VA_ARGS__)
#define xor_(...) rewriter.create<LLVM::XOrOp>(loc, __VA_ARGS__)
#define or_(...) rewriter.create<LLVM::OrOp>(loc, __VA_ARGS__)
#define bitcast(val__, type__)                                                 \
  rewriter.create<LLVM::BitcastOp>(loc, type__, val__)
#define addrspacecast(...)                                                     \
  rewriter.create<LLVM::AddrSpaceCastOp>(loc, __VA_ARGS__)
#define gep(...) rewriter.create<LLVM::GEPOp>(loc, __VA_ARGS__)
#define ptr_ty(...) LLVM::LLVMPointerType::get(__VA_ARGS__)
#define insert_val(...) rewriter.create<LLVM::InsertValueOp>(loc, __VA_ARGS__)
#define extract_val(...) rewriter.create<LLVM::ExtractValueOp>(loc, __VA_ARGS__)
#define insert_element(...)                                                    \
  rewriter.create<LLVM::InsertElementOp>(loc, __VA_ARGS__)
#define extract_element(...)                                                   \
  rewriter.create<LLVM::ExtractElementOp>(loc, __VA_ARGS__)
#define load(...) rewriter.create<LLVM::LoadOp>(loc, __VA_ARGS__)
#define store(...) rewriter.create<LLVM::StoreOp>(loc, __VA_ARGS__)
#define fcmp_ogt(lhs, rhs)                                                     \
  rewriter.create<LLVM::FCmpOp>(loc, rewriter.getI1Type(),                     \
                                LLVM::FCmpPredicate::ogt, lhs, rhs)
#define fcmp_olt(lhs, rhs)                                                     \
  rewriter.create<LLVM::FCmpOp>(loc, rewriter.getI1Type(),                     \
                                LLVM::FCmpPredicate::olt, lhs, rhs)
#define fcmp_eq(lhs, rhs)                                                      \
  rewriter.create<LLVM::FCmpOp>(loc, rewriter.getI1Type(),                     \
                                LLVM::FCmpPredicate::oeq, lhs, rhs)
#define icmp_eq(...)                                                           \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, __VA_ARGS__)
#define icmp_ne(...)                                                           \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, __VA_ARGS__)
#define icmp_slt(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, __VA_ARGS__)
#define icmp_sle(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sle, __VA_ARGS__)
#define icmp_sgt(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sgt, __VA_ARGS__)
#define icmp_sge(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, __VA_ARGS__)
#define icmp_ult(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult, __VA_ARGS__)
#define icmp_ule(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ule, __VA_ARGS__)
#define icmp_ugt(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ugt, __VA_ARGS__)
#define icmp_uge(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::uge, __VA_ARGS__)
#define select(...) rewriter.create<LLVM::SelectOp>(loc, __VA_ARGS__)
#define address_of(...) rewriter.create<LLVM::AddressOfOp>(loc, __VA_ARGS__)
#define barrier() rewriter.create<mlir::gpu::BarrierOp>(loc)
#define undef(...) rewriter.create<LLVM::UndefOp>(loc, __VA_ARGS__)
#define null(...) rewriter.create<LLVM::ZeroOp>(loc, __VA_ARGS__)
#define call(...) LLVM::createLLVMCallOp(rewriter, loc, __VA_ARGS__)

// Types
#define int_ty(width) rewriter.getIntegerType(width)
#define i64_ty rewriter.getIntegerType(64)
#define i32_ty rewriter.getIntegerType(32)
#define i16_ty rewriter.getIntegerType(16)
#define i32_ty rewriter.getIntegerType(32)
#define i64_ty rewriter.getIntegerType(64)
#define ui32_ty rewriter.getIntegerType(32, false)
#define ui64_ty rewriter.getIntegerType(64, false)
#define f16_ty rewriter.getF16Type()
#define bf16_ty rewriter.getBF16Type()
#define i8_ty rewriter.getIntegerType(8)
#define i1_ty rewriter.getI1Type()
#define f32_ty rewriter.getF32Type()
#define f64_ty rewriter.getF64Type()
#define vec_ty(type, num) VectorType::get(num, type)
#define void_ty(ctx) LLVM::LLVMVoidType::get(ctx)
#define struct_ty(...) LLVM::LLVMStructType::getLiteral(ctx, __VA_ARGS__)
#define array_ty(elemTy, count) LLVM::LLVMArrayType::get(elemTy, count)

// Constants
#define int_val(bitwidth, val)                                                 \
  LLVM::createLLVMIntegerConstant(rewriter, loc, bitwidth, val)
#define i1_val(val) LLVM::createConstantI1(loc, rewriter, val)
#define true_val() i1_val(true)
#define false_val() i1_val(false)
#define f16_val(...) LLVM::createConstantF16(loc, rewriter, __VA_ARGS__)
#define f32_val(...) LLVM::createConstantF32(loc, rewriter, __VA_ARGS__)
#define f64_val(...) LLVM::createConstantF64(loc, rewriter, __VA_ARGS__)
#define i8_val(val) int_val(8, val)
#define i16_val(val) int_val(16, val)
#define i32_val(...) LLVM::createConstantI32(loc, rewriter, __VA_ARGS__)
#define i64_val(...) LLVM::createConstantI64(loc, rewriter, __VA_ARGS__)
#define tid_val() getThreadId(rewriter, loc)

// Attributes
#define i32_arr_attr(...) rewriter.getI32ArrayAttr({__VA_ARGS__})
#define i64_arr_attr(...) rewriter.getI64ArrayAttr({__VA_ARGS__})
#define str_attr(str) ::mlir::StringAttr::get(ctx, (str))

namespace mlir {
namespace triton {

// Delinearize supposing order is [0, 1, .. , n]
template <typename T>
llvm::SmallVector<T> getMultiDimIndexImpl(T linearIndex,
                                          llvm::ArrayRef<T> shape) {
  // shape: {a, b, c, d}  ->  accMul: {1, a, a*b, a*b*c}
  size_t rank = shape.size();
  T accMul = product(shape.drop_back());
  T linearRemain = linearIndex;
  llvm::SmallVector<T> multiDimIndex(rank);
  for (int i = rank - 1; i >= 0; --i) {
    multiDimIndex[i] = linearRemain / accMul;
    linearRemain = linearRemain % accMul;
    if (i != 0) {
      accMul = accMul / shape[i - 1];
    }
  }
  return multiDimIndex;
}

template <typename T>
llvm::SmallVector<T> getMultiDimIndex(T linearIndex, llvm::ArrayRef<T> shape,
                                      llvm::ArrayRef<unsigned> order) {
  size_t rank = shape.size();
  assert(rank == order.size());
  auto reordered = applyPermutation(shape, order);
  auto reorderedMultiDim = getMultiDimIndexImpl<T>(linearIndex, reordered);
  llvm::SmallVector<T> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

// Linearize supposing order is [0, 1, .. , n]
template <typename T>
T getLinearIndexImpl(llvm::ArrayRef<T> multiDimIndex, llvm::ArrayRef<T> shape) {
  assert(multiDimIndex.size() == shape.size());
  // shape: {a, b, c, d}  ->  accMul: {1, a, a*b, a*b*c}
  size_t rank = shape.size();
  T accMul = product(shape.drop_back());
  T linearIndex = 0;
  for (int i = rank - 1; i >= 0; --i) {
    linearIndex += multiDimIndex[i] * accMul;
    if (i != 0) {
      accMul = accMul / shape[i - 1];
    }
  }
  return linearIndex;
}

template <typename T>
T getLinearIndex(llvm::ArrayRef<T> multiDimIndex, llvm::ArrayRef<T> shape,
                 llvm::ArrayRef<unsigned> order) {
  assert(shape.size() == order.size());
  return getLinearIndexImpl<T>(applyPermutation(multiDimIndex, order),
                               applyPermutation(shape, order));
}

namespace gpu {
Type getFunctionType(Type resultType, ValueRange operands);

LLVM::LLVMFuncOp appendOrGetExternFuncOp(RewriterBase &rewriter, Operation *op,
                                         StringRef funcName, Type funcType,
                                         StringRef libname = "",
                                         StringRef libpath = "");
} // namespace gpu

} // namespace triton

namespace LLVM {
using namespace mlir::triton;

Value createConstantI1(Location loc, OpBuilder &rewriter, bool v);
Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v);
Value createConstantI64(Location loc, OpBuilder &rewriter, int64_t v);
Value createConstantF16(Location loc, OpBuilder &rewriter, float v);
Value createConstantF32(Location loc, OpBuilder &rewriter, float v);
Value createConstantF64(Location loc, OpBuilder &rewriter, double v);
Value createNaNConstant(Location loc, OpBuilder &rewriter, Type type);
Value createIndexConstant(OpBuilder &builder, Location loc,
                          const TypeConverter *converter, int64_t value);
Value createLLVMIntegerConstant(OpBuilder &builder, Location loc, short width,
                                int64_t value);

LLVM::CallOp createLLVMCallOp(OpBuilder &builder, Location loc,
                              LLVMFuncOp funcOp, ValueRange args);
LLVM::CallIntrinsicOp
createLLVMIntrinsicCallOp(OpBuilder &builder, Location loc, StringRef intrinsic,
                          TypeRange types, ValueRange args);

// Is v an integer or floating-point scalar constant equal to 0?
bool isConstantZero(Value v);

class SharedMemoryObject {
public:
  SharedMemoryObject(Value base, Type baseElemType, ArrayRef<Value> offsets)
      : base(base), baseElemType(baseElemType),
        offsets(offsets.begin(), offsets.end()) {}

  SharedMemoryObject(Value base, Type baseElemType, int64_t rank, Location loc,
                     RewriterBase &rewriter)
      : base(base), baseElemType(baseElemType) {
    offsets.append(rank, i32_val(0));
  }

  SmallVector<Value> getOffsets() const { return offsets; }
  Value getBase() const { return base; }
  Type getBaseElemType() const { return baseElemType; }

  SmallVector<Value> getElems() const {
    SmallVector<Value> elems;
    elems.push_back(base);
    elems.append(offsets.begin(), offsets.end());
    return elems;
  }

  SmallVector<Type> getTypes() const {
    SmallVector<Type> types;
    types.push_back(base.getType());
    types.append(offsets.size(), IntegerType::get(base.getContext(), 32));
    return types;
  }

  SmallVector<Value> getStrides(triton::gpu::MemDescType memDesc, Location loc,
                                RewriterBase &rewriter) const {
    auto allocShape = memDesc.getAllocShape();
    auto allocShapePerCTA =
        triton::gpu::getShapePerCTA(memDesc.getEncoding(), allocShape);
    auto layoutOrder = triton::gpu::getOrder(memDesc.getEncoding());
    auto allocStrides = SharedMemoryObject::getStridesForShape(
        allocShapePerCTA, layoutOrder, loc, rewriter);
    return SmallVector<Value>(allocStrides.end() - offsets.size(),
                              allocStrides.end());
  }

  // TODO(Keren): deprecate the method once AMD backend has cleaned up
  Value getCSwizzleOffset(int dim) const {
    assert(dim >= 0 && dim < offsets.size());
    return offsets[dim];
  }

  // TODO(Keren): deprecate the method once AMD backend has cleaned up
  Value getBaseBeforeSlice(int dim, Location loc,
                           RewriterBase &rewriter) const {
    Value cSwizzleOffset = getCSwizzleOffset(dim);
    Value offset = sub(i32_val(0), cSwizzleOffset);
    Type type = base.getType();
    return gep(type, baseElemType, base, offset);
  }

private:
  static SmallVector<unsigned>
  getOrderForShape(ArrayRef<int64_t> shape, ArrayRef<unsigned> layoutOrder) {
    SmallVector<unsigned> order(shape.size());
    // Default minor-to-major order
    std::iota(order.rbegin(), order.rend(), 0);
    if (layoutOrder.size() > 0) {
      // If a layout order is provided, we assume it specifies the order in
      // which the dimensions are first accessed, and unspecified dimensions
      // retain the minor-to-major order. For example, if order = [2, 1, 0] and
      // layoutOrder = [0, 1], we need to shift `layoutOrder`
      // by -1 (move them right). The resulting order will then be [1, 2, 0].
      int rankDiff = layoutOrder.size() - shape.size();
      auto minRank = std::min<size_t>(shape.size(), layoutOrder.size());
      for (size_t i = 0; i < minRank; ++i)
        order[i] = layoutOrder[i] - rankDiff;
    }
    assert(isPermutationOfIota(order) && "Invalid order");
    return order;
  }

  static SmallVector<Value> getStridesForShape(ArrayRef<int64_t> shape,
                                               ArrayRef<unsigned> layoutOrder,
                                               Location loc,
                                               RewriterBase &rewriter) {
    SmallVector<Value> strides(shape.size());
    auto order = SharedMemoryObject::getOrderForShape(shape, layoutOrder);
    int64_t stride = 1;
    for (auto idx : order) {
      strides[idx] = i32_val(stride);
      stride *= shape[idx];
    }
    return strides;
  }

  Value base; // i32 ptr. The start address of the shared memory object.
  Type baseElemType;
  SmallVector<Value>
      offsets; // i32 int. The offsets are zero at the initial allocation.
};

Value getStructFromSharedMemoryObject(Location loc,
                                      const SharedMemoryObject &smemObj,
                                      RewriterBase &rewriter);

SharedMemoryObject getSharedMemoryObjectFromStruct(Location loc,
                                                   Value llvmStruct,
                                                   Type elemTy,
                                                   RewriterBase &rewriter);

// Convert an \param index to a multi-dim coordinate given \param shape and
// \param order.
SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape,
                               ArrayRef<unsigned> order);

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               unsigned linear, ArrayRef<unsigned> shape);

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape);

SmallVector<unsigned> delinearize(unsigned linear, ArrayRef<unsigned> shape,
                                  ArrayRef<unsigned> order);

// Returns a tuple with the delinearized coordinates and a boolean which is true
// iff the Value is not broadcasted (equivalently, if the value is the "first"
// lane/thread/etc. that holds the given value). In mathy terms, the boolean is
// true if the element is the canonical representative of the class.
std::tuple<SmallVector<Value>, Value>
delinearize(RewriterBase &rewriter, Location loc,
            triton::gpu::DistributedEncodingTrait layout,
            ArrayRef<int64_t> shape, StringAttr dimName, Value linear);

Value linearize(RewriterBase &rewriter, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape, ArrayRef<unsigned> order);

Value linearize(RewriterBase &rewriter, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape);

size_t linearize(ArrayRef<unsigned> multiDim, ArrayRef<unsigned> shape,
                 ArrayRef<unsigned> order);

Value addStringToModule(Location loc, RewriterBase &rewriter, StringRef key,
                        StringRef content);

// Given an elemId which represents the index of an element from the list of
// elements that are in the thread's registers (i.e. total of
// numel(sizePerThread)), it calculates the multi dim offset of the element in
// the smem buffer. Recall that the smem buffer will only store a single replica
// when converting distributed to distributed layout. Also, a replica is the
// smallest CTA tile that is common between input and output layouts.
SmallVector<Value> getMultiDimOffset(Attribute layout, Location loc,
                                     RewriterBase &rewriter,
                                     const TargetInfoBase &targetInfo,
                                     unsigned elemId, RankedTensorType type,
                                     ArrayRef<unsigned> multiDimCTAInRepId,
                                     ArrayRef<unsigned> shapePerCTATile);

// Given a multiDimOffset, this function wraps around each dimension to be
// within shape.
SmallVector<Value> getWrappedMultiDimOffset(
    RewriterBase &rewriter, Location loc, ArrayRef<Value> multiDimOffset,
    ArrayRef<unsigned> shape, SmallVector<unsigned> shapePerCTATile,
    SmallVector<int64_t> shapePerCTA);

inline bool isKernel(FunctionOpInterface funcOp) {
  return funcOp.getVisibility() == SymbolTable::Visibility::Public;
}

inline Value getStackPointer(RewriterBase &rewriter,
                             FunctionOpInterface funcOp) {
  // See NOTE: [Additional Function Arguments]
  if (!isKernel(funcOp)) {
    return funcOp.getArgument(funcOp.getNumArguments() - 2);
  }

  auto mod = funcOp->getParentOfType<ModuleOp>();
  auto globalBase = dyn_cast<LLVM::GlobalOp>(mod.lookupSymbol("global_smem"));
  assert(globalBase);
  return rewriter.create<LLVM::AddressOfOp>(funcOp.getLoc(), globalBase);
}

inline Value getGlobalScratchPtr(Location loc, RewriterBase &rewriter,
                                 FunctionOpInterface funcOp,
                                 Value allocOffset = {}) {
  // See NOTE: [Additional Function Arguments]
  if (!isKernel(funcOp)) {
    // Base for this function
    auto gmemBase = funcOp.getArgument(funcOp.getNumArguments() - 1);
    if (!allocOffset) {
      return gmemBase;
    }

    auto ptrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext(), 1);
    return gep(ptrTy, i8_ty, gmemBase, allocOffset);
  }

  // Base for entire kernel
  auto gmemBase = funcOp.getArgument(funcOp.getNumArguments() - 1);

  ModuleOp mod = funcOp.getOperation()->getParentOfType<ModuleOp>();
  auto allocSizeAttr = mod.getOperation()->getAttrOfType<mlir::IntegerAttr>(
      "ttg.global_scratch_memory_size");
  if (!allocSizeAttr) {
    return gmemBase;
  }

  Value gridIdx[3];
  Value gridDim[2];
  for (int k = 0; k < 3; ++k) {
    gridIdx[k] = rewriter.create<GetProgramIdOp>(loc, k);
  }
  for (int k = 0; k < 2; ++k) {
    gridDim[k] = rewriter.create<GetNumProgramsOp>(loc, k);
  }

  Value linearId = gridIdx[2];
  for (int k = 0; k < 2; ++k) {
    linearId = add(gridIdx[1 - k], mul(linearId, gridDim[1 - k]));
  }

  auto allocSize = allocSizeAttr.getValue().getZExtValue();

  Value offset = mul(linearId, i32_val(allocSize));
  if (allocOffset) {
    offset = add(offset, allocOffset);
  }

  auto *ctx = rewriter.getContext();
  auto res =
      gep(mlir::LLVM::LLVMPointerType::get(ctx, 1), i8_ty, gmemBase, offset);
  return res;
}

inline Value getSharedMemoryBase(Location loc, RewriterBase &rewriter,
                                 const TargetInfoBase &target, Operation *op) {
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                          target.getSharedAddressSpace());
  FunctionOpInterface func =
      op->template getParentOfType<FunctionOpInterface>();
  assert(op->hasAttr("allocation.offset"));
  size_t offset = cast<IntegerAttr>(op->getAttr("allocation.offset"))
                      .getValue()
                      .getZExtValue();
  Value offVal = i32_val(offset);
  Value base = gep(ptrTy, i8_ty, LLVM::getStackPointer(rewriter, func), offVal);
  return base;
}

// -----------------------------------------------------------------------
// MXFP utilities
// -----------------------------------------------------------------------

// Scale a mxfp4 value by a given scale.
Value mxfpScaleBf16(RewriterBase &rewriter, Location loc, Value v, Value scale,
                    bool fastMath);

} // namespace LLVM

/* ------------------------------------ */
// Returns CTA level thread idx
inline Value getThreadId(RewriterBase &rewriter, Location loc) {
  Value tid =
      rewriter.create<::mlir::gpu::ThreadIdOp>(loc, ::mlir::gpu::Dimension::x);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, tid);
}

// -----------------------------------------------------------------------
// Shared memory utilities
// -----------------------------------------------------------------------
using LLVM::getMultiDimIndex;
using LLVM::SharedMemoryObject;
using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::SharedMemoryObject;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::CTALayoutAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

inline Value dot(RewriterBase &rewriter, Location loc, ArrayRef<Value> offsets,
                 ArrayRef<Value> strides) {
  assert(offsets.size() == strides.size());
  Value ret = i32_val(0);
  for (auto [offset, stride] : llvm::zip(offsets, strides)) {
    ret = add(ret, mul(offset, stride));
  }
  return ret;
}

/// Extend 2d shared object to 3d.
///
/// If tensor has 3 dimensions, returns original shared object.
/// If tensor shape is [M, N], return shared object describing shape [1, M, N]
///
/// This Function is used to simplify processing of 2d and 3d dot operands,
/// particularly in the conversion of local_load operation.
///
/// \param rewriter
/// \param loc
/// \param smemObj
/// \param shape shape of a tensor represented by smemObj
/// \returns shared object describing 3d tensor
SharedMemoryObject
getExpandedSharedMemoryObject(ConversionPatternRewriter &rewriter, Location loc,
                              SharedMemoryObject smemObj,
                              ArrayRef<int64_t> shape);

// -----------------------------------------------------------------------
// Blocked layout indices
// -----------------------------------------------------------------------

// "Applies" the given layout by computing layout(indices) and returning the
// resulting Values.
//
// In other words, this generates LLVM-dialect MLIR code to "run" the layout
// function.
SmallVector<std::pair<StringAttr, Value>>
applyLinearLayout(Location loc, RewriterBase &rewriter,
                  const LinearLayout &layout,
                  ArrayRef<std::pair<StringAttr, Value>> indices);

inline SmallVector<Value>
emitBaseIndexWithinCTAForBlockedLayout(Location loc, RewriterBase &rewriter,
                                       const BlockedEncodingAttr &blockedLayout,
                                       RankedTensorType type) {
  MLIRContext *ctx = rewriter.getContext();
  auto shape = type.getShape();
  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = i32_val(triton::gpu::getWarpSize(blockedLayout));
  Value laneId = urem(threadId, warpSize);
  Value warpId = udiv(threadId, warpSize);
  auto sizePerThread = blockedLayout.getSizePerThread();
  auto threadsPerWarp = blockedLayout.getThreadsPerWarp();
  auto warpsPerCTA = blockedLayout.getWarpsPerCTA();
  auto threadOrder = blockedLayout.getThreadOrder();
  auto warpOrder = blockedLayout.getWarpOrder();
  auto shapePerCTA = triton::gpu::getShapePerCTA(blockedLayout, shape);
  unsigned rank = shape.size();

  // delinearize threadId to get the base index
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, warpsPerCTA, warpOrder);
  SmallVector<Value> multiDimThreadId =
      delinearize(rewriter, loc, laneId, threadsPerWarp, threadOrder);

  SmallVector<Value> multiDimBase(rank);
  for (unsigned k = 0; k < rank; ++k) {
    // Wrap around multiDimWarpId/multiDimThreadId in case
    // shapePerCTATile[k] > shapePerCTA[k]
    auto maxWarps =
        ceil<unsigned>(shapePerCTA[k], sizePerThread[k] * threadsPerWarp[k]);
    auto maxThreads = ceil<unsigned>(shapePerCTA[k], sizePerThread[k]);
    multiDimWarpId[k] = urem(multiDimWarpId[k], i32_val(maxWarps));
    multiDimThreadId[k] = urem(multiDimThreadId[k], i32_val(maxThreads));
    // multiDimBase[k] = (multiDimThreadId[k] +
    //                    multiDimWarpId[k] * threadsPerWarp[k]) *
    //                   sizePerThread[k];
    Value threadsPerWarpK = i32_val(threadsPerWarp[k]);
    Value sizePerThreadK = i32_val(sizePerThread[k]);
    multiDimBase[k] =
        mul(sizePerThreadK,
            add(multiDimThreadId[k], mul(multiDimWarpId[k], threadsPerWarpK)));
  }

  return multiDimBase;
}

// -----------------------------------------------------------------------
// Mma layout indices
// -----------------------------------------------------------------------

// Note that this may return a null Value for one or more dimensions.  This is
// valid only if you're going to slice off the relevant dimension.
inline SmallVector<Value>
emitBaseIndexWithinCTAForMmaLayoutV2V3(Location loc, RewriterBase &rewriter,
                                       const NvidiaMmaEncodingAttr &mmaLayout,
                                       RankedTensorType type) {
  auto shape = type.getShape();
  auto _warpsPerCTA = mmaLayout.getWarpsPerCTA();
  auto rank = shape.size();
  assert(rank == 2 || rank == 3);
  auto warpOrder = triton::gpu::getWarpOrder(mmaLayout);
  ArrayRef<unsigned int> instrShape = mmaLayout.getInstrShape();
  SmallVector<Value> warpsPerCTA;
  for (unsigned i = 0; i < rank; ++i)
    warpsPerCTA.push_back(i32_val(_warpsPerCTA[i]));
  auto shapePerCTA = getShapePerCTA(mmaLayout, shape);

  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = i32_val(32);
  Value laneId = urem(threadId, warpSize);
  Value warpId = udiv(threadId, warpSize);

  uint32_t repM =
      (_warpsPerCTA[rank - 2] * instrShape[rank - 2]) / shapePerCTA[rank - 2];
  uint32_t repN =
      (_warpsPerCTA[rank - 1] * instrShape[rank - 1]) / shapePerCTA[rank - 1];

  uint32_t warpsM;
  if (repM > 1)
    warpsM = _warpsPerCTA[rank - 2] / repM;
  else
    warpsM = shape[rank - 2] / instrShape[rank - 2];

  uint32_t warpsN;
  if (repN > 1)
    warpsN = _warpsPerCTA[rank - 1] / repN;
  else
    warpsN = shape[rank - 1] / instrShape[rank - 1];

  SmallVector<Value> multiDimWarpId(rank);
  multiDimWarpId = delinearize(rewriter, loc, warpId, _warpsPerCTA, warpOrder);
  Value warpIdM = urem(multiDimWarpId[rank - 2], i32_val(warpsM));
  Value warpIdN = urem(multiDimWarpId[rank - 1], i32_val(warpsN));

  Value offWarpM = mul(warpIdM, i32_val(instrShape[rank - 2]));
  Value offWarpN = mul(warpIdN, i32_val(instrShape[rank - 1]));

  SmallVector<Value> multiDimBase(rank);
  if (rank == 3)
    multiDimBase[0] = multiDimWarpId[0];

  // warpsM/N may be 0, in which case warpIDM/N is poison (division by 0), which
  // will cause LLVM to eliminate all ops that depend on the poison value.  This
  // *can* be okay, if the bad dimension is filtered out by a slice layout.  So
  // we rely on the caller to check.  Worst case we crash, which is better than
  // silently producing bad code.
  if (warpsM != 0)
    multiDimBase[rank - 2] = add(udiv(laneId, i32_val(4)), offWarpM);
  if (warpsN != 0)
    multiDimBase[rank - 1] =
        add(mul(i32_val(2), urem(laneId, i32_val(4))), offWarpN);

  return multiDimBase;
}

inline SmallVector<Value>
emitBaseIndexForMfmaLayout(Location loc, RewriterBase &rewriter,
                           const AMDMfmaEncodingAttr &mfmaLayout,
                           RankedTensorType type) {
  auto shape = type.getShape();
  auto rank = shape.size();
  assert(rank == 2 || rank == 3);
  auto _warpsPerCTA = mfmaLayout.getWarpsPerCTA();
  SmallVector<Value> warpsPerCTA;
  for (unsigned i = 0; i < rank; ++i)
    warpsPerCTA.push_back(i32_val(_warpsPerCTA[i]));
  unsigned mDim = mfmaLayout.getMDim();
  unsigned nDim = mfmaLayout.getNDim();
  assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
         (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));

  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = i32_val(triton::gpu::getWarpSize(mfmaLayout));
  Value effectiveWarpSize = warpSize;
  if (mDim == 4 && nDim == 4) {
    const int uniqueValuesPerWarp = 4;
    effectiveWarpSize = i32_val(uniqueValuesPerWarp);
  }
  Value laneId = urem(threadId, effectiveWarpSize);
  Value warpId = udiv(threadId, warpSize);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, _warpsPerCTA,
                  triton::gpu::getWarpOrder(mfmaLayout));
  if (shape[rank - 2] >= mDim) {
    assert(shape[rank - 2] % mDim == 0);
    multiDimWarpId[rank - 2] =
        urem(multiDimWarpId[rank - 2],
             i32_val(ceil<unsigned>(shape[rank - 2], mDim)));
  }
  if (shape[rank - 1] >= nDim) {
    assert(shape[rank - 1] % nDim == 0);
    multiDimWarpId[rank - 1] =
        urem(multiDimWarpId[rank - 1],
             i32_val(ceil<unsigned>(shape[rank - 1], nDim)));
  }
  Value offWarp0 = mul(multiDimWarpId[rank - 2], i32_val(mDim));
  Value offWarp1 = mul(multiDimWarpId[rank - 1], i32_val(nDim));

  SmallVector<Value> multiDimBase(rank);
  if (mfmaLayout.getIsTransposed()) {
    multiDimBase[rank - 1] =
        add(mul(i32_val(4), udiv(laneId, i32_val(mDim))), offWarp1);
    multiDimBase[rank - 2] = add(urem(laneId, i32_val(mDim)), offWarp0);
  } else {
    multiDimBase[rank - 2] =
        add(mul(i32_val(4), udiv(laneId, i32_val(nDim))), offWarp0);
    multiDimBase[rank - 1] = add(urem(laneId, i32_val(nDim)), offWarp1);
  }
  // TODO(Lixun): It is assumed when rank = 3, warpsPerCTA is set to
  // {numWarps, 1, 1}. We need to generalize the offset computation.
  if (rank == 3) {
    assert(_warpsPerCTA[1] == 1 && _warpsPerCTA[2] == 1);
    multiDimBase[0] = urem(warpId, i32_val(shape[0]));
  }
  return multiDimBase;
}

inline void emitMfmaOffsetForCTA(const AMDMfmaEncodingAttr &mfmaLayout,
                                 SmallVector<SmallVector<unsigned>> &offsets,
                                 unsigned bOff, unsigned ctaOffsetX,
                                 unsigned ctaOffsetY) {
  auto mDim = mfmaLayout.getMDim();
  auto nDim = mfmaLayout.getNDim();
  assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
         (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));
  // MFMA output tile consists of repeated "dot operand B" layout groups along
  // row axis. This variable defines number of these groups.
  DenseMap<int, int> groups{{4, 1}, {16, 1}, {32, 4}};
  unsigned numGroups = groups.at(std::min(mDim, nDim));
  const unsigned elemsPerThreadPerGroup = 4;
  auto warpSize = getWarpSize(mfmaLayout);
  assert(warpSize == 64);
  auto shapePerCta = getShapePerCTATile(mfmaLayout);
  auto rank = shapePerCta.size();
  SmallVector<unsigned> elemOff(rank, 0);
  for (unsigned block = 0; block < numGroups; block++) {
    unsigned rowOrColOffset =
        block * elemsPerThreadPerGroup * warpSize / std::min(mDim, nDim);
    for (unsigned elem = 0; elem < elemsPerThreadPerGroup; elem++) {
      if (mfmaLayout.getIsTransposed()) {
        elemOff[rank - 2] = ctaOffsetX * shapePerCta[rank - 2];
        elemOff[rank - 1] =
            ctaOffsetY * shapePerCta[rank - 1] + elem + rowOrColOffset;
      } else {
        elemOff[rank - 2] =
            ctaOffsetX * shapePerCta[rank - 2] + elem + rowOrColOffset;
        elemOff[rank - 1] = ctaOffsetY * shapePerCta[rank - 1];
      }
      if (rank == 3)
        elemOff[0] = bOff;
      offsets.push_back(elemOff);
    }
  }
}

inline void emitWmmaOffsetForCTA(const AMDWmmaEncodingAttr &wmmaLayout,
                                 SmallVector<SmallVector<unsigned>> &offsets,
                                 unsigned ctaBatchOffset, unsigned ctaOffsetX,
                                 unsigned ctaOffsetY) {
  const unsigned elemsPerThreadPerGroup = 8;
  auto warpSize = getWarpSize(wmmaLayout);
  assert(warpSize == 32);
  auto shapePerCta = getShapePerCTATile(wmmaLayout);
  auto rank = shapePerCta.size();
  assert(rank == 2 || rank == 3);
  SmallVector<unsigned> elemOffset(rank, 0);
  auto elemStride = wmmaLayout.getVersion() == 1 ? 2 : 1;
  if (rank == 3)
    elemOffset[0] = ctaBatchOffset;
  for (unsigned elem = 0; elem < elemsPerThreadPerGroup; elem++) {
    if (wmmaLayout.getIsTransposed()) {
      elemOffset[rank - 1] =
          ctaOffsetX * shapePerCta[rank - 1] + elemStride * elem;
      elemOffset[rank - 2] = ctaOffsetY * shapePerCta[rank - 2];
    } else {
      elemOffset[rank - 2] =
          ctaOffsetX * shapePerCta[rank - 2] + elemStride * elem;
      elemOffset[rank - 1] = ctaOffsetY * shapePerCta[rank - 1];
    }
    offsets.push_back(elemOffset);
  }
}

inline SmallVector<Value>
emitBaseIndexForWmmaLayout(Location loc, RewriterBase &rewriter,
                           const AMDWmmaEncodingAttr &wmmaLayout,
                           RankedTensorType type) {
  auto shape = type.getShape();
  auto _warpsPerCTA = wmmaLayout.getWarpsPerCTA();
  auto rank = _warpsPerCTA.size();
  assert(rank == 2 || rank == 3);
  SmallVector<Value> warpsPerCTA;
  for (unsigned i = 0; i < rank; ++i)
    warpsPerCTA.push_back(i32_val(_warpsPerCTA[i]));
  auto mnkDim = AMDWmmaEncodingAttr::getMNKDimPerInstr();

  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = i32_val(triton::gpu::getWarpSize(wmmaLayout));
  Value laneId =
      urem(threadId, i32_val(triton::gpu::getWarpSize(wmmaLayout) / 2));
  Value threadIdPerWarp = urem(threadId, warpSize);

  Value warpId = udiv(threadId, warpSize);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, _warpsPerCTA,
                  triton::gpu::getWarpOrder(wmmaLayout));
  if (shape[rank - 2] >= mnkDim[0]) {
    assert(shape[rank - 2] % mnkDim[0] == 0);
    multiDimWarpId[rank - 2] =
        urem(multiDimWarpId[rank - 2],
             i32_val(ceil<unsigned>(shape[rank - 2], mnkDim[0])));
  }
  if (shape[rank - 1] >= mnkDim[1]) {
    assert(shape[rank - 1] % mnkDim[1] == 0);
    multiDimWarpId[rank - 1] =
        urem(multiDimWarpId[rank - 1],
             i32_val(ceil<unsigned>(shape[rank - 1], mnkDim[1])));
  }
  Value offWarp0 = mul(multiDimWarpId[rank - 2], i32_val(mnkDim[0]));
  Value offWarp1 = mul(multiDimWarpId[rank - 1], i32_val(mnkDim[1]));

  SmallVector<Value> multiDimBase(rank);

  auto ver = wmmaLayout.getVersion();
  if (ver == 1) {
    multiDimBase[rank - 2] =
        add(udiv(threadIdPerWarp, i32_val(mnkDim[2])), offWarp0);
  } else {
    assert(ver == 2);
    if (wmmaLayout.getIsTransposed()) {
      multiDimBase[rank - 1] =
          add(mul(udiv(threadIdPerWarp, i32_val(16)),
                  i32_val(wmmaLayout.getSizePerThread()[rank - 1])),
              offWarp1);
      multiDimBase[rank - 2] = add(laneId, offWarp0);
    } else {
      multiDimBase[rank - 2] =
          add(mul(udiv(threadIdPerWarp, i32_val(16)),
                  i32_val(wmmaLayout.getSizePerThread()[rank - 2])),
              offWarp0);
      multiDimBase[rank - 1] = add(laneId, offWarp1);
    }
  }
  multiDimBase[rank - 1] = add(laneId, offWarp1);

  // TODO: It is assumed when rank = 3, warpsPerCTA is set to
  // {numWarps, 1, 1}. We need to generalize the offset computation.
  if (rank == 3) {
    assert(_warpsPerCTA[1] == 1 && _warpsPerCTA[2] == 1);
    multiDimBase[0] = urem(warpId, i32_val(shape[0]));
  }
  return multiDimBase;
}

SmallVector<SmallVector<unsigned>> emitOffsetForLayout(Attribute layout,
                                                       RankedTensorType type);

// -----------------------------------------------------------------------
// Get offsets / indices for any layout
// -----------------------------------------------------------------------

inline SmallVector<Value> emitCTAOffsetForLayout(Location loc,
                                                 RewriterBase &rewriter,
                                                 const TargetInfoBase &target,
                                                 Attribute layout,
                                                 ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  SmallVector<unsigned> CTAsPerCGA = triton::gpu::getCTAsPerCGA(layout);
  SmallVector<unsigned> CTASplitNum = triton::gpu::getCTASplitNum(layout);
  SmallVector<unsigned> CTAOrder = triton::gpu::getCTAOrder(layout);
  SmallVector<int64_t> shapePerCTA =
      triton::gpu::getShapePerCTA(CTASplitNum, shape);

  // Delinearize clusterCTAId
  Value clusterCTAId = target.getClusterCTAId(rewriter, loc);
  SmallVector<Value> multiDimClusterCTAId =
      delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

  // CTA Wrapping
  for (unsigned i = 0; i < rank; ++i) {
    // This wrapping rule must be consistent with getShapePerCTA
    unsigned splitNum = std::min<unsigned>(shape[i], CTASplitNum[i]);
    multiDimClusterCTAId[i] = urem(multiDimClusterCTAId[i], i32_val(splitNum));
  }

  SmallVector<Value> CTAOffset(rank);
  for (unsigned i = 0; i < rank; ++i)
    CTAOffset[i] = mul(multiDimClusterCTAId[i], i32_val(shapePerCTA[i]));

  return CTAOffset;
}

inline SmallVector<Value>
emitBaseIndexForLayoutImpl(Location loc, RewriterBase &rewriter,
                           const TargetInfoBase &target, Attribute layout,
                           RankedTensorType type, bool withCTAOffset) {
  auto shape = type.getShape();

  SmallVector<Value> baseIndex;
  RewriterBase::InsertionGuard guard(rewriter);
  SmallVector<Value> result;
  if (auto blockedLayout = mlir::dyn_cast<BlockedEncodingAttr>(layout)) {
    result = emitBaseIndexWithinCTAForBlockedLayout(loc, rewriter,
                                                    blockedLayout, type);
  } else if (auto mmaLayout = mlir::dyn_cast<NvidiaMmaEncodingAttr>(layout)) {
    if (mmaLayout.isAmpere() || mmaLayout.isHopper())
      result = emitBaseIndexWithinCTAForMmaLayoutV2V3(loc, rewriter, mmaLayout,
                                                      type);
  } else if (auto mfmaLayout = mlir::dyn_cast<AMDMfmaEncodingAttr>(layout)) {
    result = emitBaseIndexForMfmaLayout(loc, rewriter, mfmaLayout, type);
  } else if (auto wmmaLayout = mlir::dyn_cast<AMDWmmaEncodingAttr>(layout)) {
    result = emitBaseIndexForWmmaLayout(loc, rewriter, wmmaLayout, type);
  } else if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(layout)) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(type.getShape());
    RankedTensorType parentTy =
        RankedTensorType::get(parentShape, type.getElementType(), parentLayout);
    result = emitBaseIndexForLayoutImpl(loc, rewriter, target, parentLayout,
                                        parentTy, withCTAOffset);
    result.erase(result.begin() + sliceLayout.getDim());
    // CTAOffset has been added in emitBaseIndexForLayout of parentLayout
    return result;
  } else {
    llvm_unreachable("unsupported emitBaseIndexForLayout");
  }
  if (withCTAOffset) {
    auto CTAOffset =
        emitCTAOffsetForLayout(loc, rewriter, target, layout, shape);
    assert(CTAOffset.size() == result.size() && "Rank mismatch");
    for (unsigned k = 0; k < result.size(); ++k) {
      // Individual elements of `result` may be null.  In the caller
      // (emitBaseIndexForLayout), we assert that all such dimensions are sliced
      // off.
      if (!result[k])
        continue;
      result[k] = add(result[k], CTAOffset[k]);
    }
  }
  return result;
}

inline SmallVector<Value>
emitBaseIndexForLayout(Location loc, RewriterBase &rewriter,
                       const TargetInfoBase &target, Attribute layout,
                       RankedTensorType type, bool withCTAOffset) {
  SmallVector<Value> idx = emitBaseIndexForLayoutImpl(
      loc, rewriter, target, layout, type, withCTAOffset);

  // Check that any null values were sliced out.
  for (Value v : idx) {
    if (!v) {
      llvm::errs() << "Failed to generate indexing code, possibly due to bad "
                      "#mma layout.  Please rerun your program with "
                      "MLIR_ENABLE_DUMP=1 and file a bug."
                   << "\nloc: " << loc << "\nlayout: " << layout
                   << "\ntype: " << type << "\nwithCTAOffset: " << withCTAOffset
                   << "\n";
      llvm::report_fatal_error("Failed to generate indexing code");
    }
  }

  return idx;
}

// Emit code to compute the (laneId, warpId, blockId) for the current thread.
std::tuple</*laneId=*/Value, /*warpId=*/Value, /*blockId=*/Value>
emitHardwareTuple(Location loc, RewriterBase &rewriter,
                  const TargetInfoBase &target, bool withCTAOffset,
                  unsigned threadsPerWarp);

// Emit indices calculation within each ConversionPattern, and returns a
// [elemsPerThread X rank] index matrix.
//
// For example, for a thread a owns `elemsPerThread` elements of a tensor with
// type `type` and layout `layout`, the result will contain `elemsPerThread`
// vectors. Each vector contains the SSA values of the indices required to
// access the corresponding element, starting from the inner dimension.
SmallVector<SmallVector<Value>>
emitIndices(Location loc, RewriterBase &rewriter, const TargetInfoBase &target,
            Attribute layout, RankedTensorType type, bool withCTAOffset);

// Emits IR to load data from shared memory into registers, or to store data
// from registers into shared memory.
//
// You supply perVectorCallback, which is called once per group of register
// elements to transfer.  You can use this callback to emit IR to load or store
// data from or to shared memory.
//
// elemLlvmTy should be dstTy's element type converted to an LLVM-dialect type.
//
// If maxVecElems is provided, we won't vectorize more than this many elements.
//
// Returns true on success.
[[nodiscard]] bool emitTransferBetweenRegistersAndShared(
    RankedTensorType registerTy, triton::gpu::MemDescType sharedTy,
    Type elemLlvmTy, std::optional<int32_t> maxVecElems,
    const SharedMemoryObject &smemObj, Location loc, RewriterBase &rewriter,
    const TargetInfoBase &target,
    std::function<void(VectorType, Value /*shmemAddr*/)> perVectorCallback);

[[nodiscard]] bool emitTransferBetweenRegistersAndShared(
    LinearLayout &regLayout, triton::gpu::MemDescType sharedTy, Type elemLlvmTy,
    std::optional<int32_t> maxVecElems, const SharedMemoryObject &smemObj,
    Location loc, RewriterBase &rewriter, const TargetInfoBase &target,
    std::function<void(VectorType, Value /*shmemAddr*/)> perVectorCallback);

SmallVector<Value> loadSharedToDistributed(RankedTensorType dstTy,
                                           triton::gpu::MemDescType srcTy,
                                           Type elemLlvmTy,
                                           const SharedMemoryObject &smemObj,
                                           Location loc, RewriterBase &rewriter,
                                           const TargetInfoBase &target);

void storeDistributedToShared(
    triton::gpu::MemDescType dstTy, RankedTensorType srcTy, Type elemLlvmTy,
    ArrayRef<Value> srcVals, const SharedMemoryObject &smemObj, Location loc,
    RewriterBase &rewriter, const TargetInfoBase &target,
    std::pair<size_t, Type> *const llvmOpCount = nullptr);

inline SmallVector<Value> unpackLLElements(Location loc, Value llvmStruct,
                                           RewriterBase &rewriter) {
  assert(bool(llvmStruct) && "can not unpack null values");
  if (llvmStruct.getType().isIntOrIndexOrFloat() ||
      isa<triton::PointerType>(llvmStruct.getType()) ||
      isa<LLVM::LLVMPointerType>(llvmStruct.getType()))
    return {llvmStruct};
  ArrayRef<Type> types =
      cast<LLVM::LLVMStructType>(llvmStruct.getType()).getBody();
  SmallVector<Value> results(types.size());
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    results[i] = extract_val(type, llvmStruct, i);
  }
  return results;
}

inline Value packLLElements(Location loc,
                            const LLVMTypeConverter *typeConverter,
                            ValueRange resultVals, RewriterBase &rewriter,
                            Type type) {
  auto structType =
      dyn_cast<LLVM::LLVMStructType>(typeConverter->convertType(type));
  if (!structType) {
    assert(resultVals.size() == 1);
    return *resultVals.begin();
  }

  auto elementTypes = structType.getBody();
  if (elementTypes.size() != resultVals.size()) {
    emitError(loc) << " size mismatch when packing elements for LLVM struct"
                   << " expected " << elementTypes.size() << " but got "
                   << resultVals.size();
  }
  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structType);
  for (const auto &v : llvm::enumerate(resultVals)) {
    if (!v.value()) {
      emitError(loc)
          << "cannot insert null values into struct, but tried to insert"
          << v.value();
    }
    if (v.value().getType() != elementTypes[v.index()]) {
      LDBG("type " << type << " structType " << structType);
      LDBG("value " << v.value());
      emitError(loc) << "invalid element type in packLLElements. Expected "
                     << elementTypes[v.index()] << " but got "
                     << v.value().getType();
    }
    llvmStruct = insert_val(structType, llvmStruct, v.value(), v.index());
  }
  return llvmStruct;
}

inline SmallVector<Value> unpackLLVector(Location loc, Value llvmVec,
                                         RewriterBase &rewriter) {
  assert(bool(llvmVec) && "cannot unpack null value");
  if (llvmVec.getType().isIntOrIndexOrFloat() ||
      isa<triton::PointerType>(llvmVec.getType()) ||
      isa<LLVM::LLVMPointerType>(llvmVec.getType()))
    return {llvmVec};

  SmallVector<Value> results;
  for (int i = 0; i < cast<VectorType>(llvmVec.getType()).getNumElements();
       i++) {
    results.push_back(extract_element(llvmVec, i32_val(i)));
  }
  return results;
}

inline Value packLLVector(Location loc, ValueRange vals,
                          RewriterBase &rewriter) {
  assert(vals.size() > 0);
  auto vecType = vec_ty(vals[0].getType(), vals.size());
  Value vec = undef(vecType);
  for (int i = 0; i < vals.size(); i++) {
    vec = insert_element(vec, vals[i], i32_val(i));
  }
  return vec;
}

inline bool
isSimpleSharedMemoryAccess(ArrayRef<int64_t> shape,
                           ArrayRef<int64_t> allocShape,
                           triton::gpu::SharedEncodingAttr sharedEnc) {
  auto rank = shape.size();
  return /*no swizzling*/ sharedEnc.getMaxPhase() == 1 ||
         /*swizzling but same shape*/ shape == allocShape ||
         /*swizzling and rank-reduced and rank >= 2*/
         (shape == allocShape.take_back(rank) && rank >= 2);
}

} // namespace mlir

#endif
