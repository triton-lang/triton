#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "BufferOpsEmitter.h"

using mlir::triton::gpu::appendOrGetExternFuncOp;
using mlir::triton::gpu::getFunctionType;
using namespace triton::AMD;

namespace {

// Utility function to determine if a scalar/tensor value is zero
bool isZero(Value v) {
  if (auto constantOp = v.getDefiningOp<LLVM::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(constantOp.getValue()))
      return attr.getValue().isZero();
    if (auto attr = dyn_cast<FloatAttr>(constantOp.getValue()))
      return attr.getValue().isZero();
    if (auto denseAttr =
            dyn_cast<DenseFPElementsAttr>(constantOp.getValueAttr()))
      return denseAttr.isSplat() && denseAttr.getSplatValue<APFloat>().isZero();
    if (auto denseAttr =
            dyn_cast<DenseIntElementsAttr>(constantOp.getValueAttr()))
      return denseAttr.isSplat() && denseAttr.getSplatValue<APInt>().isZero();
  }
  return false;
}
} // namespace

namespace mlir::LLVM::AMD {
BufferEmitter::BufferEmitter(RewriterBase &rw, Location loc, TargetInfo ti)
    : rewriter(rw), loc(loc), targetInfo(ti) {}

Value BufferEmitter::createResourceDescriptor(Value basePtr) {
  // 1. Create the resource descriptor
  // bits 0-11: dst sel, ignored by these intrinsics
  // bits 12-14: data format (ignored, must be nonzero, 7=float)
  // bits 15-18: data format (ignored, must be nonzero, 4=32bit)
  // bit 19: In nested heap (0 here)
  // bit 20: Behavior on unmap (0 means  "return 0 / ignore")
  // bits 21-22: Index stride for swizzles (N/A)
  // bit 23: Add thread ID (0)
  // bit 24: Reserved to 1 (RDNA) or 0 (CDNA)
  // bits 25-26: Reserved (0)
  // bit 27: Buffer is non-volatile (CDNA only)
  // bits 28-29: Out of bounds select (RDNA only)
  //             (0 = structured,
  //              1 = check index,
  //              2 = none,
  //              3 = either swizzles or testing against offset field)
  // bits 30-31: Type (must be 0)
  uint32_t flags = (7 << 12) | (4 << 15);
  if (targetInfo.getISAFamily() == ISAFamily::RDNA2 ||
      targetInfo.getISAFamily() == ISAFamily::RDNA3) {
    flags |= (1 << 24);
    uint32_t oob = 3;
    flags |= (oob << 28);
  }
  Value stride = int_val(16, 0);
  Value flagsConst = int_val(32, flags);
  Type rsrcType = LLVM::LLVMPointerType::get(rewriter.getContext(), 8);
  Value numRecordsByte = int_val(32, std::numeric_limits<int>::max() - 1);

  Value resource = rewriter.createOrFold<ROCDL::MakeBufferRsrcOp>(
      loc, rsrcType, basePtr, stride, numRecordsByte, flagsConst);
  return resource;
}

Value BufferEmitter::emitLoad(Type type, Value rsrcDesc, Value offset,
                              Value pred, Value falseVal,
                              triton::CacheModifier cm) {
  SmallVector<Value, 6> args;
  fillCommonArgs(type, rsrcDesc, offset, pred, cm, /*isBufferLoad=*/true, args);
  Type bufferType = getBufferOpType(type);
  Value data = rewriter.create<ROCDL::RawPtrBufferLoadOp>(
      loc, bufferType, args, ArrayRef<NamedAttribute>());
  data = bitcast(data, type);
  if (!isZero(falseVal))
    data = select(pred, data, falseVal);
  return data;
}

void BufferEmitter::emitStore(Value rsrcDesc, Value offset, Value data,
                              Value pred, triton::CacheModifier cm) {
  VectorType vecTy = cast<VectorType>(data.getType());
  Type bufferType = getBufferOpType(vecTy);
  if (vecTy != bufferType)
    data = bitcast(data, bufferType);
  SmallVector<Value, 6> args{data};
  fillCommonArgs(vecTy, rsrcDesc, offset, pred, cm, /*isBufferLoad=*/false,
                 args);
  rewriter.create<ROCDL::RawPtrBufferStoreOp>(loc, TypeRange{}, args,
                                              ArrayRef<NamedAttribute>());
}

Type BufferEmitter::getBufferOpType(Type type) {
  int64_t vecSize = 1;
  Type elementType = type;
  if (auto vecType = dyn_cast<VectorType>(type)) {
    vecSize = vecType.getNumElements();
    elementType = vecType.getElementType();
  }

  const int valueElemNBits = std::max(8u, elementType.getIntOrFloatBitWidth());
  const size_t totalWidthBits = valueElemNBits * vecSize;

  // For bf16, always convert to i16
  Type bufferElementType = elementType;
  if (elementType.isBF16())
    bufferElementType = rewriter.getI16Type();

  // If we are dealing with a subword type (e.g., i8 or f16) but we
  // still need multiple words, then pack the subwords into 32bit integers
  // and update the vector length and the type
  int64_t bufferVecSize = vecSize;
  if (valueElemNBits < 32) {
    if (totalWidthBits > 32) {
      bufferElementType = rewriter.getI32Type();
      bufferVecSize = totalWidthBits / 32;
    } else {
      bufferElementType = rewriter.getIntegerType(totalWidthBits);
      bufferVecSize = 1;
    }
  }

  // This is the buffer type that the buffer operation will use. It
  // will be bitcast-able to the original type. So if the types
  // ended up different, we simply have to emit a `bitcastOp` to convert
  Type bufferType = type;
  if (bufferVecSize != vecSize || bufferElementType != elementType)
    bufferType = VectorType::get(bufferVecSize, bufferElementType);
  if (bufferVecSize == 1)
    bufferType = getElementTypeOrSelf(bufferType);

  return bufferType;
}

void BufferEmitter::fillCommonArgs(Type type, Value rsrcDesc,
                                   Value vOffsetElems, Value pred,
                                   triton::CacheModifier cm, bool isBufferLoad,
                                   SmallVector<Value> &args) {

  // 1. Create the (masked) offset
  Type elementType = getElementTypeOrSelf(type);
  const int valueElemNBits = std::max(8u, elementType.getIntOrFloatBitWidth());
  const int elementByteWidth = valueElemNBits / 8;
  // Please note: the index passed is not in bytes, but in number of elements
  // In order to pass the index to the buffer operation, we need to convert in
  // bytes (i.e., we need to multiply by `elementByteWidth`)
  Value vOffsetOutOfBunds = int_val(
      32, static_cast<int>(std::numeric_limits<int>::max() + int64_t(1)));
  Value vOffsetBytes = mul(int_val(32, elementByteWidth), vOffsetElems);
  Value maskedOffsetBytes = select(pred, vOffsetBytes, vOffsetOutOfBunds);

  // 2. Set the sgprOffset to 0
  Value sgprOffset = int_val(32, 0);

  // 3. Create the cache modifiers word
  //            bit 0 = glc, bit 1 = slc, bit 2 = dlc (gfx10/gfx11),
  //            bit 3 = swz, bit 4 = scc (gfx90a)
  //    gfx940: bit 0 = sc0, bit 1 = nt, bit 3 = swz, bit 4 = sc1
  //    gfx12+: bits [0-2] = th, bits [3-4] = scope,
  //            bit 6 = swz
  //       all: volatile op (bit 31, stripped at lowering)
  //
  // MI300 cache modifiers:
  // Op   | cm  | SC1 | SC0 | NT |
  // -----+-----+-----+-----+----+--
  // Load | .ca |  0  |  0  | 0  |
  //      | .cg |  0  |  x  | 1  |
  //      | .cs |  1  |  0  | x  |
  //      | .cv |  1  |  1  | x  |
  // -----+-----+-----+-----+----+--
  // Store| .wb |  1  |  0  | 0  |
  //      | .cg |  0  |  0  | 0  |
  //      | .cs |  1  |  0  | 1  |
  //      | .wt |  1  |  1  | x  |

  int32_t aux = 0;
  switch (cm) {
  case triton::CacheModifier::CA:
    aux = 0;
    break;
  case triton::CacheModifier::CG:
    if (isBufferLoad)
      aux |= 2;
    break;
  case triton::CacheModifier::CS:
    aux |= 16;
    if (!isBufferLoad)
      aux |= 2;
    break;
  case triton::CacheModifier::CV:
    aux |= 17;
    break;
  case triton::CacheModifier::WB:
    aux |= 16;
    break;
  case triton::CacheModifier::WT:
    aux |= 17;
    break;
  default:
    aux = 0;
  }
  Value cacheModifiers = int_val(32, aux);

  // 5. Add the arguments
  args.push_back(rsrcDesc);
  args.push_back(maskedOffsetBytes);
  args.push_back(sgprOffset);
  args.push_back(cacheModifiers);
}
} // namespace mlir::LLVM::AMD
