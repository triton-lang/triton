/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/Ops.cpp.inc"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

// -- WarpGroupDotOp --
mlir::LogicalResult WarpGroupDotOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // type is the same as the accumulator
  auto accTy = cast<RankedTensorType>(operands[2].getType());
  inferredReturnTypes.push_back(accTy);

  // verify encodings
  auto aEnc =
      cast<triton::gpu::TensorOrMemDesc>(operands[0].getType()).getEncoding();
  auto bEnc =
      cast<triton::gpu::TensorOrMemDesc>(operands[1].getType()).getEncoding();
  auto retEnc = accTy.getEncoding();
  if (aEnc) {
    assert(bEnc);
    Dialect &dialect = aEnc.getDialect();
    auto interface = cast<DialectInferLayoutInterface>(&dialect);
    if (interface->inferDotOpEncoding(aEnc, 0, retEnc, location).failed())
      return mlir::failure();
    if (interface->inferDotOpEncoding(bEnc, 1, retEnc, location).failed())
      return mlir::failure();
  }
  return mlir::success();
}

void WarpGroupDotOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto &a = getAMutable();
  auto &b = getBMutable();
  if (isa<mlir::triton::gpu::MemDescType>(a.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &a,
                         mlir::triton::gpu::SharedMemory::get());
  if (isa<mlir::triton::gpu::MemDescType>(b.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &b,
                         mlir::triton::gpu::SharedMemory::get());
}

bool WarpGroupDotOp::needsPartialAccumulator() {
  const auto &a = getA();
  const auto &d = getD();
  auto aTensorTy = cast<triton::gpu::TensorOrMemDesc>(a.getType());
  auto aElTy = cast<triton::gpu::TensorOrMemDesc>(a.getType()).getElementType();
  bool isFP8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType, Float8E5M2FNUZType,
                         Float8E4M3FNUZType>(aElTy);
  bool accFP32 =
      cast<triton::gpu::TensorOrMemDesc>(d.getType()).getElementType().isF32();
  uint32_t maxNumImpreciseAcc = getMaxNumImpreciseAcc();
  return isFP8 && accFP32 && maxNumImpreciseAcc <= aTensorTy.getShape()[1];
}

bool WarpGroupDotOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  return aShape[aShape.size() - 1] == bShape[aShape.size() - 2];
}

// -- WarpGroupDotWaitOp --
LogicalResult WarpGroupDotWaitOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  for (Value operand : operands)
    inferredReturnTypes.push_back(operand.getType());
  return mlir::success();
}

static LogicalResult
verifyBarrierType(Operation *op, mlir::triton::gpu::MemDescType barrierType) {
  if (!barrierType.getElementType().isInteger(64) ||
      barrierType.getShape() != ArrayRef<int64_t>({1}))
    return op->emitOpError(
        "barrier allocation must be a descriptor of 1xi64 type");
  return success();
}

// -- InitBarrierOp --
LogicalResult InitBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

void InitBarrierOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getAllocMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

// -- InvalBarrierOp --
LogicalResult InvalBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

void InvalBarrierOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getAllocMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

// -- BarrierExpectOp --
LogicalResult BarrierExpectOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

void BarrierExpectOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getAllocMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

// -- WaitBarrierOp --
LogicalResult WaitBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

void WaitBarrierOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The wait will flip the phase therefore it reads and writes the barrier.
  effects.emplace_back(MemoryEffects::Read::get(), &getAllocMutable(),
                       mlir::triton::gpu::SharedMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getAllocMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

// -- ArriveBarrierOp --
LogicalResult ArriveBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  if (getCount() < 1)
    return emitOpError("count must be greater than or equal to 1");
  return success();
}

void ArriveBarrierOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The arrive will increment the pending arrival count inside the barrier.
  effects.emplace_back(MemoryEffects::Read::get(), &getAllocMutable(),
                       mlir::triton::gpu::SharedMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getAllocMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

// -- TensorDescToTMAPtrOp --
LogicalResult TensorDescToTMAPtrOp::canonicalize(TensorDescToTMAPtrOp op,
                                                 PatternRewriter &rewriter) {
  // tensor_desc_to_tma_ptr(reinterpret_tensor_desc(ptr)) -> ptr
  if (auto reinterpret =
          op.getDesc().getDefiningOp<triton::ReinterpretTensorDescOp>()) {
    rewriter.replaceOp(op, reinterpret.getRawDesc());
    return success();
  }
  return failure();
}

// -- AsyncTMACopyGlobalToLocalOp --
LogicalResult AsyncTMACopyGlobalToLocalOp::verify() {
  if (failed(verifyBarrierType(*this, getBarrier().getType())))
    return failure();
  if (getCoord().size() < 1 || getCoord().size() > 5)
    return emitOpError("TMA copies must have between 1 and 5 coordinates");
  if (!getResult().getType().getMutableMemory())
    return emitOpError("Cannot store into immutable memory");
  return success();
}

void AsyncTMACopyGlobalToLocalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getDescPtrMutable(),
                       mlir::triton::GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getBarrierMutable(),
                       mlir::triton::gpu::SharedMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getResultMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

// -- AsyncTMACopyLocalToGlobalOp --
void AsyncTMACopyLocalToGlobalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDescPtrMutable(),
                       mlir::triton::GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

// -- AsyncTMAGatherOp --
LogicalResult AsyncTMAGatherOp::verify() {
  if (failed(verifyBarrierType(*this, getBarrier().getType())))
    return failure();

  triton::gpu::MemDescType resultType = getResult().getType();
  if (!resultType.getMutableMemory())
    return emitOpError("cannot store into immutable memory");
  return DescriptorGatherOp::verifyResultType(*this, resultType,
                                              getXOffsets().getType());
}

void AsyncTMAGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getDescPtrMutable(),
                       mlir::triton::GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getBarrierMutable(),
                       mlir::triton::gpu::SharedMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getResultMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

// -- AsyncTMAScatter --
LogicalResult AsyncTMAScatterOp::verify() {
  return DescriptorGatherOp::verifyResultType(*this, getSrc().getType(),
                                              getXOffsets().getType());
}

void AsyncTMAScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDescPtrMutable(),
                       mlir::triton::GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

// -- TCGen5MMAOp --
void TCGen5MMAOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDMutable(),
                       mlir::triton::nvidia_gpu::TensorMemory::get());
  if (isa<triton::gpu::SharedMemorySpaceAttr>(
          getA().getType().getMemorySpace())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         mlir::triton::gpu::SharedMemory::get());

  } else {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         mlir::triton::nvidia_gpu::TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Read::get(), &getBMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

bool TCGen5MMAOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  return aShape[aShape.size() - 1] == bShape[aShape.size() - 2];
}

Value TCGen5MMAOp::useAccumulator() { return getUseD(); }

void TCGen5MMAOp::setUseAccumulator(Value flag) {
  getUseDMutable().assign(flag);
}

void TCGen5MMAOp::setBarrier(Value barrier) {
  getBarrierMutable().assign(barrier);
}

Value TCGen5MMAOp::getAccumulator() { return getD(); }

void TCGen5MMAOp::setAccumulator(Value accum) { getDMutable().assign(accum); }

Value TCGen5MMAOp::getPredicate() { return getPred(); }

void TCGen5MMAOp::setPredicate(Value pred) { getPredMutable().assign(pred); }

// -- TMEMStoreOp --
LogicalResult TMEMStoreOp::verify() {
  if (!isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(
          getDst().getType().getMemorySpace()))
    return emitOpError("destination must be a tensor memory buffer.");
  if (!isa<triton::nvidia_gpu::TensorMemoryEncodingAttr,
           TensorMemoryScalesEncodingAttr>(getDst().getType().getEncoding()))
    return emitOpError("should use tensor memory encoding.");
  if (!getDst().getType().getMutableMemory()) {
    return emitOpError("Cannot store into an immutable alloc");
  }
  return success();
}

// -- TCGen5MMAScaledOp --
void TCGen5MMAScaledOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDMutable(),
                       mlir::triton::nvidia_gpu::TensorMemory::get());
  if (isa<triton::gpu::SharedMemorySpaceAttr>(
          getA().getType().getMemorySpace())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         mlir::triton::gpu::SharedMemory::get());

  } else {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         mlir::triton::nvidia_gpu::TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Read::get(), &getBMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

bool TCGen5MMAScaledOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  bool transB = false;
  if (auto bSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getB().getType().getEncoding())) {
    transB = !bSharedLayout.getTransposed();
  }
  auto aKdim = aShape[aShape.size() - 1];
  auto bKdim = bShape[aShape.size() - 2];
  if (this->getAType() == ScaleDotElemType::E2M1 && !transA)
    aKdim *= 2;
  if (this->getBType() == ScaleDotElemType::E2M1 && !transB)
    bKdim *= 2;

  return aKdim == bKdim;
}

bool TCGen5MMAScaledOp::verifyOutputDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();
  auto cShape = this->getD().getType().getShape();
  auto oMdim = cShape[cShape.size() - 2];
  auto oNdim = cShape[cShape.size() - 1];

  int aMdim = aShape[aShape.size() - 2];
  int bNdim = bShape[bShape.size() - 1];
  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  bool transB = false;
  if (auto bSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getB().getType().getEncoding())) {
    transB = !bSharedLayout.getTransposed();
  }
  if (this->getAType() == ScaleDotElemType::E2M1 && transA)
    aMdim *= 2;
  if (this->getBType() == ScaleDotElemType::E2M1 && transB)
    bNdim *= 2;

  if (aMdim != oMdim || bNdim != oNdim)
    return false;
  return true;
}

Value TCGen5MMAScaledOp::useAccumulator() { return getUseD(); }

void TCGen5MMAScaledOp::setUseAccumulator(Value flag) {
  getUseDMutable().assign(flag);
}

void TCGen5MMAScaledOp::setBarrier(Value barrier) {
  getBarrierMutable().assign(barrier);
}

Value TCGen5MMAScaledOp::getAccumulator() { return getD(); }

void TCGen5MMAScaledOp::setAccumulator(Value accum) {
  getDMutable().assign(accum);
}

Value TCGen5MMAScaledOp::getPredicate() { return getPred(); }

void TCGen5MMAScaledOp::setPredicate(Value pred) {
  getPredMutable().assign(pred);
}

int64_t TCGen5MMAScaledOp::getBlockM() {
  ArrayRef<int64_t> shape = getA().getType().getShape();
  int64_t blockM = shape[shape.size() - 2];
  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  if (this->getAType() == ScaleDotElemType::E2M1 && transA)
    blockM *= 2;
  return blockM;
}

int64_t TCGen5MMAScaledOp::getBlockN() {
  ArrayRef<int64_t> shape = getB().getType().getShape();
  int64_t blockN = shape[shape.size() - 1];
  bool transB = false;
  if (auto bSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getB().getType().getEncoding())) {
    transB = !bSharedLayout.getTransposed();
  }
  if (this->getBType() == ScaleDotElemType::E2M1 && transB)
    blockN *= 2;
  return blockN;
}

int64_t TCGen5MMAScaledOp::getBlockK() {
  ArrayRef<int64_t> shape = getA().getType().getShape();
  int64_t blockK = shape[shape.size() - 1];
  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  if (this->getAType() == ScaleDotElemType::E2M1 && !transA)
    blockK *= 2;
  return blockK;
}

// -- TMEMLoadOp --
// -- TMEMLoadOp --
LogicalResult TMEMLoadOp::verify() {
  if (!isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(
          getSrc().getType().getMemorySpace()))
    return emitOpError("source must be a tensor memory buffer.");
  if (!isa<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
          getSrc().getType().getEncoding()))
    return emitOpError("should use tensor memory encoding.");
  return success();
}

// -- TMEMAllocOp --
LogicalResult TMEMAllocOp::verify() {
  if (!isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(
          getType().getMemorySpace()))
    return emitOpError("should create a buffer of tensor memory.");
  if (!isa<triton::nvidia_gpu::TensorMemoryEncodingAttr,
           TensorMemoryScalesEncodingAttr>(getType().getEncoding()))
    return emitOpError("should use tensor memory encoding.");
  if (!getSrc()) {
    if (!getType().getMutableMemory())
      return emitError("uninitialized alloc must have a mutable memdesc type");
  }
  return success();
}

void TMEMAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  Operation *op = getOperation();
  // If allocation is immutable, mark it as no side effect allow things like
  // CSE, DCE to work in early compiler passes.
  // After the memory offset is computed, we attach the true side effect to the
  // op.
  if (!getType().getMutableMemory() && !op->hasAttr("tensor_memory_col_offset"))
    return;
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       mlir::triton::nvidia_gpu::TensorMemory::get());
  if (getSrc())
    effects.emplace_back(MemoryEffects::Write::get(),
                         getOperation()->getOpResult(0),
                         mlir::triton::nvidia_gpu::TensorMemory::get());
}

static bool isDescendingOrder(triton::gpu::MemDescType type) {
  auto order = triton::gpu::getOrder(type);
  auto rank = type.getRank();
  for (int i = 0; i < rank; ++i) {
    if (order[i] != rank - 1 - i)
      return false;
  }
  return true;
}

// -- TMEMCopyOp --
LogicalResult TMEMCopyOp::verify() {
  if (!isa<triton::gpu::SharedMemorySpaceAttr>(
          getSrc().getType().getMemorySpace()))
    return emitOpError("The source must be a shared memory buffer");
  if (!isa<TensorMemoryEncodingAttr, TensorMemoryScalesEncodingAttr>(
          getDst().getType().getEncoding()))
    return emitOpError("The destination must be a tensor memory buffer.");

  if (getBarrier() && !isa<triton::gpu::SharedMemorySpaceAttr>(
                          getBarrier().getType().getMemorySpace())) {
    return emitOpError("The optional barrier should be a shared memory buffer");
  }
  if (!getDst().getType().getMutableMemory()) {
    return emitOpError("Cannot copy into an immutable alloc");
  }

  auto srcTy = cast<triton::gpu::MemDescType>(getSrc().getType());
  auto sharedEnc =
      cast<triton::gpu::SwizzledSharedEncodingAttr>(srcTy.getEncoding());

  if (sharedEnc.getMaxPhase() != 1 || sharedEnc.getPerPhase() != 1 ||
      sharedEnc.getVec() != 1)
    return emitOpError("The source should not have swizzling applied for now");

  if (!isDescendingOrder(srcTy)) {
    return emitOpError("The source must be in a row-major order.");
  }

  // Given that we want to support flexible input SMEM shapes, kinds of shape
  // checking we can do here are limited. For simplicity, shape checking is
  // omitted.
  return success();
}

void TMEMCopyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       mlir::triton::nvidia_gpu::TensorMemory::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

// -- TMEMSubSliceOp --
LogicalResult TMEMSubSliceOp::verify() {
  auto srcTy = cast<triton::gpu::MemDescType>(getSrc().getType());
  auto encoding = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
      srcTy.getEncoding());
  if (!encoding)
    return emitOpError("The source must be a tensor memory buffer.");
  if (encoding.getBlockM() != 128)
    return emitOpError("The source must be a 128xN layout.");
  auto dstTy = cast<triton::gpu::MemDescType>(getResult().getType());
  auto dstEncoding = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
      dstTy.getEncoding());
  if (!dstEncoding)
    return emitOpError("The destination must be a tensor memory buffer.");
  if (dstEncoding.getBlockM() != encoding.getBlockM() ||
      dstEncoding.getCTASplitM() != encoding.getCTASplitM() ||
      dstEncoding.getCTASplitN() != encoding.getCTASplitN() ||
      dstEncoding.getUnpacked() != encoding.getUnpacked())
    return emitOpError("The destination must have the same block size and "
                       "CTASplit size as the source.");
  return mlir::success();
}

void TMEMSubSliceOp::build(OpBuilder &builder, OperationState &state,
                           Value alloc, int offset, int size) {
  auto allocTy = cast<triton::gpu::MemDescType>(alloc.getType());
  SmallVector<int64_t> shape(allocTy.getShape());
  shape.back() = size;
  auto encoding =
      cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(allocTy.getEncoding());
  unsigned newBlockN = std::min<unsigned>(encoding.getBlockN(), size);
  auto newEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
      builder.getContext(), encoding.getBlockM(), newBlockN,
      encoding.getUnpacked(), encoding.getCTASplitM(), encoding.getCTASplitN());
  auto subsliceType = gpu::MemDescType::get(
      shape, allocTy.getElementType(), newEncoding, allocTy.getMemorySpace(),
      allocTy.getMutableMemory());
  build(builder, state, subsliceType, alloc, offset);
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
