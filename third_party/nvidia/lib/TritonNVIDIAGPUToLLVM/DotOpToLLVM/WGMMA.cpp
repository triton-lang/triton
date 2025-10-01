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

#include "MMAHelpers.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::NVIDIA;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::MemDescType;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::NVMMASharedEncodingAttr;

triton::nvgpu::WGMMAEltType getMmaRetType(Value d) {
  auto dTy = cast<RankedTensorType>(d.getType()).getElementType();
  if (dTy.isF32()) {
    return triton::nvgpu::WGMMAEltType::f32;
  } else if (dTy.isF16()) {
    return triton::nvgpu::WGMMAEltType::f16;
  } else if (dTy.isInteger(32)) {
    return triton::nvgpu::WGMMAEltType::s32;
  } else {
    llvm::report_fatal_error("Unsupported mma result type found");
  }
}

triton::nvgpu::WGMMAEltType getMmaOperandType(Value a, bool allowTF32) {
  auto aTy = cast<triton::gpu::TensorOrMemDesc>(a.getType()).getElementType();
  if (aTy.isF16()) {
    return triton::nvgpu::WGMMAEltType::f16;
  } else if (aTy.isBF16()) {
    return triton::nvgpu::WGMMAEltType::bf16;
  } else if (aTy.isF32() && allowTF32) {
    return triton::nvgpu::WGMMAEltType::tf32;
  } else if (aTy.isInteger(8)) {
    return triton::nvgpu::WGMMAEltType::s8;
  } else if (llvm::isa<Float8E5M2Type>(aTy)) {
    return triton::nvgpu::WGMMAEltType::e5m2;
  } else if (llvm::isa<Float8E4M3FNType>(aTy)) {
    return triton::nvgpu::WGMMAEltType::e4m3;
  } else {
    llvm::report_fatal_error("Unsupported mma operand type found");
  }
}

// Return a vector of Value of the accumulator start at startIndex and pack the
// values into 32bits in case the accumulator is fp16.
//
// `elements` contains all loaded register values for operand A.
// This consists of operand A for possibly multiple wgmma instructions.
// For each wgmma, each warp in a warp group feeds a single "warp matrix"
// Each warp matrix consists of 2x2 "quads".
// Each thread holds several elements in each quad. Right before a wgmma,
// the sum of bitwidth of
// the elements in each quad should add up to 32.
//
// These values are stored unrolled in `elements`.
// The ordering of dimensions is as follows:
// batch (only 1 batch for Hopper currently)
// matM (m-index of the "warp matrix")
// matK (k-index of the "warp matrix")
// quadK (k-index of the "quad" in the core matrix)
// quadM (m-index of the "quad" in the core matrix)
// vecIdx (index of the element in the quad; this is always along the k-dim)
//
// This ordering is decided when a tensor in DotOpEnc is lowered into llvm.
// For WGMMA this happens in both SharedToDotOperand and MMAToDotOperand.
// Thus, both lowerings must obey this above ordering for the below code to be
// correct.
llvm::SmallVector<Value> loadReg(ConversionPatternRewriter &rewriter,
                                 Location loc,
                                 const SmallVector<Value> &elements,
                                 int startIndex, int numElements,
                                 Operation *insertBefore) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(insertBefore);

  if (!elements[0].getType().isIntOrFloat() ||
      elements[0].getType().getIntOrFloatBitWidth() >= 32) {
    llvm::SmallVector<Value> mmaOut(numElements);
    for (int i = 0; i < numElements; ++i)
      mmaOut[i] = elements[startIndex + i];
    return mmaOut;
  }
  Type elementType = elements[0].getType();
  int numElemsPer32Bits = 32 / elementType.getIntOrFloatBitWidth();

  // For FP16 and BF16 we need to pack accumulator into 32-bit integers.
  int num32BitValues = numElements / numElemsPer32Bits;
  llvm::SmallVector<Value> mmaOut(num32BitValues);
  Type packTy = vec_ty(elementType, numElemsPer32Bits);
  for (int i = 0; i < num32BitValues; ++i) {
    Value pack = rewriter.create<LLVM::UndefOp>(loc, packTy);
    for (int j = 0; j < numElemsPer32Bits; ++j) {
      Value element = elements[startIndex + i * numElemsPer32Bits + j];
      pack = b.insert_element(packTy, pack, element, b.i32_val(j));
    }
    pack = b.bitcast(pack, rewriter.getIntegerType(32));
    mmaOut[i] = pack;
  }
  return mmaOut;
}

// If the accumulator is fp16 unpack it from 32-bit integers.
SmallVector<Value> unpackAccumulator(ConversionPatternRewriter &rewriter,
                                     Location loc,
                                     const SmallVector<Value> &packed,
                                     RankedTensorType tensorTy) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (!tensorTy.getElementType().isF16())
    return packed;
  // For fp16 the accumulator is pack into 32-bit integers so we need to unpack
  // it.
  SmallVector<Value> results;
  for (Value elem : packed) {
    elem = b.bitcast(elem, vec_ty(rewriter.getF16Type(), 2));
    results.push_back(
        b.extract_element(rewriter.getF16Type(), elem, b.i32_val(0)));
    results.push_back(
        b.extract_element(rewriter.getF16Type(), elem, b.i32_val(1)));
  }
  return results;
}

static Value faddAccumulate(ConversionPatternRewriter &rewriter, Location loc,
                            Value a, Value b) {
  int numEl = cast<LLVM::LLVMStructType>(a.getType()).getBody().size();
  Value newStruct = rewriter.create<LLVM::UndefOp>(loc, a.getType());
  for (int i = 0; i < numEl; ++i) {
    Value lhs = rewriter.create<LLVM::ExtractValueOp>(loc, a, i);
    Value rhs = rewriter.create<LLVM::ExtractValueOp>(loc, b, i);
    Value add = rewriter.create<LLVM::FAddOp>(loc, lhs, rhs);
    newStruct = rewriter.create<LLVM::InsertValueOp>(loc, newStruct, add, i);
  }
  return newStruct;
}

static SmallVector<Value> emitWait(ConversionPatternRewriter &rewriter,
                                   Location loc, SmallVector<Value> acc,
                                   int pendings) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Type> types(acc.size(), acc[0].getType());
  auto structTy =
      LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structTy);
  int i = 0;
  for (Value v : acc) {
    llvmStruct = b.insert_val(structTy, llvmStruct, v, i++);
  }
  Value res = rewriter.create<triton::nvgpu::WGMMAWaitGroupOp>(loc, llvmStruct,
                                                               pendings);
  SmallVector<Value> results;
  for (int i = 0; i < acc.size(); ++i) {
    results.push_back(b.extract_val(types[0], res, i));
  }
  return results;
}

LogicalResult convertDot(const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Operation *op, Value a, Value b, Value c, Value d,
                         Value useCOperand, Value loadedA, Value loadedB,
                         Value loadedC, bool allowTF32,
                         bool needsPartialAccumulator,
                         uint32_t maxNumImpreciseAcc, bool sync, Value thread) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  auto aTensorTy = cast<triton::gpu::TensorOrMemDesc>(a.getType());
  auto bTensorTy = cast<triton::gpu::MemDescType>(b.getType());
  auto dTensorTy = cast<RankedTensorType>(d.getType());
  auto aSharedLayout =
      dyn_cast<NVMMASharedEncodingAttr>(aTensorTy.getEncoding());
  auto bSharedLayout = cast<NVMMASharedEncodingAttr>(bTensorTy.getEncoding());
  auto mmaEncoding = cast<NvidiaMmaEncodingAttr>(dTensorTy.getEncoding());
  bool transA = false;
  Value baseA;
  Value baseB;
  if (aSharedLayout)
    baseA = getOffsetedBase(loadedA, cast<MemDescType>(aTensorTy),
                            typeConverter, rewriter, loc);
  baseB = getOffsetedBase(loadedB, bTensorTy, typeConverter, rewriter, loc);
  if (aSharedLayout) {
    transA = aSharedLayout.getTransposed();
  }
  bool transB = !bSharedLayout.getTransposed();
  auto dShapePerCTA = getShapePerCTA(dTensorTy);
  auto instrShape = mmaEncoding.getInstrShape();
  auto accSize = 2 * (instrShape[1] / 4);
  unsigned M = 4 * instrShape[0];
  unsigned N = instrShape[1];
  unsigned K = instrShape[2];
  bool zeroAcc = isZeroConst(c);
  auto instrMNK = mmaEncoding.getInstrShape();
  auto warpSize = mmaEncoding.getWarpsPerCTA();
  auto shapePerCTATile = SmallVector<unsigned>{instrMNK[0] * warpSize[0],
                                               instrMNK[1] * warpSize[1]};
  int numRepM = ceil<unsigned>(dShapePerCTA[0], shapePerCTATile[0]);
  int numRepN = ceil<unsigned>(dShapePerCTA[1], shapePerCTATile[1]);
  int numRepK = ceil<unsigned>(aTensorTy.getShape()[1], instrShape[2]);
  DotOpMmaSmemLoader aLoader;
  SmallVector<Value> structA;
  auto warpGroups = {warpSize[0] / 4, warpSize[1]};
  if (aSharedLayout) {
    aLoader =
        DotOpMmaSmemLoader::build(loc, rewriter, cast<MemDescType>(aTensorTy),
                                  baseA, {M, K}, 3, dTensorTy, 0);
  } else {
    structA = unpackLLElements(loc, loadedA, rewriter);
  }
  DotOpMmaSmemLoader bLoader = DotOpMmaSmemLoader::build(
      loc, rewriter, bTensorTy, baseB, {K, N}, 3, dTensorTy, 1);

  auto fc = unpackLLElements(loc, loadedC, rewriter);

  triton::nvgpu::WGMMAEltType eltTypeC = getMmaRetType(d);
  triton::nvgpu::WGMMAEltType eltTypeA = getMmaOperandType(a, allowTF32);
  triton::nvgpu::WGMMAEltType eltTypeB = getMmaOperandType(b, allowTF32);

  triton::nvgpu::WGMMALayout layoutA = transA ? triton::nvgpu::WGMMALayout::col
                                              : triton::nvgpu::WGMMALayout::row;
  triton::nvgpu::WGMMALayout layoutB = transB ? triton::nvgpu::WGMMALayout::row
                                              : triton::nvgpu::WGMMALayout::col;

  auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
  Operation *startSequence = rewriter.create<NVVM::WgmmaFenceAlignedOp>(loc);
  SmallVector<Value> mmaResults;
  for (int m = 0; m < numRepM; ++m) {
    for (int n = 0; n < numRepN; ++n) {
      llvm::SmallVector<Value> mmaOut =
          loadReg(rewriter, loc, fc, (m * numRepN + n) * accSize, accSize,
                  startSequence);
      llvm::SmallVector<Type> elemTypes;
      for (Value accEl : mmaOut)
        elemTypes.push_back(accEl.getType());
      auto accTy =
          LLVM::LLVMStructType::getLiteral(rewriter.getContext(), elemTypes);
      Value d;
      Value useC = tb.i1_val(0);
      if (!zeroAcc) {
        d = packLLElements(loc, typeConverter, mmaOut, rewriter, accTy);
        useC = tb.i1_val(1);
      }
      if (useCOperand)
        useC = tb.and_(useC, useCOperand);
      uint32_t numLowPrecisionAcc = 0;
      Value partialAcc;
      for (int k = 0; k < numRepK; ++k) {
        Value a;
        if (aSharedLayout) {
          a = aLoader.smemLoad(m, k, rewriter, loc);
        } else {
          auto aDotOpEnc =
              cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
          assert(aDotOpEnc.getKWidth() ==
                 32 / aTensorTy.getElementTypeBitWidth());

          unsigned regASize = (instrShape[0] * instrShape[2]) / 32;
          llvm::SmallVector<Value> regA =
              loadReg(rewriter, loc, structA, (m * numRepK + k) * regASize,
                      regASize, startSequence);
          auto regATy = LLVM::LLVMStructType::getLiteral(
              rewriter.getContext(),
              SmallVector<Type>(regA.size(), regA[0].getType()));
          a = packLLElements(loc, typeConverter, regA, rewriter, regATy);
        }
        auto b = bLoader.smemLoad(k, n, rewriter, loc);
        numLowPrecisionAcc += K;
        // If using native accumulation would cause use to do more low precion
        // accumulation than allowed do a separate allocation.
        bool requireAddAccumulator =
            needsPartialAccumulator &&
            (numLowPrecisionAcc >= maxNumImpreciseAcc || k == numRepK - 1);
        Value mmaAcc = needsPartialAccumulator ? partialAcc : d;
        mmaAcc = rewriter.create<triton::nvgpu::WGMMAOp>(
            loc, accTy, a, b, useC, mmaAcc, M, N, K, eltTypeC, eltTypeA,
            eltTypeB, layoutA, layoutB);
        useC = tb.i1_val(1);
        if (needsPartialAccumulator)
          partialAcc = mmaAcc;
        else
          d = mmaAcc;
        // If we need accumulate separately to have higher precision, insert
        // adds.
        if (requireAddAccumulator) {
          d = d ? faddAccumulate(rewriter, loc, d, partialAcc) : partialAcc;
          numLowPrecisionAcc = 0;
          partialAcc = Value();
        }
      }
      auto acc = unpackLLElements(loc, d, rewriter);
      for (int i = 0; i < acc.size(); ++i) {
        mmaResults.push_back(acc[i]);
      }
    }
  }
  rewriter.create<NVVM::WgmmaGroupSyncAlignedOp>(loc);

  if (sync)
    mmaResults = emitWait(rewriter, loc, mmaResults, 0);

  SmallVector<Value> results =
      unpackAccumulator(rewriter, loc, mmaResults, dTensorTy);

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      mmaEncoding.getContext(),
      SmallVector<Type>(results.size(), dTensorTy.getElementType()));
  auto res = packLLElements(loc, typeConverter, results, rewriter, structTy);
  rewriter.replaceOp(op, res);
  return success();
}

LogicalResult convertWGMMA(triton::nvidia_gpu::WarpGroupDotOp op,
                           triton::nvidia_gpu::WarpGroupDotOp::Adaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Value thread) {
  auto AEnc = op.getA().getType().getEncoding();
  auto BEnc = op.getB().getType().getEncoding();
  return convertDot(typeConverter, rewriter, op.getLoc(), op.getOperation(),  //
                    op.getA(), op.getB(), op.getC(), op.getD(), op.getUseC(), //
                    adaptor.getA(), adaptor.getB(), adaptor.getC(),           //
                    op.getInputPrecision() == InputPrecision::TF32,
                    op.needsPartialAccumulator(), op.getMaxNumImpreciseAcc(),
                    !op.getIsAsync(), thread);
}
