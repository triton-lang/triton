//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "xpu/lib/Conversion/TritonXPUToLLVM/PatternTritonXPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {

struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const xpu::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {
    isBf16RoundToMid = ::triton::tools::getBoolEnv("TRITONXPU_BF16_ROUND_MID");
  }

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getPtrContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    LDBG("getVectorSize contiguity = " << contiguity << " pointeeBitWidth = "
                                       << pointeeBitWidth);
    // The maximum vector size is 512 bits on XPUs.
    return std::min<unsigned>(512 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

  void getVectorInfo(Type tensorType, unsigned &vecSize,
                     unsigned &elemNbits) const {
    vecSize = 1u;
    elemNbits = 32u;
    if (auto vecType = mlir::dyn_cast<mlir::VectorType>(tensorType)) {
      unsigned numElems = vecType.getNumElements();
      Type elemTy = vecType.getElementType();
      elemNbits = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                      ? 64u
                      : elemTy.getIntOrFloatBitWidth();
      // The maximum vector size is 512 bits on XPU2.
      vecSize = std::min<unsigned>(512 / elemNbits, numElems);
    }
  }

  void setHaddr(ConversionPatternRewriter &rewriter, mlir::Location &loc,
                Value ptr) const {
    Value ptrInt = ptrtoint(i64_ty, ptr);
    Value ptrIntH32 = lshr(ptrInt, int_val(64, 32));
    Value ptrIntS32 = trunc(i32_ty, ptrIntH32);
    rewriter.create<mlir::LLVM::XPU::SetHaddrOp>(loc, ptrIntS32);
  }

  void createGM2LMOp(ConversionPatternRewriter &rewriter,
                     mlir::MLIRContext *ctx, mlir::Location &loc, Value src,
                     Value dst, Value offset, Value size) const {
    switch (static_cast<XPUArch>(targetInfo.getXPUArch())) {
    case XPUArch::XPU2: {
      setHaddr(rewriter, loc, src);
      Value srcAs0 = addrspace_cast(ptr_ty(ctx, 0), src);
      rewriter.create<mlir::LLVM::XPU::GM2LMOp>(loc, srcAs0, dst, offset, size);
      break;
    }
    case XPUArch::XPU3: {
      rewriter.create<mlir::LLVM::XPU::GM2LMOp_v3>(loc, src, dst, offset, size);
      break;
    }
    default:
      llvm_unreachable(
          "Failed to create GM2LMOp with unsupported xpu architecture.");
    }
  }

  void createLM2GMOp(ConversionPatternRewriter &rewriter,
                     mlir::MLIRContext *ctx, mlir::Location &loc, Value src,
                     Value dst, Value offset, Value size) const {
    switch (static_cast<XPUArch>(targetInfo.getXPUArch())) {
    case XPUArch::XPU2: {
      setHaddr(rewriter, loc, dst);
      Value dstAs0 = addrspace_cast(ptr_ty(ctx, 0), dst);
      rewriter.create<mlir::LLVM::XPU::LM2GMOp>(loc, src, dstAs0, offset, size);
      break;
    }
    case XPUArch::XPU3: {
      rewriter.create<mlir::LLVM::XPU::LM2GMOp_v3>(loc, src, dst, offset, size);
      break;
    }
    default:
      llvm_unreachable(
          "Failed to create LM2GMOp with unsupported xpu architecture.");
    }
  }

  void createSM2GMOp(ConversionPatternRewriter &rewriter,
                     mlir::MLIRContext *ctx, mlir::Location &loc, Value src,
                     Value dst, Value offset, Value size) const {
    switch (static_cast<XPUArch>(targetInfo.getXPUArch())) {
    case XPUArch::XPU2: {
      setHaddr(rewriter, loc, dst);
      Value dstAs0 = addrspace_cast(ptr_ty(ctx, 0), dst);
      rewriter.create<mlir::LLVM::XPU::SM2GMOp>(loc, src, dstAs0, offset, size);
      break;
    }
    case XPUArch::XPU3: {
      rewriter.create<mlir::LLVM::XPU::SM2GMOp_v3>(loc, src, dst, offset, size);
      break;
    }
    default:
      llvm_unreachable(
          "Failed to create LM2GMOp with unsupported xpu architecture.");
    }
  }

  void createMemOp(ConversionPatternRewriter &rewriter, mlir::MLIRContext *ctx,
                   mlir::Location &loc, Value bufPtr, Value gmPtr, Value offset,
                   Value size, MemCpyType memCpyType) const {
    switch (static_cast<MemCpyType>(memCpyType)) {
    case MemCpyType::GM2LM:
      createGM2LMOp(rewriter, ctx, loc, bufPtr, gmPtr, offset, size);
      break;
    case MemCpyType::LM2GM:
      createLM2GMOp(rewriter, ctx, loc, gmPtr, bufPtr, offset, size);
      break;
    case MemCpyType::SM2GM:
      createSM2GMOp(rewriter, ctx, loc, bufPtr, gmPtr, offset, size);
      break;
    default:
      llvm_unreachable("Memory Op only includes GM2LM, LM2GM, SM2GM");
    }
  }

  void createMfenceOp(ConversionPatternRewriter &rewriter,
                      mlir::Location &loc) const {
    // The magic number 5(101) of MfenceOp means mfencing on LM and GM
    rewriter.create<mlir::LLVM::XPU::MfenceOp>(loc, i32_val(5));
  }

  Value getStartPtr(ConversionPatternRewriter &rewriter, mlir::MLIRContext *ctx,
                    mlir::Location &loc, Value gmPtr, Value zeroPtr,
                    Value rowLen, Value elemBytes) const {
    Value gmPtrInt = ptrtoint(i64_ty, gmPtr);
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value offset = sdiv(sub(gmPtrInt, zeroPtrInt), elemBytes);
    Value startOffsetBytes = mul(mul(sdiv(offset, rowLen), rowLen), elemBytes);
    Value startPtr = gep(ptr_ty(ctx, 0), i8_ty, zeroPtr,
                         startOffsetBytes); // convert ptr first, then move
    return startPtr;
  }

  void lowerLocallyContinuousUnfixedStride(
      Operation *op, Location loc, ConversionPatternRewriter &rewriter,
      int64_t _rowLen, int64_t _bufLen, int64_t _elemBytes, Value llGMPtr,
      Value llLMPtr, Value llLen, Value offsetBytes, MemCpyType memCpyType,
      Block *oldBlock, Block *newBlock) const {
    // clang-format off
    /* *****************************************************************************
    def getStartPtr(gmPtr, zeroPtr, rowLen, elemBytes):
        offset = (gmPtr - zeroPtr) / elemBytes
        startOffsetBytes = (offset / rowLen) * rowLen * elemBytes
        return zeroPtr + startOffsetBytes

    _rowMaxTail = _bufLen % _rowLen
    _rowNum = _bufLen / _rowLen
    rowBytes = rowLen * elemBytes
    tailLen = min(rowLen - (gmPtr.front() - zeroPtr) / elemBytes % rowLen, bufLen)
    if _rowMaxTail == 0:
      for i in range(_rowNum):
        gmStartPtr = llGMPtrs[i * _rowLen]
        lmOffsetBytes = (i * _rowLen) * elemBytes
        lmStartPtr = lmPtr + lmOffsetBytes;
        gm2lm(gmStartPtr, lmStartPtr, remainBytes)
    else:
      if 0 < tailLen < rowMaxTail:
        gm2lm(gmPtr.front(), lmPtr, tailBytes)
        for i in range(_rowNum):
          gmStartPtr = getStartPtr(gmPtr[_rowMaxTail+i*_rowLen], zeroPtr, rowLen, elemBytes)
          lmOffsetBytes = (tailLen + i * rowLen) * elemBytes
          lmStartPtr = lmPtr + lmOffsetBytes
          gm2lm(gmStartPtr, lmStartPtr, rowBytes)
        gmStartPtr = getStartPtr(gmPtr.back(), zeroPtr, rowLen, elemBytes)
        offset = tailLen + rowNum * rowLen
        lmOffsetBytes = offset * elemBytes
        lmStartPtr = lmPtr + lmOffsetBytes
        remainBytes = (bufLen - offset) * elemBytes
        gm2lm(gmStartPtr, lmStartPtr, remainBytes)
      else:
          gm2lm(gmPtr.front(), lmPtr, tailBytes)
          if _rowNum >= 1:
            for i in range(_rowNum-1):
              gmPtr1 = gmPtr[_rowMaxTail+i*_rowLen]
              gmPtr2 = gmPtr[_rowMaxTail+(i+1)*_rowLen]
              gmPtr = select(tailLen == rowMaxTail, gmPtr1, gmPtr2)
              gmStartPtr = getStartPtr(gmPtr[_rowMaxTail+(i+1)*_rowLen], zeroPtr, rowLen, elemBytes)
              lmOffsetBytes = (tailLen + i * rowLen) * elemBytes
              lmStartPtr = lmPtr + lmOffsetBytes
              gm2lm(gmStartPtr, lmStartPtr, rowBytes)
            gmStartPtr = getStartPtr(gmPtr.back(), zeroPtr, rowLen, elemBytes)
            offset = tailLen + (rowNum - 1) * rowLen
            lmOffsetBytes = offset * elemBytes
            lmStartPtr = lmPtr + lmOffsetBytes
            remainBytes = (bufLen - offset) * elemBytes
            gm2lm(gmStartPtr, lmStartPtr, remainBytes)
    ********************************************************************************/
    // clang-format on
    MLIRContext *ctx = rewriter.getContext();

    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    Value gmFrontPtr = llGMPtrs.front();
    Value gmBackPtr = llGMPtrs.back();
    Value lmPtr = llLMPtrs.front();

    auto zeroOp = findDefOpBwd<LLVM::GEPOp>(gmFrontPtr);
    Value zeroPtr = cast<LLVM::GEPOp>(zeroOp).getBase();
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value gmFrontPtrInt = ptrtoint(i64_ty, gmFrontPtr);

    int64_t _rowMaxTail = _bufLen % _rowLen;
    int64_t _rowNum = _bufLen / _rowLen;
    Value rowMaxTail = i64_val(_rowMaxTail);
    Value rowNum = i64_val(_rowNum);
    Value rowLen = i64_val(_rowLen);
    Value bufLen = i64_val(_bufLen);
    Value elemBytes = i64_val(_elemBytes);
    Value rowBytes = trunc(i32_ty, mul(rowLen, elemBytes));

    if (_rowMaxTail == 0) {
      // GM2LM/LM2GM Row Data
      for (int64_t i = 0; i < _rowNum; ++i) {
        Value gmStartPtr = llGMPtrs[i * _rowLen];
        Value lmOffsetBytes = mul(i64_val(i * _rowLen), elemBytes);
        Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
        createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                    rowBytes, memCpyType);
      }
    } else {
      Value gmFrontOffset = sdiv(sub(gmFrontPtrInt, zeroPtrInt), elemBytes);
      Value tailLen = smin(sub(rowLen, srem(gmFrontOffset, rowLen)), bufLen);

      Block *thenBB = rewriter.createBlock(newBlock);
      Block *elseBB = rewriter.createBlock(newBlock);
      Block *mfenceBB = rewriter.createBlock(newBlock);
      rewriter.setInsertionPointToEnd(oldBlock);

      Value condTailSgt = icmp_sgt(tailLen, i64_val(0));
      Value condTailSlt = icmp_slt(tailLen, rowMaxTail);
      Value condTailDiff = and_(condTailSgt, condTailSlt);
      rewriter.create<LLVM::CondBrOp>(loc, condTailDiff, thenBB, elseBB);
      // 1. ThenBB
      rewriter.setInsertionPointToEnd(thenBB);
      {
        // 1.1 GM2LM/LM2GM Tail Data
        Value tailBytes = trunc(i32_ty, mul(tailLen, elemBytes));
        createMemOp(rewriter, ctx, loc, gmFrontPtr, lmPtr, offsetBytes,
                    tailBytes, memCpyType);
        // 1.2 GM2LM/LM2GM Row Data
        for (int64_t i = 0; i < _rowNum; ++i) {
          Value gmPtr = llGMPtrs[_rowMaxTail + i * _rowLen];
          Value gmStartPtr = getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr,
                                         rowLen, elemBytes);
          Value lmOffsetBytes =
              mul(add(tailLen, i64_val(i * _rowLen)), elemBytes);
          Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
          createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                      rowBytes, memCpyType);
        }
        // 1.3 GM2LM/LM2GM Remain Data
        Value gmPtr = llGMPtrs.back();
        Value gmStartPtr =
            getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr, rowLen, elemBytes);
        Value offset = add(tailLen, i64_val(_rowNum * _rowLen));
        Value lmOffsetBytes = mul(offset, elemBytes);
        Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
        Value remainBytes = trunc(i32_ty, mul(sub(bufLen, offset), elemBytes));
        createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                    remainBytes, memCpyType);
      }
      rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                  mfenceBB); // Jump to mfenceBB

      // 2. elseBB
      rewriter.setInsertionPointToEnd(elseBB);
      {
        // 1.1 GM2LM/LM2GM Tail Data
        Value tailBytes = trunc(i32_ty, mul(tailLen, elemBytes));
        createMemOp(rewriter, ctx, loc, gmFrontPtr, lmPtr, offsetBytes,
                    tailBytes, memCpyType);
        if (_rowNum >= 1) {
          // 1.2 GM2LM/LM2GM Row Data
          Value gmCond = icmp_eq(tailLen, rowMaxTail);
          for (int64_t i = 0; i < _rowNum - 1; ++i) {
            Value gmPtr1 = llGMPtrs[_rowMaxTail + i * _rowLen];
            Value gmPtr2 = llGMPtrs[_rowMaxTail + (i + 1) * _rowLen];
            Value gmPtr = select(gmCond, gmPtr1, gmPtr2);
            Value gmStartPtr = getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr,
                                           rowLen, elemBytes);
            Value lmOffsetBytes =
                mul(add(tailLen, i64_val(i * _rowLen)), elemBytes);
            Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
            createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                        rowBytes, memCpyType);
          }
          // 1.3 GM2LM/LM2GM Remain Data
          Value gmPtr = llGMPtrs.back();
          Value gmStartPtr = getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr,
                                         rowLen, elemBytes);
          Value offset = add(tailLen, i64_val((_rowNum - 1) * _rowLen));
          Value lmOffsetBytes = mul(offset, elemBytes);
          Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
          Value remainBytes =
              trunc(i32_ty, mul(sub(bufLen, offset), elemBytes));
          createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                      remainBytes, memCpyType);
        }
      }
      rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                  mfenceBB); // Jump to mfenceBB

      // 3. mefenceBB
      rewriter.setInsertionPointToEnd(mfenceBB);
    }
  }

  void lowerLocallyContinuousLargeRow(Operation *op, Location loc,
                                      ConversionPatternRewriter &rewriter,
                                      size_t rowSize, size_t rowStride,
                                      Value llGMPtr, Value llLMPtr, Value llLen,
                                      Value bufLen, Value elemBytes,
                                      Value offsetBytes, MemCpyType memCpyType,
                                      Block *oldBlock, Block *newBlock) const {

    /* *************************************************
    gapLen = strideLen - rowLen
    bankOffset = (bankPtrInt - zeroPtrInt) / elemBytes
    rowOffset = bankOffset / strideLen * strideLen
    blockOffset = ((bankOffset - rowOffset) / rowLen) * rowLen
    realTailLen = rowLen - (bankOffset - (blockOffset + rowOffset))

    if 0 < realTailLen < bufLen:
      gm2lm(bankPtr, lmPtr, realTailLen * elemBytes)
      gm2lm(bankPtr + (realTailLen + gapLen) * elemBytes, lmPtr + realTailLen,
    elemBytes,（bufLen - realTailLen）* elemBytes)

    else :
      gm2lm(bankPtr, lmPtr, bufLen * elemBytes)
    * ************************************************/

    MLIRContext *ctx = rewriter.getContext();

    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    auto bankPtr = llGMPtrs[0];
    auto lmBuf = llLMPtrs[0];
    if (bufLen.getType().isInteger(64)) {
      bufLen = trunc(i32_ty, bufLen);
    }

    auto zeroOp = findDefOpBwd<LLVM::GEPOp>(bankPtr);
    auto zeroPtr = cast<LLVM::GEPOp>(zeroOp).getBase();
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value bankPtrInt = ptrtoint(i64_ty, bankPtr);

    size_t gapSize = rowStride - rowSize;
    Value rowLen = i32_val(rowSize);
    Value strideLen = i32_val(rowStride);
    Value gapLen = i32_val(gapSize);
    Value gapBytes = mul(gapLen, elemBytes);
    Value bankOffset =
        sdiv(trunc(i32_ty, sub(bankPtrInt, zeroPtrInt)), elemBytes);
    Value rowOffset = rowStride == 0
                          ? i32_val(0)
                          : mul(sdiv(bankOffset, strideLen), strideLen);
    Value blockOffset =
        rowStride == 0 ? i32_val(0)
                       : mul(sdiv(sub(bankOffset, rowOffset), rowLen), rowLen);
    Value realTailLen =
        sub(rowLen, sub(bankOffset, add(blockOffset, rowOffset)));
    Value realTailBytes = mul(realTailLen, elemBytes);

    zeroPtr = gep(ptr_ty(ctx, 1), i8_ty, zeroPtr, i32_val(0));
    bankPtr = gep(ptr_ty(ctx, 1), i8_ty, bankPtr, i32_val(0));
    Value lmPtr = gep(ptr_ty(ctx, 0), i8_ty, lmBuf, i32_val(0));

    Block *thenBB = rewriter.createBlock(newBlock);
    Block *elseBB = rewriter.createBlock(newBlock);
    Block *mfenceBB = rewriter.createBlock(newBlock);
    rewriter.setInsertionPointToEnd(oldBlock);

    Value condRemSgt = icmp_sgt(realTailLen, i32_val(0));
    Value condRemSlt = icmp_slt(realTailLen, bufLen);
    Value condRemDiff = and_(condRemSgt, condRemSlt);
    rewriter.create<LLVM::CondBrOp>(loc, condRemDiff, thenBB, elseBB);
    rewriter.setInsertionPointToEnd(thenBB);
    // 1. ThenBB
    // 1.1 GM2LM Tail Data
    Value tailLen = realTailLen;
    if (llLen) {
      auto llLens = unpackLLElements(loc, llLen, rewriter);
      if (llLens[0].getType().isInteger(64)) {
        Value limitedLen =
            smin(smax(llLens[0], i64_val(0)), sext(i64_ty, bufLen));
        tailLen = smin(realTailLen, trunc(i32_ty, limitedLen));
      } else if (llLens[0].getType().isInteger(1)) {
        Value limitedLen = bufLen;
        tailLen = smin(realTailLen, limitedLen);
      } else {
        Value limitedLen = smin(smax(llLens[0], i32_val(0)), bufLen);
        tailLen = smin(realTailLen, limitedLen);
      }
    }
    Value tailBytes = mul(tailLen, elemBytes);
    createMemOp(rewriter, ctx, loc, bankPtr, lmPtr, offsetBytes, tailBytes,
                memCpyType);

    // 1.2 GM2LM Remain Data
    Value startCond;
    if (llLen) {
      auto llLens = unpackLLElements(loc, llLen, rewriter);
      if (llLens[0].getType().isInteger(64)) {
        startCond = icmp_sge(sext(i64_ty, realTailLen), llLens[0]);
      } else if (llLens[0].getType().isInteger(1)) {
        startCond = icmp_sge(realTailLen, bufLen);
      } else {
        startCond = icmp_sge(realTailLen, llLens[0]);
      }
    }
    Value startPtrInt =
        add(bankPtrInt, zext(i64_ty, add(realTailBytes, gapBytes)));
    Value startPtr =
        rowStride == 0 ? zeroPtr : inttoptr(ptr_ty(ctx, 1), startPtrInt);
    startPtr = startCond ? select(startCond, zeroPtr, startPtr) : startPtr;
    Value dstStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr,
                            realTailBytes); // convert ptr first, then move

    Value remainLen = sub(bufLen, realTailLen);
    Value remainBytes = mul(remainLen, elemBytes);
    createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                remainBytes, memCpyType);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                mfenceBB); // Jump to mfenceBB

    // 2. elseBB
    rewriter.setInsertionPointToEnd(elseBB);
    // GM2LM the whole bufLen
    Value readBytes = mul(bufLen, elemBytes);
    createMemOp(rewriter, ctx, loc, bankPtr, lmPtr, offsetBytes, readBytes,
                memCpyType);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                mfenceBB); // Jump to mfenceBB

    // 3. mefenceBB
    rewriter.setInsertionPointToEnd(mfenceBB);
  }

  void lowerLocallyContinuousSmallRow(Operation *op, Location loc,
                                      ConversionPatternRewriter &rewriter,
                                      size_t rowSize, size_t rowStride,
                                      Value llGMPtr, Value llLMPtr, Value llLen,
                                      Value bufLen, Value elemBytes,
                                      Value offsetBytes, MemCpyType memCpyType,
                                      Block *oldBlock, Block *newBlock) const {

    /* *************************************************
    bankOffset = (bankPtrInt - zeroPtrInt) / elemBytes
    rowOffset = bankOffset / strideLen * strideLen
    blockOffset = ((bankOffset - rowOffset) / rowLen)
    rowHeadLen = bankOffset - (blockOffset + rowOffset)
    realTailLen = rowLen - rowHeadLen
    rowNum = (bufLen - realTailLen - 1) / rowLen

    gm2lm(bankPtr, lmPtr, realTailLen * elemBytes)

    for(i = 0; i < rowNum; i++) {
        gm2lm(bankPtr + ((i + 1) * strideLen - rowHeadLen) * elemBytes, lmPtr +
    (realTailLen + i * rowLen) * elemBytes, rowLen * elemBytes)
    }

    remLen = bufLen - realTailLen - rowNum * rowLen
    gm2lm(bankPtr + ((rowNum + 1) * strideLen - rowHeadLen) * elemBytes, lmPtr +
    (realTailLen + rowNum * rowLen) * elemBytes, (remLen * elemBytes)
    *************************************************/

    MLIRContext *ctx = rewriter.getContext();

    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    auto bankPtr = llGMPtrs[0];
    auto lmBuf = llLMPtrs[0];
    if (bufLen.getType().isInteger(64)) {
      bufLen = trunc(i32_ty, bufLen);
    }
    auto zeroOp = findDefOpBwd<LLVM::GEPOp>(bankPtr);
    auto zeroPtr = cast<LLVM::GEPOp>(zeroOp).getBase();
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value bankPtrInt = ptrtoint(i64_ty, bankPtr);

    Value rowLen = i32_val(rowSize);
    Value strideLen = i32_val(rowStride);
    Value bankOffset =
        sdiv(trunc(i32_ty, sub(bankPtrInt, zeroPtrInt)), elemBytes);
    Value rowOffset = rowStride == 0
                          ? i32_val(0)
                          : mul(sdiv(bankOffset, strideLen), strideLen);
    Value blockOffset =
        rowStride == 0 ? i32_val(0)
                       : mul(sdiv(sub(bankOffset, rowOffset), rowLen), rowLen);
    Value realTailLen =
        sub(rowLen, sub(bankOffset, add(blockOffset, rowOffset)));
    Value realTailBytes = mul(realTailLen, elemBytes);
    Value rowBytes = mul(rowLen, elemBytes);
    Value rowHeadLen = sub(rowLen, realTailLen);
    Value rowHeadBytes = sub(rowBytes, realTailBytes);
    Value realRemainLen = sub(sub(bufLen, realTailLen), i32_val(1));
    Value rowNum = sdiv(realRemainLen, rowLen);

    zeroPtr = gep(ptr_ty(ctx, 1), i8_ty, zeroPtr, i32_val(0));
    bankPtr = gep(ptr_ty(ctx, 1), i8_ty, bankPtr, i32_val(0));
    Value lmPtr = gep(ptr_ty(ctx, 0), i8_ty, lmBuf, i32_val(0));

    Block *judgeBB = rewriter.createBlock(newBlock, TypeRange{i32_ty}, {loc});
    Block *gm2lmRowBB = rewriter.createBlock(newBlock);
    Block *stepBB = rewriter.createBlock(newBlock);
    Block *gm2lmRemBB = rewriter.createBlock(newBlock);

    // 1.  GM2LM Tail Data
    rewriter.setInsertionPointToEnd(oldBlock);
    Value tailLen = realTailLen;
    if (llLen) {
      auto llLens = unpackLLElements(loc, llLen, rewriter);
      if (llLens[0].getType().isInteger(64)) {
        tailLen = smin(realTailLen, trunc(i32_ty, llLens[0]));
      } else {
        tailLen = smin(realTailLen, llLens[0]);
      }
    }
    Value tailBytes = mul(tailLen, elemBytes);
    createMemOp(rewriter, ctx, loc, bankPtr, lmPtr, offsetBytes, tailBytes,
                memCpyType);

    Value _init = i32_val(0);
    Value _step = i32_val(1);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{_init},
                                judgeBB); // Jump to judgeBB
    Value iter = judgeBB->getArgument(0);

    // 2. GM2LM Row Data
    rewriter.setInsertionPointToEnd(judgeBB);
    Value condSlt = icmp_slt(iter, rowNum);
    rewriter.create<LLVM::CondBrOp>(loc, condSlt, gm2lmRowBB, gm2lmRemBB);

    rewriter.setInsertionPointToEnd(gm2lmRowBB);
    Value skipStride = mul(add(iter, i32_val(1)), strideLen);
    Value skipStrideBytes = mul(skipStride, elemBytes);
    Value skipRowLen = mul(iter, rowLen);
    Value startPtrInt =
        add(bankPtrInt, zext(i64_ty, sub(skipStrideBytes, rowHeadBytes)));
    Value startPtr =
        rowStride == 0 ? zeroPtr : inttoptr(ptr_ty(ctx, 1), startPtrInt);
    startPtr = gep(ptr_ty(ctx, 1), i8_ty, startPtr, i32_val(0));
    Value startCond;
    if (llLen) {
      auto llLens = unpackLLElements(loc, llLen, rewriter);
      if (llLens[0].getType().isInteger(64)) {
        startCond =
            icmp_sgt(sext(i64_ty, add(skipRowLen, realTailLen)), llLens[0]);
      } else {
        startCond = icmp_sgt(add(skipRowLen, realTailLen), llLens[0]);
      }
    }
    startPtr = startCond ? select(startCond, zeroPtr, startPtr) : startPtr;
    Value dstOffset = add(realTailLen, skipRowLen);
    Value dstOffsetBytes = mul(dstOffset, elemBytes);
    Value dstStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr,
                            dstOffsetBytes); // convert ptr first, then move
    createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                rowBytes, memCpyType);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, stepBB); // Jump to stepBB

    rewriter.setInsertionPointToEnd(stepBB);
    Value _index = add(iter, _step);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{_index},
                                judgeBB); // Jump back to judgeBB

    // 3 GM2LM Remain Data
    rewriter.setInsertionPointToEnd(gm2lmRemBB);
    {
      Value skipStride = mul(add(rowNum, i32_val(1)), strideLen);
      Value skipStrideBytes = mul(skipStride, elemBytes);
      Value skipRowLen = mul(rowNum, rowLen);
      Value remainBytes =
          mul(sub(bufLen, add(realTailLen, skipRowLen)), elemBytes);
      Value startPtrInt =
          add(bankPtrInt, zext(i64_ty, sub(skipStrideBytes, rowHeadBytes)));
      Value startPtr =
          rowStride == 0 ? zeroPtr : inttoptr(ptr_ty(ctx, 1), startPtrInt);
      startPtr = gep(ptr_ty(ctx, 1), i8_ty, startPtr, i32_val(0));
      Value startCond;
      if (llLen) {
        auto llLens = unpackLLElements(loc, llLen, rewriter);
        if (llLens[0].getType().isInteger(64)) {
          startCond =
              icmp_sgt(sext(i64_ty, add(skipRowLen, realTailLen)), llLens[0]);
        } else {
          startCond = icmp_sgt(add(skipRowLen, realTailLen), llLens[0]);
        }
      }
      startPtr = startCond ? select(startCond, zeroPtr, startPtr) : startPtr;
      Value dstOffset = add(realTailLen, skipRowLen);
      Value dstOffsetBytes = mul(dstOffset, elemBytes);
      Value dstStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr,
                              dstOffsetBytes); // convert ptr first, then move
      createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                  remainBytes, memCpyType);
    }
  }

protected:
  const xpu::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  bool isBf16RoundToMid = false;
};

struct XPULoadOpConversion : public ConvertOpToLLVMPattern<triton::xpu::LoadOp>,
                             public LoadStoreConversionBase {
  XPULoadOpConversion(LLVMTypeConverter &converter,
                      const xpu::TargetInfo &targetInfo,
                      ModuleAxisInfoAnalysis &axisAnalysisPass,
                      PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  void VecBF16ToFP32Unordered(triton::xpu::LoadOp op, mlir::MLIRContext *ctx,
                              Location &loc,
                              ConversionPatternRewriter &rewriter,
                              Type &resElemTy, int numElems, int resVecSize,
                              int ptrDataVecSize, Value &lmBasePtr,
                              SmallVector<Value> &loadedVals) const {
    VectorType vecBf16Ty = VectorType::get(ptrDataVecSize, bf16_ty);
    VectorType veci16Ty = VectorType::get(ptrDataVecSize, i16_ty);
    VectorType veci32Ty = VectorType::get(resVecSize, i32_ty);
    VectorType vec1Ty = VectorType::get(ptrDataVecSize, i1_ty);
    VectorType halfVecBf16Ty = VectorType::get(resVecSize, bf16_ty);
    VectorType VecFp16Ty = VectorType::get(ptrDataVecSize, f16_ty);
    lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
    int mask = 0xaaaaaaaa;
    Value maskVal = i32_val(mask);
    maskVal = bitcast(maskVal, vec1Ty);
    Value maskNegVal = i32_val(~mask);
    maskNegVal = bitcast(maskNegVal, vec1Ty);
    int16_t pad = 0x8000;
    Value padVec = rewriter.create<LLVM::UndefOp>(loc, veci16Ty);
    for (size_t elemIdx = 0; elemIdx < ptrDataVecSize; ++elemIdx) {
      padVec = insert_element(veci16Ty, padVec, i16_val(pad), i16_val(elemIdx));
    }
    for (int i = 0; i < numElems / 2; ++i) {
      Value elemPtr = gep(ptr_ty(ctx, 0), vecBf16Ty, lmBasePtr, i32_val(i));
      Value veven;
      if (isBf16RoundToMid) {
        veven = rewriter.create<mlir::LLVM::XPU::VLOAD_MHOp>(
            loc, veci16Ty, elemPtr, padVec, maskVal);
      } else {
        veven = rewriter.create<mlir::LLVM::XPU::VLOAD_MZOp>(loc, veci16Ty,
                                                             elemPtr, maskVal);
      }
      veven = bitcast(veven, resElemTy);
      loadedVals.emplace_back(veven);
      Value vodd;
      if (isBf16RoundToMid) {
        vodd = rewriter.create<mlir::LLVM::XPU::VLOAD_MHOp>(
            loc, veci16Ty, elemPtr, padVec, maskNegVal);
      } else {
        vodd = rewriter.create<mlir::LLVM::XPU::VLOAD_MZOp>(
            loc, veci16Ty, elemPtr, maskNegVal);
      }
      vodd = bitcast(vodd, VecFp16Ty);
      Value voddSl =
          rewriter.create<mlir::LLVM::XPU::VSHUFFLE2Op>(loc, VecFp16Ty, vodd);
      voddSl = bitcast(voddSl, resElemTy);
      loadedVals.emplace_back(voddSl);
    }
    if (numElems % 2 == 1) {
      int remainedIdx = numElems - 1;
      lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      Value elemPtr =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i32_val(remainedIdx));
      Value loaded = load(halfVecBf16Ty, elemPtr);
      loaded = rewriter.create<LLVM::FPExtOp>(loc, resElemTy, loaded);
      loadedVals.emplace_back(loaded);
    }
    return;
  }

  void VecBF16ToFP32(triton::xpu::LoadOp op, mlir::MLIRContext *ctx,
                     Location &loc, ConversionPatternRewriter &rewriter,
                     Type &resElemTy, int numElems, int resVecSize,
                     int ptrDataVecSize, SmallVector<Value> &loadedVals) const {
    VectorType vecFp16Ty = VectorType::get(ptrDataVecSize, f16_ty);
    Value padVec = rewriter.create<LLVM::UndefOp>(loc, vecFp16Ty);
    int16_t pad = isBf16RoundToMid ? 0x8000 : 0;
    for (size_t elemIdx = 0; elemIdx < ptrDataVecSize; ++elemIdx) {
      padVec =
          insert_element(vecFp16Ty, padVec, f16_val(pad), i16_val(elemIdx));
    }
    SmallVector<Value> newLoadedVals;
    for (int i = 0; i < numElems / 2; ++i) {
      Value val = bitcast(loadedVals[i], vecFp16Ty);
      Value vl = rewriter.create<mlir::LLVM::XPU::VMERGE_L_HFOp>(loc, vecFp16Ty,
                                                                 padVec, val);
      vl = bitcast(vl, resElemTy);
      newLoadedVals.emplace_back(vl);
      Value vh = rewriter.create<mlir::LLVM::XPU::VMERGE_H_HFOp>(loc, vecFp16Ty,
                                                                 padVec, val);
      vh = bitcast(vh, resElemTy);
      newLoadedVals.emplace_back(vh);
    }
    if (numElems % 2 == 1) {
      int remainedIdx = numElems - 1;
      Value ext = rewriter.create<LLVM::FPExtOp>(loc, resElemTy,
                                                 loadedVals[remainedIdx]);
      newLoadedVals.emplace_back(ext);
    }
    loadedVals = newLoadedVals;
    return;
  }

  LogicalResult
  matchAndRewrite(triton::xpu::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // original values
    Value res = op.getResult();
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();
    Value index = op.getIndex();

    int32_t stride = op.getStride();
    int32_t tensor_col_size = op.getTensorColSize();
    bool coreDealMultiRows = tensor_col_size != -1;
    bool isDiscreteSame = (stride == 0);
    bool isUnknown = stride != 0 && stride != 1 && !op.getIsDiscrete();
    bool bf16Tofp32Unordered = op.getBf16Tofp32Unordered();

    LDBG("Lower LoadOp for " << ptr);

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();
    Value llIndex = adaptor.getIndex();

    // Determine Type
    Type ptrTy = ptr.getType();
    Type resTy = res.getType();

    Type ptrElemTy = typeConverter->convertType(getElementTypeOrSelf(ptrTy));
    Type resElemTy = typeConverter->convertType(getElementTypeOrSelf(resTy));
    Type ptrDataVecTy = resElemTy;

    unsigned ptrNumElems = getTotalElemsPerThread(ptrTy);
    unsigned resNumElems = getTotalElemsPerThread(resTy);

    Type ptrElemScalarTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      ptrElemScalarTy =
          mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
              .getPointeeType();
    } else {
      // Scalar
      ptrElemScalarTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }

    Type resElemScalarTy = getElementTypeOrSelf(resElemTy);

    // Get the LLVM values
    auto llPtrs = unpackLLElements(loc, llPtr, rewriter);
    stride =
        (stride >= 0 && ptrNumElems * stride <= targetInfo.getXPUBufferSize())
            ? stride
            : 1;

    assert(llPtrs.size() == ptrNumElems);
    bool isVectorized = false;
    unsigned vecSize = 1u;
    unsigned elemNbits =
        isa<triton::PointerType, LLVM::LLVMPointerType>(resElemScalarTy)
            ? 64u
            : resElemScalarTy.getIntOrFloatBitWidth();
    if (mlir::isa<mlir::VectorType>(resElemTy)) {
      isVectorized = true;
      getVectorInfo(resElemTy, vecSize, elemNbits);
    }

    // fp16Tofp32
    if (resElemScalarTy.isF32() && ptrElemScalarTy.isF16()) {
      Value fp16LM = bitcast(llPtrs[0], ptr_ty(ctx, 0));
      Value fp32LM = bitcast(llPtrs[0], ptr_ty(ctx, 0));
      ValueRange singleOperandRange(
          {fp16LM, fp32LM, i32_val(ptrNumElems * stride)});
      mlir::LLVM::XPU::createDeviceCall("_ZN3xpu10fp16tofp32EPKNS_7float16EPfi",
                                        rewriter, op, singleOperandRange, loc);
      createMfenceOp(rewriter, loc);
      ptrElemScalarTy = resElemScalarTy;
    }
    // bf16Tofp32
    bool bf16Tofp32 = false;
    if (resElemScalarTy.isF32() && ptrElemScalarTy.isBF16()) {
      int ptrVecSize = std::min(ptrNumElems, vecSize * 2);
      ptrDataVecTy = isVectorized ? VectorType::get(ptrVecSize, ptrElemScalarTy)
                                  : ptrElemScalarTy;
      bf16Tofp32 = true;
    }

    unsigned ptrDataVecSize = 1u;
    unsigned ptrDataNbits =
        isa<triton::PointerType, LLVM::LLVMPointerType>(ptrElemScalarTy)
            ? 64u
            : ptrElemScalarTy.getIntOrFloatBitWidth();
    if (mlir::isa<mlir::VectorType>(ptrDataVecTy)) {
      getVectorInfo(ptrDataVecTy, ptrDataVecSize, ptrDataNbits);
    }

    SmallVector<Value> loadedVals;
    Value lmBasePtr = bitcast(llPtrs[0], ptr_ty(ctx, 0));
    if (index) {
      ptrNumElems = resNumElems;
      unsigned _stride = (bf16Tofp32 && isVectorized) ? ptrNumElems * stride / 2
                                                      : ptrNumElems * stride;
      Value idx = mul(llIndex, i32_val(_stride));
      lmBasePtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, lmBasePtr, idx);
    }

    if (op.getSVOpt()) {
      Value elemPtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      Value loaded = load(ptrElemScalarTy, elemPtr);
      if (bf16Tofp32) {
        loaded = rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
      }
      loadedVals.push_back(loaded);
    } else if (isDiscreteSame) {
      Value elemPtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      Value loaded = load(ptrElemScalarTy, elemPtr);
      if (bf16Tofp32) {
        loaded = rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
      }
      for (size_t elemIdx = 0; elemIdx < resNumElems; elemIdx++) {
        if (isVectorized) {
          Value newVector = rewriter.create<LLVM::UndefOp>(loc, resElemTy);
          for (size_t idx = 0; idx < vecSize; ++idx) {
            newVector =
                insert_element(resElemTy, newVector, loaded, i32_val(idx));
          }
          loadedVals.push_back(newVector);
        } else {
          loadedVals.push_back(loaded);
        }
      }
    } else if (op.getIsDiscrete()) {
      if (isVectorized) {
        for (size_t vecIdx = 0; vecIdx < resNumElems; ++vecIdx) {
          Value newVector = rewriter.create<LLVM::UndefOp>(loc, resElemTy);
          for (size_t elemIdx = 0; elemIdx < vecSize; ++elemIdx) {
            auto idx = vecIdx * vecSize + elemIdx;
            Value elemPtr = bitcast(llPtrs[idx], ptr_ty(ctx, 0));
            Value loaded = load(ptrElemScalarTy, elemPtr);
            if (bf16Tofp32) {
              loaded =
                  rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
            }
            // insert val to newVector
            newVector =
                insert_element(resElemTy, newVector, loaded, i32_val(elemIdx));
          }
          loadedVals.push_back(newVector);
        }
      } else {
        for (size_t elemIdx = 0; elemIdx < resNumElems; elemIdx++) {
          Value elemPtr = bitcast(llPtrs[elemIdx], ptr_ty(ctx, 0));
          Value loaded = load(ptrElemScalarTy, elemPtr);
          if (bf16Tofp32) {
            loaded =
                rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
          }
          loadedVals.push_back(loaded);
        }
      }
    } else if (stride > 1 && isVectorized) {
      // Vgather
      VectorType offsetTy =
          VectorType::get(ptrDataVecSize, int_ty(ptrDataNbits));
      Value offsetVec = rewriter.create<LLVM::UndefOp>(loc, offsetTy);
      for (size_t elemIdx = 0; elemIdx < ptrDataVecSize; ++elemIdx) {
        Value offsetVal =
            int_val(ptrDataNbits, (ptrDataNbits / 8u) * stride * elemIdx);
        offsetVec = insert_element(offsetTy, offsetVec, offsetVal,
                                   int_val(ptrDataNbits, elemIdx));
      }
      lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      for (size_t vecIdx = 0;
           vecIdx < (index ? ptrNumElems : (ptrNumElems / ptrDataVecSize));
           ++vecIdx) {
        Value vecPtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, lmBasePtr,
                           int_val(ptrDataNbits, vecIdx * stride));
        Value tmpPtr = bitcast(vecPtr, ptr_ty(ctx, 0));
        Value vgather;
        if (ptrElemScalarTy.isF32()) {
          vgather = rewriter.create<mlir::LLVM::XPU::VGatherFOp>(
              loc, offsetTy, tmpPtr, offsetVec);
        } else if (ptrElemScalarTy.isF16() || ptrElemScalarTy.isBF16()) {
          vgather = rewriter.create<mlir::LLVM::XPU::VGatherHFOp>(
              loc, offsetTy, tmpPtr, offsetVec);
        } else {
          llvm_unreachable("Only support FP16/BF16/FP32 in VGather!");
        }
        Value loaded = bitcast(vgather, resElemTy);
        loadedVals.push_back(loaded);
      }
      if (bf16Tofp32) {
        VecBF16ToFP32(op, ctx, loc, rewriter, resElemTy, resNumElems, vecSize,
                      ptrDataVecSize, loadedVals);
      }
    } else { // Continuous || Unknown(No VGather)
      if (coreDealMultiRows && !isUnknown) {
        // Unknown Manipulation in GM2LMOp Conversion
        /*  Small Col Size Opt GM2LM (14 legal data)

            Before Opt:
                1 1 1 1 1 1 1 1
                1 1 1 1 1 1 0 0

            After Opt:
                1 1 1 1 1 1 1 0
                1 1 1 1 1 1 1 0
        */
        Value bufPtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
        auto mem_col_size =
            mlir::cast<RankedTensorType>(resTy).getShape()[1] * vecSize;
        auto tensor_row_size =
            mlir::cast<RankedTensorType>(resTy).getShape()[0] / 64;
        unsigned rowRemainElem = mem_col_size - tensor_col_size;

        for (size_t row_idx = 0; row_idx < tensor_row_size; ++row_idx) {
          for (size_t col_idx = 0; col_idx < tensor_col_size; ++col_idx) {
            auto buf_global_idx = row_idx * tensor_col_size + col_idx;
            Value elemPtr = gep(ptr_ty(ctx, 0), ptrElemScalarTy, bufPtr,
                                i32_val(buf_global_idx));
            Value loaded = load(ptrElemScalarTy, elemPtr);
            if (bf16Tofp32) {
              loaded =
                  rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
            }
            loadedVals.push_back(loaded);
          }

          for (size_t remainElem = rowRemainElem; remainElem > 0;
               --remainElem) {
            Value loaded = int_val(ptrDataNbits, 0);
            loaded = bitcast(loaded, ptrElemScalarTy);
            if (bf16Tofp32) {
              loaded =
                  rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
            }
            loadedVals.push_back(loaded);
          }
        }

        if (isVectorized) {
          SmallVector<Value> loadedVecVals;
          for (size_t vecStart = 0; vecStart < resNumElems; ++vecStart) {
            Value newVector = rewriter.create<LLVM::UndefOp>(loc, resElemTy);
            for (size_t elemStart = 0; elemStart < vecSize; ++elemStart) {
              // insert val to newVector
              newVector =
                  insert_element(resElemTy, newVector,
                                 loadedVals[vecStart * vecSize + elemStart],
                                 i32_val(elemStart));
            }
            loadedVecVals.push_back(newVector);
          }
          loadedVals = loadedVecVals;
        }
      } else {
        if (isVectorized) {
          if (bf16Tofp32) {
            if (bf16Tofp32Unordered) {
              VecBF16ToFP32Unordered(op, ctx, loc, rewriter, resElemTy,
                                     resNumElems, vecSize, ptrDataVecSize,
                                     lmBasePtr, loadedVals);
            } else {
              lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
              for (size_t elemIdx = 0; elemIdx < resNumElems / 2; elemIdx++) {
                Value elemPtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, lmBasePtr,
                                    i32_val(elemIdx * stride));
                Value loaded = load(ptrDataVecTy, elemPtr);
                loadedVals.push_back(loaded);
              }
              int remainedIdx = 2 * (resNumElems / 2);
              if (resNumElems - remainedIdx) {
                VectorType halfVecBf16Ty = VectorType::get(vecSize, bf16_ty);
                Value elemPtr = gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr,
                                    i32_val(remainedIdx * stride));
                Value loaded = load(halfVecBf16Ty, elemPtr);
                loadedVals.push_back(loaded);
              }
              VecBF16ToFP32(op, ctx, loc, rewriter, resElemTy, resNumElems,
                            vecSize, ptrDataVecSize, loadedVals);
            }
          } else {
            for (size_t elemIdx = 0; elemIdx < resNumElems; elemIdx++) {
              lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
              Value elemPtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, lmBasePtr,
                                  i32_val(elemIdx * stride));
              Value loaded = load(ptrDataVecTy, elemPtr);
              loadedVals.push_back(loaded);
            }
          }
        } else {
          for (size_t elemIdx = 0; elemIdx < ptrNumElems; elemIdx++) {
            lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
            Value elemPtr = gep(ptr_ty(ctx, 0), ptrElemScalarTy, lmBasePtr,
                                i32_val(elemIdx * stride));
            Value loaded = load(ptrElemScalarTy, elemPtr);
            if (bf16Tofp32) {
              loaded =
                  rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
            }
            loadedVals.push_back(loaded);
          }
        }
      }
    }

    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct XPUStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::StoreOp>,
      public LoadStoreConversionBase {
  XPUStoreOpConversion(LLVMTypeConverter &converter,
                       const xpu::TargetInfo &targetInfo,
                       ModuleAxisInfoAnalysis &axisAnalysisPass,
                       PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  void VecFP32ToBF16Unordered(triton::xpu::StoreOp op, mlir::MLIRContext *ctx,
                              Location &loc,
                              ConversionPatternRewriter &rewriter, int numElems,
                              int valueVecSize, int ptrDataVecSize,
                              SmallVector<Value> &valueElems,
                              Value &lmBasePtr) const {
    VectorType vecBf16Ty = VectorType::get(ptrDataVecSize, bf16_ty);
    VectorType veci16Ty = VectorType::get(ptrDataVecSize, i16_ty);
    VectorType veci32Ty = VectorType::get(valueVecSize, i32_ty);
    VectorType vec1Ty = VectorType::get(ptrDataVecSize, i1_ty);
    VectorType halfVecBf16Ty = VectorType::get(valueVecSize, bf16_ty);
    lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0)); // vecBf16Ty
    int mask = 0xaaaaaaaa;
    Value maskVal = i32_val(mask);
    maskVal = bitcast(maskVal, vec1Ty);
    Value maskNegVal = i32_val(~mask);
    maskNegVal = bitcast(maskNegVal, vec1Ty);
    Value poseVal = i32_val(16);
    uint32_t one = 0x0001;
    uint32_t magic = 0x7fff;
    for (int i = 0; i < numElems / 2; ++i) {
      Value veven = bitcast(valueElems[2 * i], veci32Ty);
      if (!isBf16RoundToMid) {
        SmallVector<Value, 4> vevenAndOperands({i32_val(one), veven});
        auto vevenAnd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, vevenAndOperands, "vand.u.mz $0{mr1}, $1, $2",
            "=&v,r,v",
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        SmallVector<Value, 4> evenOperands({i32_val(magic), vevenAnd.getRes()});
        auto vevenSvAdd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, evenOperands, "vadd.u.mz $0{mr1}, $1, $2", "=&v,r,v",
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        veven = add(veven, vevenSvAdd.getRes());
      }
      veven = bitcast(veven, veci16Ty);
      Value elemPtr = gep(ptr_ty(ctx, 0), vecBf16Ty, lmBasePtr, i32_val(i));
      rewriter.create<mlir::LLVM::XPU::VSTORE_MHOp>(loc, veven, elemPtr,
                                                    maskVal);
      Value vodd = bitcast(valueElems[2 * i + 1], veci32Ty);
      if (!isBf16RoundToMid) {
        SmallVector<Value, 4> oddAndOperands({i32_val(one), vodd});
        auto voddAnd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, oddAndOperands, "vand.u.mz $0{mr1}, $1, $2",
            "=&v,r,v",
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        SmallVector<Value, 4> oddOperands({i32_val(magic), voddAnd.getRes()});
        auto voddSvAdd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, oddOperands, "vadd.u.mz $0{mr1}, $1, $2", "=&v,r,v",
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        vodd = add(vodd, voddSvAdd.getRes());
      }
      Value voddSr = rewriter.create<mlir::LLVM::XPU::SVSRLPOp>(loc, veci32Ty,
                                                                poseVal, vodd);
      voddSr = bitcast(voddSr, veci16Ty);
      rewriter.create<mlir::LLVM::XPU::VSTORE_MHOp>(loc, voddSr, elemPtr,
                                                    maskNegVal);
    }
    if (numElems % 2 == 1) {
      int remainedIdx = numElems - 1;
      lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      Value elemPtr =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i32_val(remainedIdx));
      Value elem = valueElems[remainedIdx];
      Value trunc = rewriter.create<LLVM::FPTruncOp>(loc, halfVecBf16Ty, elem);
      store(trunc, elemPtr);
    }
    return;
  }

  void VecFP32ToBF16(triton::xpu::StoreOp op, mlir::MLIRContext *ctx,
                     Location &loc, ConversionPatternRewriter &rewriter,
                     int numElems, int valueVecSize, int ptrDataVecSize,
                     SmallVector<Value> &valueElems, Value &lmBasePtr) const {
    VectorType vecBf16Ty = VectorType::get(ptrDataVecSize, bf16_ty);
    VectorType vecI16Ty = VectorType::get(ptrDataVecSize, i16_ty);
    VectorType vec1Ty = VectorType::get(ptrDataVecSize, i1_ty);
    VectorType halfVecBf16Ty = VectorType::get(valueVecSize, bf16_ty);
    VectorType veci32Ty = VectorType::get(valueVecSize, i32_ty);
    constexpr int mask = 0xaaaaaaaa; // 0b10101010101010101010101010101010
    Value maskVal = i32_val(mask);
    maskVal = bitcast(maskVal, vec1Ty);
    SmallVector<int16_t> offset_v = {0,  0,  0,  2,  0,  4,  0,  6,  0,  8, 0,
                                     10, 0,  12, 0,  14, 0,  16, 0,  18, 0, 20,
                                     0,  22, 0,  24, 0,  26, 0,  28, 0,  30};
    Value offsetVec = rewriter.create<LLVM::UndefOp>(loc, vecI16Ty);
    for (size_t elemIdx = 0; elemIdx < ptrDataVecSize; ++elemIdx) {
      offsetVec = insert_element(vecI16Ty, offsetVec,
                                 i16_val(offset_v[elemIdx]), i16_val(elemIdx));
    }
    lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0)); // halfVecBf16Ty
    uint32_t one = 0x0001;
    uint32_t magic = 0x7fff;
    for (int i = 0; i < numElems / 2; ++i) {
      Value dstPtr1 =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i16_val(2 * i));
      Value vl = bitcast(valueElems[2 * i], veci32Ty);
      if (!isBf16RoundToMid) {
        SmallVector<Value, 4> vlAndOperands({i32_val(one), vl});
        auto vlAnd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, vlAndOperands, "vand.u.mz $0{mr1}, $1, $2",
            "=&v,r,v",
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        SmallVector<Value, 4> vlOperands({i32_val(magic), vlAnd.getRes()});
        auto vlSvAdd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, vlOperands, "vadd.u.mz $0{mr1}, $1, $2", "=&v,r,v",
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        vl = add(vl, vlSvAdd.getRes());
      }
      vl = bitcast(vl, vecI16Ty);
      rewriter.create<mlir::LLVM::XPU::SCATTER_MHOp>(loc, vl, maskVal, dstPtr1,
                                                     offsetVec);
      Value vh = bitcast(valueElems[2 * i + 1], veci32Ty);
      if (!isBf16RoundToMid) {
        SmallVector<Value, 4> vhAndOperands({i32_val(one), vh});
        auto vhAnd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, vhAndOperands, "vand.u.mz $0{mr1}, $1, $2",
            "=&v,r,v",
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        SmallVector<Value, 4> vhOperands({i32_val(magic), vhAnd.getRes()});
        auto vhSvAdd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, vhOperands, "vadd.u.mz $0{mr1}, $1, $2", "=&v,r,v",
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        vh = add(vh, vhSvAdd.getRes());
      }
      vh = bitcast(vh, vecI16Ty);
      Value dstPtr2 =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i16_val(2 * i + 1));
      rewriter.create<mlir::LLVM::XPU::SCATTER_MHOp>(loc, vh, maskVal, dstPtr2,
                                                     offsetVec);
    }
    if (numElems % 2 == 1) {
      int remainedIdx = numElems - 1;
      Value elemPtr =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i32_val(remainedIdx));
      Value elem = valueElems[remainedIdx];
      Value trunc = rewriter.create<LLVM::FPTruncOp>(loc, halfVecBf16Ty, elem);
      store(trunc, elemPtr);
    }
    return;
  }

  LogicalResult
  matchAndRewrite(triton::xpu::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // original values
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value index = op.getIndex();

    int32_t tensor_col_size = op.getTensorColSize();
    bool coreDealMultiRows = tensor_col_size != -1;
    bool bf16Tofp32Unordered = op.getBf16Tofp32Unordered();

    // adaptor values
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();
    Value llIndex = adaptor.getIndex();

    // Determine Type
    Type ptrTy = ptr.getType();
    auto valueTy = value.getType();

    Type ptrElemTy = typeConverter->convertType(getElementTypeOrSelf(ptrTy));
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type valueElemScalarTy = getElementTypeOrSelf(valueElemTy);
    Type ptrElemScalarTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      ptrElemScalarTy =
          mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
              .getPointeeType();
    } else {
      // Scalar
      ptrElemScalarTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }
    Type ptrDataVecTy = valueElemTy;

    unsigned valueNumElems = getTotalElemsPerThread(valueTy);
    unsigned ptrNumElems = getTotalElemsPerThread(ptrTy);

    // Get the LLVM values
    auto llPtrs = unpackLLElements(loc, llPtr, rewriter);
    auto llVals = unpackLLElements(loc, llValue, rewriter);
    // Determine the vectorization size
    bool isVectorized = mlir::isa<mlir::VectorType>(valueElemTy);
    unsigned valueVecSize = 1u;
    unsigned valueScalarNbits = 32u;
    if (mlir::isa<mlir::VectorType>(valueElemTy)) {
      isVectorized = true;
      getVectorInfo(valueElemTy, valueVecSize, valueScalarNbits);
    }

    // fp32 to bf16
    bool fp32Tobf16 = false;
    if (valueElemScalarTy.isF32() && ptrElemScalarTy.isBF16()) {
      int ptrVecSize = std::min(ptrNumElems, valueVecSize * 2);
      ptrDataVecTy = isVectorized ? VectorType::get(ptrVecSize, ptrElemScalarTy)
                                  : ptrElemScalarTy;
      fp32Tobf16 = true;
    }

    bool fp32Tofp16 = false;
    if (valueElemScalarTy.isF32() && ptrElemScalarTy.isF16()) {
      ptrElemScalarTy = valueElemScalarTy;
      fp32Tofp16 = true;
    }

    unsigned ptrDataVecSize = 1u;
    unsigned ptrDataNbits =
        isa<triton::PointerType, LLVM::LLVMPointerType>(ptrElemScalarTy)
            ? 64u
            : ptrElemScalarTy.getIntOrFloatBitWidth();
    if (mlir::isa<mlir::VectorType>(ptrDataVecTy)) {
      getVectorInfo(ptrDataVecTy, ptrDataVecSize, ptrDataNbits);
    }

    Value lmBasePtr = bitcast(llPtrs[0], ptr_ty(ctx, 0));
    if (index) {
      ptrNumElems = valueNumElems;
      unsigned _stride =
          (fp32Tobf16 && isVectorized) ? ptrNumElems / 2 : ptrNumElems;
      Value idx = mul(llIndex, i32_val(_stride));
      lmBasePtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, lmBasePtr, idx);
    }

    if (coreDealMultiRows) {
      /*  Small Col Size Opt LM2GM (14 legal data)

          Before Opt:
            1 1 1 1 1 1 1 0
            1 1 1 1 1 1 1 0

          After Opt:
            1 1 1 1 1 1 1 1
            1 1 1 1 1 1 0 0
      */
      if (isVectorized) {
        lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
        SmallVector<Value> valueElemsScalar;
        for (size_t vecStart = 0; vecStart < valueNumElems; ++vecStart) {
          for (size_t elemStart = 0; elemStart < valueVecSize; ++elemStart) {
            // extract val to ptrElemScalarTy
            Value ext_val = extract_element(ptrElemScalarTy, llVals[vecStart],
                                            i32_val(elemStart));
            valueElemsScalar.push_back(ext_val);
          }
        }
        llVals = valueElemsScalar;
      }

      auto mem_col_size = mlir::cast<RankedTensorType>(valueTy).getShape()[1] *
                          valueVecSize; // 16
      auto tensor_row_size =
          mlir::cast<RankedTensorType>(valueTy).getShape()[0] /
          64; // 128 / 64 = 2

      for (size_t row_idx = 0; row_idx < tensor_row_size; ++row_idx) {
        for (size_t col_idx = 0; col_idx < tensor_col_size; ++col_idx) {
          auto mem_global_idx = row_idx * mem_col_size + col_idx;
          auto buf_global_idx = row_idx * tensor_col_size + col_idx;
          Value elem = llVals[mem_global_idx];
          if (fp32Tobf16) {
            elem = rewriter.create<LLVM::FPTruncOp>(loc, ptrElemScalarTy, elem);
          }
          Value elemPtr = gep(ptr_ty(ctx, 0), ptrElemScalarTy, lmBasePtr,
                              i32_val(buf_global_idx));
          store(elem, elemPtr);
        }
      }
    } else {
      if (isVectorized) {
        if (valueElemScalarTy.isF32() && ptrElemScalarTy.isBF16()) {
          if (bf16Tofp32Unordered) {
            VecFP32ToBF16Unordered(op, ctx, loc, rewriter, valueNumElems,
                                   valueVecSize, ptrDataVecSize, llVals,
                                   lmBasePtr);
          } else {
            VecFP32ToBF16(op, ctx, loc, rewriter, valueNumElems, valueVecSize,
                          ptrDataVecSize, llVals, lmBasePtr);
          }
        } else {
          lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
          for (size_t elemIdx = 0; elemIdx < valueNumElems; elemIdx++) {
            Value elem = llVals[elemIdx];
            Value elemPtr =
                gep(ptr_ty(ctx, 0), ptrDataVecTy, lmBasePtr, i32_val(elemIdx));
            store(elem, elemPtr);
          }
        }
      } else {
        lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
        for (size_t elemIdx = 0; elemIdx < ptrNumElems; elemIdx++) {
          Value elem = llVals[elemIdx];
          if (fp32Tobf16) {
            elem = rewriter.create<LLVM::FPTruncOp>(loc, ptrElemScalarTy, elem);
          }
          Value elemPtr =
              gep(ptr_ty(ctx, 0), ptrElemScalarTy, lmBasePtr, i32_val(elemIdx));
          store(elem, elemPtr);
        }
      }
    }
    createMfenceOp(rewriter, loc);

    // fp32 to fp16
    if (fp32Tofp16) {
      Value fp16LM = bitcast(llPtrs[0], ptr_ty(ctx, 0));
      Value fp32LM = bitcast(llPtrs[0], ptr_ty(ctx, 0));
      ValueRange singleOperandRange({fp32LM, fp16LM, i32_val(ptrNumElems)});
      mlir::LLVM::XPU::createDeviceCall("_ZN3xpu10fp32tofp16EPKfPNS_7float16Ei",
                                        rewriter, op, singleOperandRange, loc);
      createMfenceOp(rewriter, loc);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct XPUAllocaOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::AllocaOp>,
      public LoadStoreConversionBase {
  XPUAllocaOpConversion(LLVMTypeConverter &converter,
                        const xpu::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::AllocaOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    auto resTy = op.getType();
    Type valueElemTy;
    if (auto resTensorTy = mlir::dyn_cast<RankedTensorType>(resTy)) {
      // Tensor
      valueElemTy =
          mlir::cast<triton::PointerType>(resTensorTy.getElementType())
              .getPointeeType();
    } else {
      // Scalar
      valueElemTy = mlir::cast<triton::PointerType>(resTy).getPointeeType();
    }

    unsigned numElems = getTotalElemsPerThread(resTy);

    auto allocNumElems = numElems;
    if (static_cast<XPUArch>(targetInfo.getXPUArch()) == XPUArch::XPU2 &&
        valueElemTy.isF16()) {
      // algin to 32, cause fp16tofp32 use vector<32*fp16> instruction
      allocNumElems = (allocNumElems + 31) / 32 * 32;
      // double space to accommodate 32*fp32
      allocNumElems *= 2;
    }
    for (auto user : op->getUsers()) {
      if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMOp>(user)) {
        auto fixedStride = gm2lmOp.getFixedStride();
        if (fixedStride > 0 &&
            fixedStride * numElems <= targetInfo.getXPUBufferSize()) {
          allocNumElems *= fixedStride;
        }
      }
    }

    allocNumElems =
        align(allocNumElems, valueElemTy, 64); // 64 bytes aligned for LM
    auto lmPtrTy = LLVM::LLVMPointerType::get(ctx, 0);
    auto lmBuf = allocate(lmPtrTy, valueElemTy, i32_val(allocNumElems));

    SmallVector<Value> lmPtrs;
    for (int i = 0; i < numElems; i++) {
      lmPtrs.push_back(gep(lmPtrTy, valueElemTy, lmBuf, i32_val(i)));
    }
    Type llvmResultStructTy = typeConverter->convertType(resTy);
    Value resultStruct = packLLElements(loc, typeConverter, lmPtrs, rewriter,
                                        llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct XPUGM2LMOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::GM2LMOp>,
      public LoadStoreConversionBase {
  XPUGM2LMOpConversion(LLVMTypeConverter &converter,
                       const xpu::TargetInfo &targetInfo,
                       ModuleAxisInfoAnalysis &axisAnalysisPass,
                       PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::GM2LMOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::GM2LMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value len = op.getLen();
    int32_t tensor_col_size = op.getTensorColSize();
    bool coreDealMultiRows = tensor_col_size != -1;
    bool async = op.getAsync();

    // adaptor values
    Value llLen = adaptor.getLen();
    Value llGMPtr = adaptor.getPtr();
    Value llLMPtr = adaptor.getBufPtr();
    Value resultStruct = llLMPtr;

    Type ptrTy = ptr.getType();
    Type elemTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      elemTy = mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
                   .getPointeeType();
    } else {
      // Scalar
      elemTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }
    unsigned elemNbits = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                             ? 64u
                             : elemTy.getIntOrFloatBitWidth();
    unsigned numElems = getTotalElemsPerThread(ptrTy);

    assert(llLMPtr && "llBufPtr should not be null.");
    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);

    Value elemBytes = i32_val(elemNbits / 8u);
    Value offsetBytes = i32_val(0);

    bool mask = false;
    unsigned lenElemBit = 32;
    llvm::SmallVector<Value> llLens;
    if (op.getLen()) {
      auto lenElemTy = getElementTypeOrSelf(op.getLen().getType());
      lenElemBit = lenElemTy.getIntOrFloatBitWidth();
      mask = llGMPtrs.size() > 1 ? mlir::isa<mlir::IntegerType>(lenElemTy) &&
                                       (lenElemBit == 32 || lenElemBit == 64)
                                 : false;
    }

    Value bufLen = i32_val(numElems);
    Value readLen = bufLen;
    if (mask) {
      llLens = unpackLLElements(loc, llLen, rewriter);
      bufLen = int_val(lenElemBit, numElems);
      readLen = smin(smax(llLens[0], int_val(lenElemBit, 0)), bufLen);
      if (lenElemBit == 64) {
        readLen = trunc(i32_ty, readLen);
      }
    }
    Value readBytes = mul(readLen, elemBytes);

    OffsetState offsetState = static_cast<OffsetState>(op.getOffsetState());
    int32_t fixedStride = op.getFixedStride();
    if (offsetState == OffsetState::Unknown) {
      /*  Small Col Size Opt Mask(14 < 16)

          Before Opt:
              T T T T T T T T
              T T T T T T F F

          After Opt:
              T T T T T T T F
              T T T T T T T F
      */
      SmallVector<bool> maskLists;
      if (coreDealMultiRows) {
        auto mem_col_size =
            mlir::cast<RankedTensorType>(ptrTy).getShape()[1]; // 16
        auto tensor_row_size =
            mlir::cast<RankedTensorType>(ptrTy).getShape()[0] /
            64;                                                  // 128 / 64 = 2
        unsigned rowRemainElem = mem_col_size - tensor_col_size; // 16 - 15 = 1

        for (size_t row_idx = 0; row_idx < tensor_row_size; ++row_idx) {
          for (size_t col_idx = 0; col_idx < tensor_col_size; ++col_idx) {
            maskLists.push_back(true);
          }

          for (size_t remainElem = rowRemainElem; remainElem > 0;
               --remainElem) {
            maskLists.push_back(false);
          }
        }
      }

      if (fixedStride > 0 &&
          numElems * fixedStride <= targetInfo.getXPUBufferSize()) {
        // Unknown FixedStride Vgather
        readBytes = mul(i32_val(fixedStride), readBytes);
      } else {
        // Unknown
        readBytes = elemBytes;
        for (size_t i = 0; i < llGMPtrs.size(); ++i) {
          //  Protect Ptr Boundary Condition
          Value base;
          if (coreDealMultiRows) {
            base = mask ? select(int_val(1, maskLists[i]), llGMPtrs[i],
                                 llGMPtrs[0])
                        : llGMPtrs[i];
          } else {
            if (mask) { // Has Mask
              if (llLens[0].getType().isInteger(32)) {
                base = select(icmp_slt(i32_val(i), llLens[0]), llGMPtrs[i],
                              llGMPtrs[0]);
              } else if (llLens[0].getType().isInteger(64)) {
                base = select(icmp_slt(i64_val(i), llLens[0]), llGMPtrs[i],
                              llGMPtrs[0]);
              } else {
                llvm_unreachable("Unsupported Mask Int Type");
              }
            } else {
              base = llGMPtrs[i];
            }
          }
          Value dstPtr = bitcast(llLMPtrs[i], ptr_ty(ctx, 0));
          Value srcPtr = bitcast(base, ptr_ty(ctx, 1));
          createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                        readBytes);
          if (!async)
            createMfenceOp(rewriter, loc);
        }

        rewriter.replaceOp(op, {resultStruct});
        return success();
      }
    } else if (offsetState == OffsetState::Discrete) {
      // Reorder the local buffer ptrs.
      SmallVector<Value> newLmBufPtrs(llGMPtrs.size());
      Value basePtrInt = ptrtoint(i64_ty, llGMPtrs[0]);
      for (size_t idx = 0; idx < llGMPtrs.size(); ++idx) {
        Value elemPtrInt = ptrtoint(i64_ty, llGMPtrs[idx]); // convert to int
        Value offsetBytes =
            sub(elemPtrInt, basePtrInt); // get the offset(Bytes)
        Value elemPtr = gep(ptr_ty(ctx, 0), i8_ty, llLMPtrs[0], offsetBytes);
        newLmBufPtrs[idx] = elemPtr;
      }
      resultStruct = packLLElements(loc, typeConverter, newLmBufPtrs, rewriter,
                                    llLMPtr.getType());
    } else if (offsetState == OffsetState::DiscreteSame) {
      readBytes = elemBytes;
      SmallVector<Value> newLmBufPtrs(llLMPtrs.size(), llLMPtrs[0]);
      resultStruct = packLLElements(loc, typeConverter, newLmBufPtrs, rewriter,
                                    llLMPtr.getType());
    } else if (offsetState == OffsetState::LocallyContinuous) {
      int64_t _rowLen = op.getRowLen();
      int64_t _rowStride = op.getRowStride();
      if (_rowLen % numElems == 0) {
        offsetState = OffsetState::Continuous;
        LLVM_DEBUG(
            llvm::dbgs()
            << "[OffsetState]: GM2LM Update LocallyContinuous to Continuous\n");
      } else {
        auto oldBlock = op->getBlock();
        auto newBlock = oldBlock->splitBlock(op->getNextNode());
        int64_t _elemBytes = elemNbits / 8u;
        int64_t _bufLen = static_cast<int64_t>(numElems);
        LLVM_DEBUG(llvm::dbgs() << "[GM2LM LocallyContinuous]: rowLen is "
                                << _rowLen << ", rowStride is " << _rowStride
                                << ", bufLen is " << _bufLen << "\n");
        if (_rowStride == -1) {
          lowerLocallyContinuousUnfixedStride(
              op, loc, rewriter, _rowLen, _bufLen, _elemBytes, llGMPtr, llLMPtr,
              llLen, offsetBytes, MemCpyType::GM2LM, oldBlock, newBlock);
        } else {
          if (_rowLen > _bufLen) {
            lowerLocallyContinuousLargeRow(
                op, loc, rewriter, _rowLen, _rowStride, llGMPtr, llLMPtr, llLen,
                bufLen, elemBytes, offsetBytes, MemCpyType::GM2LM, oldBlock,
                newBlock);
          } else {
            lowerLocallyContinuousSmallRow(
                op, loc, rewriter, _rowLen, _rowStride, llGMPtr, llLMPtr, llLen,
                bufLen, elemBytes, offsetBytes, MemCpyType::GM2LM, oldBlock,
                newBlock);
          }
        }

        if (!async)
          createMfenceOp(rewriter, loc);

        resultStruct = packLLElements(loc, typeConverter, llLMPtrs, rewriter,
                                      llLMPtr.getType());
        rewriter.replaceOp(op, {resultStruct});
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, newBlock);
        return success();
      }
    }

    Value dstPtr = bitcast(llLMPtrs[0], ptr_ty(ctx, 0));
    Value srcPtr = bitcast(llGMPtrs[0], ptr_ty(ctx, 1));
    createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes, readBytes);
    if (!async)
      createMfenceOp(rewriter, loc);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct XPULM2GMOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::LM2GMOp>,
      public LoadStoreConversionBase {

  XPULM2GMOpConversion(LLVMTypeConverter &converter,
                       const xpu::TargetInfo &targetInfo,
                       ModuleAxisInfoAnalysis &axisAnalysisPass,
                       PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::LM2GMOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::LM2GMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value len = op.getLen();
    int32_t offsetStateInt = op.getOffsetState();
    OffsetState offsetState = static_cast<OffsetState>(offsetStateInt);
    auto tensor_col_size = op.getTensorColSize();

    // adaptor values
    Value llPtr = adaptor.getPtr();
    Value llLen = adaptor.getLen();
    Value llBufPtr = adaptor.getBufPtr();
    assert(llBufPtr && "llBufPtr should not be null.");

    // Get elemTy and numElems
    Type ptrTy = ptr.getType();
    Type ptrElemTy = typeConverter->convertType(getElementTypeOrSelf(ptrTy));
    Type elemTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      elemTy = mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
                   .getPointeeType();
    } else {
      // Scalar
      elemTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }
    unsigned elemNbits = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                             ? 64u
                             : elemTy.getIntOrFloatBitWidth();
    Value elemBytes = i32_val(elemNbits / 8u);
    unsigned numElems = getTotalElemsPerThread(ptrTy);

    // Get base, readBytes and offsetBytes
    auto llPtrs = unpackLLElements(loc, llPtr, rewriter);

    Value base = llPtrs[0];
    Value offsetBytes = i32_val(0);

    llvm::SmallVector<Value> llLens;
    bool mask = false;
    unsigned lenElemBit = 32;
    if (op.getLen()) {
      auto lenElemTy = getElementTypeOrSelf(op.getLen().getType());
      lenElemBit = lenElemTy.getIntOrFloatBitWidth();
      mask = llPtrs.size() > 1 ? mlir::isa<mlir::IntegerType>(lenElemTy) &&
                                     (lenElemBit == 32 || lenElemBit == 64)
                               : false;
    }
    Value bufLen = i32_val(numElems);
    Value readLen = bufLen;
    if (mask) {
      llLens = unpackLLElements(loc, llLen, rewriter);
      bufLen = int_val(lenElemBit, numElems);
      readLen = smin(smax(llLens[0], int_val(lenElemBit, 0)), bufLen);
      if (lenElemBit == 64) {
        readLen = trunc(i32_ty, readLen);
      }
    }
    Value readBytes = mul(readLen, elemBytes);
    auto lmBufPtrs = unpackLLElements(loc, llBufPtr, rewriter);
    Value lmBuf = lmBufPtrs[0];

    // Create LM2GM and mfence
    switch (offsetState) {
    case OffsetState::Continuous: {
      Value srcPtr = bitcast(lmBuf, ptr_ty(ctx, 0));
      Value basePtr = bitcast(base, ptr_ty(ctx, 1));
      createLM2GMOp(rewriter, ctx, loc, srcPtr, basePtr, offsetBytes,
                    readBytes);
      break;
    }
    case OffsetState::LocallyContinuous: {
      int64_t _rowLen = op.getRowLen();
      int64_t _rowStride = op.getRowStride();
      if (_rowLen % numElems == 0) {
        offsetState = OffsetState::Continuous;
        LLVM_DEBUG(
            llvm::dbgs()
            << "[OffsetState]: LM2GM Update LocallyContinuous to Continuous\n");
      } else {
        auto oldBlock = op->getBlock();
        auto newBlock = oldBlock->splitBlock(op->getNextNode());
        int64_t _elemBytes = elemNbits / 8u;
        int64_t _bufLen = static_cast<int64_t>(numElems);
        LLVM_DEBUG(llvm::dbgs() << "[LM2GM LocallyContinuous]: rowLen is "
                                << _rowLen << ", rowStride is " << _rowStride
                                << ", bufLen is " << _bufLen << "\n");

        if (_rowStride == -1) {
          lowerLocallyContinuousUnfixedStride(
              op, loc, rewriter, _rowLen, _bufLen, _elemBytes, llPtr, llBufPtr,
              llLen, offsetBytes, MemCpyType::LM2GM, oldBlock, newBlock);
        } else {
          if (_rowLen > _bufLen) {
            lowerLocallyContinuousLargeRow(
                op, loc, rewriter, _rowLen, _rowStride, llPtr, llBufPtr, llLen,
                bufLen, elemBytes, offsetBytes, MemCpyType::LM2GM, oldBlock,
                newBlock);
          } else {
            lowerLocallyContinuousSmallRow(
                op, loc, rewriter, _rowLen, _rowStride, llPtr, llBufPtr, llLen,
                bufLen, elemBytes, offsetBytes, MemCpyType::LM2GM, oldBlock,
                newBlock);
          }
        }
        createMfenceOp(rewriter, loc);
        rewriter.eraseOp(op);
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, newBlock);
        return success();
      }
      break;
    }
    case OffsetState::Unknown: {
      for (size_t llPtrIdx = 0; llPtrIdx < llPtrs.size(); ++llPtrIdx) {
        Value maskedIdx;
        if (mask) {
          auto llLenTy = llLens[0].getType();
          if (llLenTy.isInteger(32)) {
            maskedIdx = select(icmp_slt(i32_val(llPtrIdx), llLens[0]),
                               i32_val(llPtrIdx), i32_val(0));
          } else if (llLenTy.isInteger(64)) {
            maskedIdx = select(icmp_slt(i64_val(llPtrIdx), llLens[0]),
                               i64_val(llPtrIdx), i64_val(0));
          } else {
            llvm_unreachable("Unsupported Mask Int Type");
          }
        } else {
          maskedIdx = i32_val(llPtrIdx);
        }

        lmBuf = bitcast(lmBuf, ptr_ty(ctx, 0));
        Value elemPtr = gep(ptr_ty(ctx, 0), elemTy, lmBuf, maskedIdx);
        Value srcPtr = bitcast(elemPtr, ptr_ty(ctx, 0));
        // Protect Ptr Boundary Condition
        Value dstPtr;
        if (mask) {
          auto llLenTy = llLens[0].getType();
          if (llLenTy.isInteger(32)) {
            dstPtr = select(icmp_slt(i32_val(llPtrIdx), llLens[0]),
                            llPtrs[llPtrIdx], llPtrs[0]);
          } else if (llLenTy.isInteger(64)) {
            dstPtr = select(icmp_slt(i64_val(llPtrIdx), llLens[0]),
                            llPtrs[llPtrIdx], llPtrs[0]);
          } else {
            llvm_unreachable("Unsupported Mask Int Type");
          }
        } else {
          dstPtr = llPtrs[llPtrIdx];
        }
        createLM2GMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                      elemBytes);
      }
      break;
    }
    default:
      llvm_unreachable("Unknown offset state");
      break;
    }
    createMfenceOp(rewriter, loc);
    rewriter.eraseOp(op);

    return success();
  }
};

struct XPUAtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {

  XPUAtomicRMWOpConversion(LLVMTypeConverter &converter,
                           const xpu::TargetInfo &targetInfo,
                           ModuleAxisInfoAnalysis &axisAnalysisPass,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value ptr = op.getPtr();
    Value val = op.getVal();
    Value mask = op.getMask();
    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value llPtr = adaptor.getPtr();
    Value llValue = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto llPtrs = unpackLLElements(loc, llPtr, rewriter);
    auto llValues = unpackLLElements(loc, llValue, rewriter);

    auto resTy = op.getType();
    Type valueElemTy = getElementTypeOrSelf(getElementTypeOrSelf(resTy));
    unsigned numElems = getTotalElemsPerThread(resTy);

    std::string funcName;
    if (valueElemTy.isF16()) {
      switch (atomicRmwAttr) {
      case RMWOp::ADD:
        funcName = "_ZN3xpu9atomicAddEPU3AS1DF16_DF16_";
        break;
      case RMWOp::FADD:
        funcName = "_ZN3xpu9atomicAddEPU3AS1DF16_DF16_";
        break;
      default:
        return failure();
      }
    } else {
      switch (atomicRmwAttr) {
      case RMWOp::ADD:
        funcName = "_ZN3xpu9atomicAddEPU3AS1ff";
        break;
      case RMWOp::FADD:
        funcName = "_ZN3xpu9atomicAddEPU3AS1ff";
        break;
      default:
        return failure();
      }
    }

    SmallVector<Value> resultVals(numElems);
    for (unsigned i = 0; i < numElems; ++i) {
      ValueRange operandRange({llPtrs[i], llValues[i]});
      Value devCall = mlir::LLVM::XPU::createDeviceCall(
          funcName, rewriter, op, valueElemTy, operandRange, loc);
      resultVals[i] = devCall;
    }

    Type structTy = this->getTypeConverter()->convertType(resTy);
    Value resultStruct =
        packLLElements(loc, typeConverter, resultVals, rewriter, structTy);
    rewriter.replaceOp(op, resultStruct);

    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  patterns.add<XPULoadOpConversion, XPUStoreOpConversion, XPUAllocaOpConversion,
               XPUGM2LMOpConversion, XPULM2GMOpConversion,
               XPUAtomicRMWOpConversion>(typeConverter, targetInfo,
                                         axisInfoAnalysis, benefit);
}
