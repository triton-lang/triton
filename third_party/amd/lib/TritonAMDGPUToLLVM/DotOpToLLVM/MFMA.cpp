/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
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
#include "../PatternTritonGPUOpToLLVM.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

using ::mlir::LLVM::AMD::scaleDotElemTypeToMLIRType;
using ::mlir::LLVM::AMD::shuffleXor;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::LinearEncodingAttr;

using ValueTable = std::map<std::array<int, 3>, Value>;

/// Get matrix format flag passed through BLGP/CBSZ args in V_MFMA_*_F8F6F4
/// instructions.
///
/// Values:
/// - 0: E4M3(FP8)
/// - 1: E5M2(BF8)
/// - 2: E2M3(FP6)
/// - 3: E3M2(BF6)
/// - 4: E2M1(FP4)
static inline int32_t getMfmaF8F6F4MatrixFormat(Type t) {
  return llvm::TypeSwitch<Type, int32_t>(t)
      .Case<Float8E4M3FNType>([](Type) { return 0; })
      .Case<Float8E5M2Type>([](Type) { return 1; })
      .Case<Float6E3M2FNType>([](Type) { return 2; })
      .Case<Float6E2M3FNType>([](Type) { return 3; })
      .Case<Float4E2M1FNType>([](Type) { return 4; })
      .Default([](Type) { return -1; });
}

struct DotOpMFMAConversionHelper {
  AMDMfmaEncodingAttr mfmaLayout;

  ConversionPatternRewriter &rewriter;
  const LLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  virtual ~DotOpMFMAConversionHelper() = default;

  explicit DotOpMFMAConversionHelper(AMDMfmaEncodingAttr mfmaLayout,
                                     ConversionPatternRewriter &rewriter,
                                     const LLVMTypeConverter *typeConverter,
                                     Location loc)
      : mfmaLayout(mfmaLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(mfmaLayout.getContext()) {}

  Value generateMFMAOp(StringRef intrinsicName, Value valA, Value valB,
                       Value valC, int cbsz = 0, int abid = 0,
                       int blgp = 0) const {
    assert(cbsz >= 0 && cbsz <= 4);
    assert(abid >= 0 && abid <= 15);
    assert(blgp >= 0 && blgp <= 7);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value zeroFlag = b.i32_val(0);
    Value cbszFlag = cbsz != 0 ? b.i32_val(cbsz) : zeroFlag;
    Value abidFlag = abid != 0 ? b.i32_val(abid) : zeroFlag;
    Value blgpFlag = blgp != 0 ? b.i32_val(blgp) : zeroFlag;

    auto resType = valC.getType();
    OperationState loweredOp(loc, intrinsicName);
    loweredOp.addTypes(resType);
    loweredOp.addOperands({valA, valB, valC, cbszFlag, abidFlag, blgpFlag});
    return rewriter.create(loweredOp)->getResult(0);
  }

  int getNumSubmatrices(Type elementType, int mDim, int nDim) const {
    if (mDim == 4 || nDim == 4)
      return 16;
    return 1;
  }

  Value processSubBlocks(int numSubBlocks, Value acc, bool reduceSubBlocks,
                         bool zeroSubBlocks) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    assert((numSubBlocks & (numSubBlocks - 1)) == 0 &&
           "numSubBlocks in not pow 2!");
    if (numSubBlocks == 1)
      return acc;
    constexpr int warpSize = 64;
    int subBlockSize = warpSize / numSubBlocks;
    Value laneId = getLaneId(rewriter, loc);
    auto vecTy = dyn_cast<VectorType>(acc.getType());
    auto elemType = vecTy.getElementType();
    assert(elemType.getIntOrFloatBitWidth() == 32);
    int numScalars = vecTy.getNumElements();
    std::vector<Value> accScalar(numScalars);
    for (int i = 0; i < numScalars; ++i)
      accScalar[i] = b.extract_element(elemType, acc, b.i32_val(i));

    if (reduceSubBlocks) {
      while (subBlockSize < warpSize) {
        for (int i = 0; i < numScalars; ++i) {
          Value other_acc =
              shuffleXor(loc, rewriter, accScalar[i], subBlockSize);
          if (elemType.isInteger(32))
            accScalar[i] = b.add(accScalar[i], other_acc);
          else
            accScalar[i] = b.fadd(accScalar[i], other_acc);
        }
        subBlockSize *= 2;
      }
    }
    if (zeroSubBlocks) {
      Value zero;
      if (elemType.isInteger(32))
        zero = b.i32_val(0);
      else if (elemType.isF64())
        zero = b.f64_val(0.0);
      else
        zero = b.f32_val(0.0);
      auto cond = b.icmp_ult(laneId, b.i32_val(subBlockSize));
      for (int i = 0; i < numScalars; ++i)
        accScalar[i] = b.select(cond, accScalar[i], zero);
    }

    Value reducedAcc = b.undef(vecTy);
    for (int i = 0; i < numScalars; ++i)
      reducedAcc =
          b.insert_element(vecTy, reducedAcc, accScalar[i], b.i32_val(i));
    return reducedAcc;
  }

  /// @brief MFMA 4x4 is computes 16 matrix multiplications, this functions adds
  /// these 16 matrices to get final 4x4 matrix
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value reduceSubBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, true, false);
  }

  /// @brief Zeroes out redundant values in all sub-blocks except first one
  ///
  /// Every warp in mfma 4x4 layout holds only 4 unique values(scalar or
  /// vectors) in blocks of 4 consecutive threads, There are 16 copies of these
  /// 4 values across all threads of the warp. Need to zero out 15 copies to use
  /// accumulator between dot operations.
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value zeroAuxiliarBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, false, true);
  }

  /// Dot operand layout minimal tile is kDimInstrSize elements across
  /// K dimension. If dot operand K dimension is smaller, layout
  /// assigns tensor elements to multiple different hardware locations.
  /// In this case mfma instruction adds elements in accumulator
  /// multiple times.
  ///
  /// Let say A=[1,2]; B=[3,4], C = A*B = 1*3+2*4 = 11
  /// Consider instruction K size is 4,
  /// in this case operands will be duplicated:
  /// A' = [1,2,1,2] B' = [3,4,3,4]
  /// C' = (1*3+2*4) + (1*3+2*4) = 22
  ///
  /// Following code adjusts accumulator values in such cases.
  /// If accumulator is integer, shift accumulator right by
  /// log2(duplicationRate). If accumulator is float, multiply accum
  /// with 1/duplicationRate constant.
  void adjustAccForSmallKDim(SmallVector<Value> &fc, Value &acc, Type dstElemTy,
                             int b, int m, int n, int64_t numRepM,
                             int64_t numRepN, int64_t kDimInstrSize,
                             int64_t kDimOperandSize,
                             unsigned elemsPerVec) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    for (unsigned v = 0; v < elemsPerVec; ++v) {
      Value accElem = tb.extract_element(dstElemTy, acc, tb.i32_val(v));
      if (kDimInstrSize > kDimOperandSize) {
        assert(kDimInstrSize % kDimOperandSize == 0);
        int duplicationRate = kDimInstrSize / kDimOperandSize;
        assert(llvm::isPowerOf2_32(duplicationRate));
        if (dstElemTy.isInteger()) {
          auto shiftSize = llvm::Log2_32(duplicationRate);
          assert(!accElem.getType().isUnsignedInteger() &&
                 "MFMA uses signed accumulator");
          accElem = tb.ashr(accElem, tb.i32_val(shiftSize));
        } else {
          auto multiplierAttr =
              rewriter.getFloatAttr(dstElemTy, 1.0 / duplicationRate);
          auto multiplierVal =
              rewriter.create<LLVM::ConstantOp>(loc, dstElemTy, multiplierAttr);
          accElem = tb.fmul(accElem, multiplierVal);
        }
      }
      auto linearIdx = b * numRepM * numRepN * elemsPerVec +
                       m * numRepN * elemsPerVec + n * elemsPerVec + v;
      fc[linearIdx] = accElem;
    }
  }

  template <typename T>
  void packAndReplaceResult(T &op, SmallVector<Value> &fc,
                            const FailureOr<MfmaIntrinsic> &maybeMfmaIntrinsic,
                            Type dstElemTy, Type elemtTy,
                            size_t mmaCount) const {
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(fc.size(), dstElemTy));
    Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);

    rewriter.replaceOp(op, res);
  }

  // Conduct the Dot conversion.
  LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    // Check if this dot has come with priority set by setprio.
    auto setPrioOp = dyn_cast_or_null<ROCDL::SetPrioOp>(op->getPrevNode());

    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto mnkDim = mfmaLayout.getInstrShape();
    auto mDim = mnkDim[0];
    auto nDim = mnkDim[1];
    auto kDim = mnkDim[2];
    auto mfmaVersion = mfmaLayout.getVersion();

    Value a = op.getA();
    Value b = op.getB();
    Value d = op.getD();
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto dTensorTy = cast<RankedTensorType>(d.getType());
    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();

    const auto kDimOperandSize = aTensorTy.getShape().back();

    bool allowXF32 =
        op.getInputPrecision() == InputPrecision::TF32 && mfmaVersion == 3;
    StringRef intrinsicName;
    FailureOr<MfmaIntrinsic> maybeMfmaIntrinsic = MfmaIntrinsic::get(
        op.getLoc(), mfmaVersion, mDim, nDim, kDim, elemTyA, elemTyB,
        /*withScale=*/false, allowXF32);
    if (failed(maybeMfmaIntrinsic))
      return op.emitError(
          "no matching matrix core intrinsic due to unsupported element type");

    unsigned kBase = maybeMfmaIntrinsic->kBase;

    auto aEncoding = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    auto bEncoding = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
    int kWidth = aEncoding.getKWidth();

    intrinsicName = maybeMfmaIntrinsic->name;

    // If we are using XF32, the kWidth (and kBase) is double that of F32.
    if (aTensorTy.getElementType().isF32() && allowXF32)
      kWidth *= 2;

    const auto kDimInstrSize = mfmaLayout.getInstrShapeForOperand(kWidth, 0)[1];

    auto repA = mfmaLayout.getRepForOperand(aTensorTy.getShape(), kWidth, 0);
    auto repB = mfmaLayout.getRepForOperand(bTensorTy.getShape(), kWidth, 1);

    assert(repA[2] == repB[1]);

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();

    auto numRepM = repA[1];
    auto numRepN = repB[2];
    auto numRepK = repA[2];
    auto numRepB = repA[0];
    assert(repA[0] == repB[0]);

    int numBroadcastA = 1;
    int numBroadcastB = 1;
    int numRepKA = numRepK;
    int numRepKB = numRepK;
    if ((mDim == 64 && nDim == 4)) {
      numBroadcastB = 16;
      numRepKA *= 16;
    }
    if ((mDim == 4 && nDim == 64)) {
      numBroadcastA = 16;
      numRepKB *= 16;
    }
    numRepK = std::max(numRepKA, numRepKB);
    int numBroadcast = std::max(numBroadcastA, numBroadcastB);

    bool preserveBF16 = intrinsicName.contains(".bf16") && mfmaVersion >= 4;
    auto operandA = getValuesFromDotOperandLayoutStruct(
        loadedA, numRepB, numRepM, numRepKA, kWidth, kBase,
        aTensorTy.getElementType(), allowXF32, preserveBF16);
    auto operandB = getValuesFromDotOperandLayoutStruct(
        loadedB, numRepB, numRepN, numRepKB, kWidth, kBase,
        aTensorTy.getElementType(), allowXF32, preserveBF16);

    int warpSize = triton::gpu::lookupThreadsPerWarp(rewriter);
    int elemsPerVec = mDim * nDim / warpSize;
    int numVecInKBase = numRepK * kWidth / kBase;

    auto dstElemTy = dTensorTy.getElementType();
    auto fc = unpackLLElements(loc, loadedC, rewriter);
    SmallVector<int64_t> fcStrides =
        computeStrides({numRepB, numRepM, numRepN, elemsPerVec});

    Value firstMfma;
    auto vecTy = vec_ty(dstElemTy, elemsPerVec);
    for (int b = 0; b < numRepB; ++b) {
      for (int m = 0; m < numRepM; ++m) {
        for (int n = 0; n < numRepN; ++n) {
          Value acc = tb.undef(vecTy);

          for (int v = 0; v < elemsPerVec; ++v) {
            int linearIdx = linearize({b, m, n, v}, fcStrides);
            Value c = fc[linearIdx];
            acc = tb.insert_element(vecTy, acc, c, tb.i32_val(v));
          }

          for (int k = 0; k < numVecInKBase; ++k) {
            Value op1 = operandA[{b, m, k}];
            Value op2 = operandB[{b, n, k}];
            int cbsz = 0;
            int abid = 0;

            if (numBroadcastA > 1) {
              assert(!mfmaLayout.getIsTransposed());
              cbsz = llvm::Log2_32(numBroadcastA);
              abid = k % numBroadcastA;
              op1 = operandA[{b, m, k / numBroadcastA}];
            }

            if (numBroadcastB > 1) {
              assert(numBroadcastA == 1);
              assert(mfmaLayout.getIsTransposed());
              cbsz = llvm::Log2_32(numBroadcastB);
              abid = k % numBroadcastB;
              op2 = operandB[{b, n, k / numBroadcastB}];
            }

            if (mfmaLayout.getIsTransposed())
              std::swap(op1, op2);

            acc = generateMFMAOp(intrinsicName, op1, op2, acc, cbsz, abid);

            if (!firstMfma)
              firstMfma = acc;
          }

          adjustAccForSmallKDim(fc, acc, dstElemTy, b, m, n, numRepM, numRepN,
                                kDimInstrSize, kDimOperandSize, elemsPerVec);
        }
      }
    }

    // Originally, setprio (high) is set to the high-level dot op. After dot is
    // being lowered to the series of mfma operations, it should be moved next
    // to the first mfma leaving the first mfma staying at the low priority. In
    // this way, incoming warp can be effectively waiting on the first mfma
    // instruction (low priority) while the other warp is executing mfma with
    // high priority. Otherwise, incoming warp can break the cluster.
    if (setPrioOp && firstMfma)
      setPrioOp->moveAfter(firstMfma.getDefiningOp());

    const size_t mmaCount = numRepB * numRepM * numRepN * numVecInKBase;
    packAndReplaceResult(op, fc, maybeMfmaIntrinsic, dstElemTy, elemTyA,
                         mmaCount);

    return success();
  }

  /// Process the elements in rawElems and prepare a vector for mfma input.
  /// rawElems is a vector of kBase elements. Each element is of the raw
  /// element type from the input. We need to prepare a vector of kBase
  /// elements of appropriate element type required by mfma instructions.
  Value prepareOperands(Value rawElems, int kBase, Type type, bool preserveBF16,
                        bool isConstantScale = false) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value results;

    // Construct a vector type of kBase elements with desired type
    auto vecTy = vec_ty(type, kBase);
    if (type.isBF16() && !preserveBF16)
      vecTy = vec_ty(i16_ty, kBase);
    Value vec = b.undef(vecTy);

    // For each element in rawElems, extract the element as the desired type,
    // bitcast it if needed, and insert it into vec.
    for (int elemId = 0; elemId < kBase; ++elemId) {
      auto val = b.extract_element(type, rawElems, b.i32_val(elemId));
      if (type.isBF16() && !preserveBF16) {
        // rocdl.mfma.f32.32x32x8bf16.1k calls for input of i16 type
        auto cast = b.bitcast(val, i16_ty);
        vec = b.insert_element(vecTy, vec, cast, b.i32_val(elemId));
      } else {
        vec = b.insert_element(vecTy, vec, val, b.i32_val(elemId));
      }
    }

    // Now we have a vector of kBase elements of desired type.
    // Then we need to prepare vec for results.
    if (type.getIntOrFloatBitWidth() == 8) {
      if (1 == kBase) {
        // This is only for the scale operands of scaled mfma on CDNA4
        if (isConstantScale) {
          // If the scale is constant(created by arith::ConstantOp), it will
          // be put in a sgpr instead of vgpr. In that case, instead of
          // vgpr[7:0], the instruction reads sgpr[30:23] as the scale value.
          // So we need to manually left shift the scale by 23 bits to meet
          // the requirement.
          results = b.shl(i32_ty, b.zext(i32_ty, b.bitcast(vec, i8_ty)),
                          b.i32_val(23));
        } else {
          results = b.zext(i32_ty, b.bitcast(vec, i8_ty));
        }
      }

      if (2 == kBase)
        // This case can occur during scale tensor packing when there aren't
        // enough elements to fill all 4 opSel slots. For example, with an A
        // tensor of size 16x256 and using 16x16x128 block sizes, we end up with
        // only 2 elements to pack,  resulting in a kBase of 2.
        results = b.zext(i32_ty, b.bitcast(vec, i16_ty));
      if (4 == kBase)
        // This is for int8 on pre- CDNA3 GPUs and scale tensors on CDNA4 GPUs
        results = b.bitcast(vec, i32_ty);
      if (8 == kBase)
        results = b.bitcast(vec, i64_ty);
      if (16 == kBase)
        // This is only for the operands of scaled mfma on CDNA4
        results = b.bitcast(vec, vec_ty(i32_ty, 4));
      if (32 == kBase)
        results = b.bitcast(vec, vec_ty(i32_ty, 8));
    } else {
      results = vec;
    }
    return results;
  }

  /// Converts dot operand structure to value table and converts types
  /// appropriate for mfma instructions
  virtual ValueTable getValuesFromDotOperandLayoutStruct(
      Value value, int batch, int nonKRep, int kRepInKWidth, int kWidth,
      int kBase, Type type, bool allowXF32, bool preserveBF16,
      bool isConstantScale = false) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    auto elems = unpackLLElements(loc, value, rewriter);
    // number of kBase-element vectors
    int numVecInKBase = kRepInKWidth * kWidth / kBase;
    if (numVecInKBase == 0) {
      numVecInKBase = 1;
      nonKRep /= kBase / (kRepInKWidth * kWidth);
      assert(nonKRep > 0 && "nonKrep too small");
    }
    ValueTable dotOpVals;

    SmallVector<int64_t> strides =
        computeStrides({batch, nonKRep, numVecInKBase, kBase});
    for (int b = 0; b < batch; ++b) {
      for (int nonK = 0; nonK < nonKRep; nonK++) {
        for (int kBaseVec = 0; kBaseVec < numVecInKBase; kBaseVec++) {
          // For each kBase-element vector

          // Step 1: construct each kBase-element vector by
          //         - extracting kBase elements from elems and
          //         - putting them into a kBase-element vector, i.e. rawElems
          Type elemTy = typeConverter->convertType(type);
          Type ty = vec_ty(elemTy, kBase);
          Value rawElems = tb.undef(ty);
          for (int k = 0; k < kBase; ++k) {
            auto index = linearize({b, nonK, kBaseVec, k}, strides);
            rawElems =
                tb.insert_element(ty, rawElems, elems[index], tb.i32_val(k));
          }

          // Step 2: process rawElems based on element type
          // Note that for f32/fp64 input and XF32 is not allowed, nothing needs
          // to be done and rawElems is inserted into the ValueTable directly
          if ((type.isF32() || type.isF64()) && !allowXF32) {
            dotOpVals[{b, nonK, kBaseVec}] =
                tb.extract_element(type, rawElems, tb.i32_val(0));
          } else {
            Value vals;
            if (type.isF32() && allowXF32) {
              vals = prepareOperands(rawElems, kBase, f32_ty, preserveBF16);
            } else if (type.getIntOrFloatBitWidth() == 8) {
              vals = prepareOperands(rawElems, kBase, i8_ty, preserveBF16,
                                     isConstantScale);
            } else if (type.isBF16()) {
              vals = prepareOperands(rawElems, kBase, bf16_ty, preserveBF16);
            } else {
              assert(type.isF16() && "Unsupported data type");
              vals = prepareOperands(rawElems, kBase, f16_ty, preserveBF16);
            }

            // Step 3: Insert the processed vals into the ValueTable
            dotOpVals[{b, nonK, kBaseVec}] = vals;
          }
        }
      }
    }
    return dotOpVals;
  }
};

struct ScaledDotOpMFMAConversionHelper : DotOpMFMAConversionHelper {
  virtual ~ScaledDotOpMFMAConversionHelper() = default;

  ScaledDotOpMFMAConversionHelper(AMDMfmaEncodingAttr mfmaLayout,
                                  ConversionPatternRewriter &rewriter,
                                  const LLVMTypeConverter *typeConverter,
                                  Location loc)
      : DotOpMFMAConversionHelper(mfmaLayout, rewriter, typeConverter, loc) {}

  Value generateScaledMFMAOp(StringRef intrinsicName, Value valA, Value valB,
                             Value valC, Type elemTypeA, Type elemTypeB) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto resType = valC.getType();
    Value zeroFlag = b.i32_val(0);
    OperationState loweredOp(loc, intrinsicName);
    int32_t cbsz = getMfmaF8F6F4MatrixFormat(elemTypeA);
    int32_t blgp = getMfmaF8F6F4MatrixFormat(elemTypeB);
    assert((cbsz != -1) && (blgp != -1));
    loweredOp.addTypes(resType);
    // If both scales are constant 0, the LLVM backend will use V_MFMA_*_F8F6F4
    // instructions instead of V_MFMA_SCALE_*_F8F6F4 to reduce memory access.
    loweredOp.addOperands({valA, valB, valC, b.i32_val(cbsz), b.i32_val(blgp),
                           zeroFlag, zeroFlag, zeroFlag, zeroFlag});
    return rewriter.create(loweredOp)->getResult(0);
  }

  Value generateScaledMFMAOp(StringRef intrinsicName, Value valA, Value valB,
                             Value valC, Value valScaleA, Value valScaleB,
                             Type elemTypeA, Type elemTypeB, int opSelA,
                             int opSelB) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto resType = valC.getType();
    Value valOpSelA = b.i32_val(opSelA);
    Value valOpSelB = b.i32_val(opSelB);
    OperationState loweredOp(loc, intrinsicName);
    int32_t cbsz = getMfmaF8F6F4MatrixFormat(elemTypeA);
    int32_t blgp = getMfmaF8F6F4MatrixFormat(elemTypeB);
    assert((cbsz != -1) && (blgp != -1));
    loweredOp.addTypes(resType);
    loweredOp.addOperands({valA, valB, valC, b.i32_val(cbsz), b.i32_val(blgp),
                           valOpSelA, valScaleA, valOpSelB, valScaleB});
    return rewriter.create(loweredOp)->getResult(0);
  }

  LogicalResult convertScaledDot(DotScaledOp op,
                                 DotScaledOpAdaptor adaptor) const {
    // Check if this dot has come with priority set by setprio.
    auto setPrioOp = dyn_cast_or_null<ROCDL::SetPrioOp>(op->getPrevNode());

    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto mnkDim = mfmaLayout.getInstrShape();
    auto mDim = mnkDim[0];
    auto nDim = mnkDim[1];
    auto kDim = mnkDim[2];
    auto mfmaVersion = mfmaLayout.getVersion();

    Value a = op.getA();
    Value b = op.getB();
    Value aScale = op.getAScale();
    Value bScale = op.getBScale();
    if ((aScale && !bScale) || (!aScale && bScale)) {
      llvm::report_fatal_error("Single scale is not supported\n");
    }

    bool existBothScales = aScale && bScale;
    bool isAScaleConstant = aScale && isa<arith::ConstantOp, triton::SplatOp>(
                                          aScale.getDefiningOp());
    bool isBScaleConstant = bScale && isa<arith::ConstantOp, triton::SplatOp>(
                                          bScale.getDefiningOp());
    Value d = op.getD();
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto dTensorTy = cast<RankedTensorType>(d.getType());
    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();
    ScaleDotElemType aElemType = op.getAElemType();
    ScaleDotElemType bElemType = op.getBElemType();

    auto supportsTypes = [](ScaleDotElemType elemType) {
      return elemType == ScaleDotElemType::E2M1 ||
             elemType == ScaleDotElemType::E4M3 ||
             elemType == ScaleDotElemType::E5M2;
    };

    if (!supportsTypes(aElemType) || !supportsTypes(bElemType)) {
      llvm::report_fatal_error("NYI: mxfp6\n");
    }

    int64_t kDimOperandSize = aTensorTy.getShape().back();

    auto ctx = op.getContext();
    constexpr bool allowXF32 = false;
    FailureOr<MfmaIntrinsic> maybeMfmaIntrinsic =
        MfmaIntrinsic::get(op.getLoc(), mfmaVersion, mDim, nDim, kDim,
                           scaleDotElemTypeToMLIRType(ctx, aElemType),
                           scaleDotElemTypeToMLIRType(ctx, bElemType),
                           /*withScale=*/true, allowXF32);
    if (failed(maybeMfmaIntrinsic))
      return op.emitError(
          "no matching matrix core intrinsic due to unsupported element type");

    StringRef intrinsicName = maybeMfmaIntrinsic->name;
    unsigned kBase = maybeMfmaIntrinsic->kBase;
    // Two fp4 are packed into an uint8.
    unsigned aKBase = aElemType == ScaleDotElemType::E2M1 ? kBase / 2 : kBase;
    unsigned bKBase = bElemType == ScaleDotElemType::E2M1 ? kBase / 2 : kBase;

    int aKWidth = aKBase;
    int bKWidth = bKBase;

    const auto kDimInstrSize = mfmaLayout.getInstrShapeForOperand(aKBase, 0)[1];

    auto repA = mfmaLayout.getRepForOperand(aTensorTy.getShape(), aKWidth, 0);
    auto repB = mfmaLayout.getRepForOperand(bTensorTy.getShape(), bKWidth, 1);
    assert(repA[2] == repB[1]);

    // For fp4 scaled mfma, each thread takes 1 element from scale. Will have
    // better way to get it when adapting other data types. Similar to
    // scaleKBase
    constexpr int scaleKWidth = 1;
    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedAScale = adaptor.getAScale();
    Value loadedBScale = adaptor.getBScale();
    Value loadedC = adaptor.getC();

    auto numRepM = repA[1];
    auto numRepN = repB[2];
    auto numRepK = repA[2];
    auto numRepB = repA[0];
    assert(repA[0] == repB[0]);

    // Scaled MFMA instructions expect scale operands as 32-bit values,
    // even though each individual scale is only 8 bits. To reduce register
    // usage, we pack 4 scales into a single 32-bit value and use the opSel
    // field to select the appropriate byte during execution. Packing is done
    // along the K dimension first; if there aren’t enough values in K, we
    // continue along the non-K dimension.
    // TODO: Support opSel selection for constant scales stored in SGPRs.
    const int scaleAKBase =
        isAScaleConstant ? 1 : std::min(4, static_cast<int>(numRepK * numRepM));
    const int scaleBKBase =
        isBScaleConstant ? 1 : std::min(4, static_cast<int>(numRepK * numRepN));

    int akPackedVals =
        isAScaleConstant ? 1 : std::min(4, static_cast<int>(numRepK));
    int bkPackedVals =
        isBScaleConstant ? 1 : std::min(4, static_cast<int>(numRepK));

    assert(scaleAKBase % akPackedVals == 0 && scaleBKBase % bkPackedVals == 0);
    int aNonKPackedVals = scaleAKBase / akPackedVals;
    int bNonKPackedVals = scaleBKBase / bkPackedVals;

    auto operandA = getValuesFromDotOperandLayoutStruct(
        loadedA, numRepB, numRepM, numRepK, aKWidth, aKBase,
        aTensorTy.getElementType(), allowXF32, /*preserveBF16=*/false);
    auto operandB = getValuesFromDotOperandLayoutStruct(
        loadedB, numRepB, numRepN, numRepK, bKWidth, bKBase,
        bTensorTy.getElementType(), allowXF32, /*preserveBF16=*/false);

    // Scales have the same replica distributions as their corresponding
    // operands.
    ValueTable operandAScale;
    ValueTable operandBScale;
    if (existBothScales) {
      auto aScaleTensorTy = cast<RankedTensorType>(aScale.getType());
      operandAScale = getValuesFromDotOperandLayoutStruct(
          loadedAScale, numRepB, numRepM, numRepK, scaleKWidth, scaleAKBase,
          aScaleTensorTy.getElementType(), allowXF32, /*preserveBF16=*/false,
          isAScaleConstant);

      auto bScaleTensorTy = cast<RankedTensorType>(bScale.getType());
      operandBScale = getValuesFromDotOperandLayoutStruct(
          loadedBScale, numRepB, numRepN, numRepK, scaleKWidth, scaleBKBase,
          bScaleTensorTy.getElementType(), allowXF32, /*preserveBF16=*/false,
          isBScaleConstant);
    }

    auto dstElemTy = dTensorTy.getElementType();
    auto fc = unpackLLElements(loc, loadedC, rewriter);

    unsigned warpSize = triton::gpu::lookupThreadsPerWarp(rewriter);
    // compute number of output elements that each thread holds for one MFMA
    // instruction. subBlocks
    const int subBlocks =
        getNumSubmatrices(aTensorTy.getElementType(), mDim, nDim);
    auto elemsPerVec = mDim * nDim * subBlocks / warpSize;
    int numVecInKBase = numRepK * aKWidth / aKBase;

    Value firstMfma;
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    auto vecTy = vec_ty(dstElemTy, elemsPerVec);

    // 2-step pingpong got local_loads + dot_scaled in the dot cluster
    // from the first step in the transform pingpong pass.
    // Here, in the second step, it splits operations into two clusters
    // The first cluster has local_load with mfma from the first half of K
    // and the second cluster with the other half K of mfma.
    // By splitting in K dim, we can retire registers used by the
    // first half of mfma, backend compiler is supposed to schedule it.
    int halfPoint = numVecInKBase * numRepB * numRepM * numRepN / 2;
    int currIter = 0;
    bool is2Step = false;
    int innerK = 0, outerK = 0, innerKBound = 1, outerKBound = 1;
    // In order to split mfma by K, change the outermost loop iterates
    // over the K in emitting the mfma operations.
    if (auto pingpongUnitAttr = op->getAttr("pingpong_2step")) {
      is2Step = true;
      outerKBound = numVecInKBase;
    } else
      innerKBound = numVecInKBase;

    for (outerK = 0; outerK < outerKBound; outerK++) {
      for (int b = 0; b < numRepB; ++b) {
        for (int m = 0; m < numRepM; ++m) {
          for (int n = 0; n < numRepN; ++n) {
            // Insert pingpong cluster barrier when needed.
            if (is2Step && currIter++ == halfPoint) {
              rewriter.create<ROCDL::SchedBarrier>(loc, 0);
              rewriter.create<ROCDL::SBarrierOp>(loc);
              rewriter.create<ROCDL::SchedBarrier>(loc, 0);
            }
            Value acc = tb.undef(vecTy);
            for (unsigned v = 0; v < elemsPerVec; ++v) {
              acc = tb.insert_element(
                  vecTy, acc,
                  fc[b * numRepM * numRepN * elemsPerVec +
                     m * numRepN * elemsPerVec + n * elemsPerVec + v],
                  tb.i32_val(v));
            }
            acc = zeroAuxiliarBlocks(subBlocks, acc);
            for (innerK = 0; innerK < innerKBound; innerK++) {
              int k = is2Step ? outerK : innerK;
              if (existBothScales) {
                int akScale = k / akPackedVals;
                int bkScale = k / bkPackedVals;
                int opSelA = 0, opSelB = 0;

                int mScale = m / aNonKPackedVals;
                int nScale = n / bNonKPackedVals;
                opSelA = (m * numRepK + k) % (aNonKPackedVals * akPackedVals);
                opSelB = (n * numRepK + k) % (bNonKPackedVals * bkPackedVals);

                if (mfmaLayout.getIsTransposed()) {
                  acc = generateScaledMFMAOp(
                      intrinsicName, operandB[{b, n, k}], operandA[{b, m, k}],
                      acc, operandBScale[{b, nScale, bkScale}],
                      operandAScale[{b, mScale, akScale}],
                      maybeMfmaIntrinsic->bElementType,
                      maybeMfmaIntrinsic->aElementType, opSelB, opSelA);
                } else {
                  acc = generateScaledMFMAOp(
                      intrinsicName, operandA[{b, m, k}], operandB[{b, n, k}],
                      acc, operandAScale[{b, mScale, akScale}],
                      operandBScale[{b, nScale, bkScale}],
                      maybeMfmaIntrinsic->aElementType,
                      maybeMfmaIntrinsic->bElementType, opSelA, opSelB);
                }
              } else {
                if (mfmaLayout.getIsTransposed()) {
                  acc = generateScaledMFMAOp(intrinsicName, operandB[{b, n, k}],
                                             operandA[{b, m, k}], acc,
                                             maybeMfmaIntrinsic->bElementType,
                                             maybeMfmaIntrinsic->aElementType);
                } else {
                  acc = generateScaledMFMAOp(intrinsicName, operandA[{b, m, k}],
                                             operandB[{b, n, k}], acc,
                                             maybeMfmaIntrinsic->aElementType,
                                             maybeMfmaIntrinsic->bElementType);
                }
              }
              if (!firstMfma)
                firstMfma = acc;
            }
            acc = reduceSubBlocks(subBlocks, acc);
            adjustAccForSmallKDim(fc, acc, dstElemTy, b, m, n, numRepM, numRepN,
                                  kDimInstrSize, kDimOperandSize, elemsPerVec);
          }
        }
      }
    }

    // Originally, setprio (high) is set to the high-level dot op. After dot is
    // being lowered to the series of mfma operations, it should be moved next
    // to the first mfma leaving the first mfma staying at the low priority. In
    // this way, incoming warp can be effectively waiting on the first mfma
    // instruction (low priority) while the other warp is executing mfma with
    // high priority. Otherwise, incoming warp can break the cluster.
    if (setPrioOp && firstMfma)
      setPrioOp->moveAfter(firstMfma.getDefiningOp());

    const size_t mmaCount =
        numRepB * numRepM * numRepN * numRepK * aKWidth / aKBase;
    packAndReplaceResult(op, fc, maybeMfmaIntrinsic, dstElemTy, elemTyA,
                         mmaCount);

    return success();
  }
};

} // namespace

namespace mlir::triton::AMD {
LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return cast<RankedTensorType>(tensor.getType());
  };

  assert(isa<DotOperandEncodingAttr>(rankedTType(op.getA()).getEncoding()) &&
         isa<DotOperandEncodingAttr>(rankedTType(op.getB()).getEncoding()) &&
         "Both A and B should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(isa<AMDMfmaEncodingAttr>(cTensorTy.getEncoding()) &&
         "Currently, we only support C with a mfma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's C operand should pass the same number of values as D.");

  auto loc = op.getLoc();
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  DotOpMFMAConversionHelper helper(mfmaLayout, rewriter, typeConverter, loc);

  return helper.convertDot(op, adaptor);
}

LogicalResult convertScaledMFMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter) {
  assert(isa<DotOperandEncodingAttr>(op.getA().getType().getEncoding()) &&
         isa<DotOperandEncodingAttr>(op.getB().getType().getEncoding()) &&
         "Both lhs and rhs should be in DotOperand layout.");

  auto aScale = op.getAScale();
  auto bScale = op.getBScale();

  // If the tt.dot_scaled is transformed from a tt.dot, both scales are None. In
  // this case, both scales remain None in this method and we will generate a
  // mfma instruction with the scale operand to be 0. Then there's an
  // optimization pass in the LLVM backend to convert such V_MFMA_SCALE_*_F8F6F4
  // instruction to V_MFMA_*_F8F6F4 to avoid LD_SCALE.
  //
  // If the tt.dot_scaled is not from a tt.dot but native, we support 0, 1, 2
  // scales and treat them in different ways:
  //
  // 1. #scales = 0: Just like those transformed from tt.dot, both scales remain
  // None.
  // 2. #scales = 1: The upstream transform guarantees to create constant
  // scales for the absent.
  // 2. #scales = 2: Both scales should exist.

  // Thus in this pass, there shouldn't be a single scale present.
  assert(((aScale && bScale) || (!aScale && !bScale)) &&
         "Single scale is not supported");

  if (aScale && bScale) {
    assert(
        isa<LinearEncodingAttr>(aScale.getType().getEncoding()) &&
        isa<LinearEncodingAttr>(bScale.getType().getEncoding()) &&
        "If scales exist, both LhsScale and RhsScale should be linear layout.");
  }

  auto cTensorTy = op.getC().getType();
  auto dTensorTy = op.getD().getType();
  assert(isa<AMDMfmaEncodingAttr>(cTensorTy.getEncoding()) &&
         "Currently, we only support C with a mfma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's C operand should pass the same number of values as D.");

  auto loc = op.getLoc();
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  ScaledDotOpMFMAConversionHelper helper(mfmaLayout, rewriter, typeConverter,
                                         loc);

  return helper.convertScaledDot(op, adaptor);
}
} // namespace mlir::triton::AMD
