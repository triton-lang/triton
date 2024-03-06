#include "nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"

using mlir::LLVM::getWrappedMultiDimOffset;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
namespace {
Value computeStMatrixAddr(Value laneId, int matStride, Location loc,
                          ConversionPatternRewriter &rewriter) {
  Value rowInMat = urem(laneId, i32_val(8)); // row in the 8x8 matrix
  // linear index of the matrix in the 2x2 matrices
  // Decompose matIndex => s_0, s_1, that is the coordinate in 2x2 matrices in
  // a warp.
  Value matIndex = udiv(laneId, i32_val(8));
  Value s0 = urem(matIndex, i32_val(2));
  Value s1 = udiv(matIndex, i32_val(2));
  Value mIndex = add(rowInMat, mul(s0, i32_val(8)));
  int m8n8Stride = 8;
  Value offset =
      add(mul(mIndex, i32_val(matStride)), mul(s1, i32_val(m8n8Stride)));
  return offset;
}

void stMatrixm8n8x4(Value offset, ArrayRef<Value> vals, int indexOffset,
                    Value smemBase, Type elemTy, Location loc,
                    ConversionPatternRewriter &rewriter) {
  SmallVector<Value> inputs;
  auto prTy = ptr_ty(rewriter.getContext(), 3);
  // Pack the input into 2xf16
  Type packedTy = vec_ty(vals[0].getType(), 2);
  for (int i = 0; i < 4; i++) {
    Value input = undef(packedTy);
    for (int j = 0; j < 2; j++) {
      input = insert_element(packedTy, input, vals[indexOffset + i * 2 + j],
                             i32_val(j));
    }
    inputs.push_back(bitcast(input, i32_ty));
  }
  Value addr = gep(smemBase.getType(), elemTy, smemBase, offset);
  rewriter.create<triton::nvgpu::StoreMatrixOp>(loc, addr, inputs);
}
} // namespace

namespace mlir {
namespace LLVM {
namespace NVIDIA {
using namespace mlir::triton;

Value shuffleCommon(Location loc, ConversionPatternRewriter &rewriter,
                    Value val, Value i, NVVM::ShflKind mode, Value clamp) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(val, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = shuffleCommon(loc, rewriter, val0, i, mode, clamp);
    val1 = shuffleCommon(loc, rewriter, val1, i, mode, clamp);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, val.getType());
  }
  Type type = val.getType();
  if (type != i32_ty) {
    val = bitcast(val, int_ty(bits));
    if (bits < 32)
      val = zext(i32_ty, val);
  }
  Value mask = i32_val(0xFFFFFFFF);
  Value result = rewriter.create<NVVM::ShflOp>(loc, i32_ty, mask, val, i, clamp,
                                               mode, UnitAttr());
  if (type != i32_ty) {
    if (bits < 32)
      result = trunc(int_ty(bits), result);
    result = bitcast(result, type);
  }
  return result;
}

Value shuffleXor(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), NVVM::ShflKind::bfly,
                       i32_val(0x1f));
}

Value shuffleUp(Location loc, ConversionPatternRewriter &rewriter, Value val,
                int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), NVVM::ShflKind::up,
                       i32_val(0x0));
}

Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i) {
  return shuffleIdx(loc, rewriter, val, i32_val(i));
}

Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 Value i) {
  return shuffleCommon(loc, rewriter, val, i, NVVM::ShflKind::idx,
                       i32_val(0x1f));
}

Value llGetPid(int axis, Location loc, ModuleOp moduleOp,
               ConversionPatternRewriter &rewriter) {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);

  // It is not easy to get the compute capability here, so we use numCTAs to
  // decide the semantic of GetProgramIdOp. If numCTAs = 1, then
  // GetProgramIdOp is converted to "%ctaid", otherwise it is converted to
  // "%clusterid".
  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

  std::string sreg = numCTAs == 1 ? "%ctaid." : "%clusterid.";
  sreg.append(1, 'x' + axis); // 0 -> 'x', 1 -> 'y', 2 -> 'z'
  return getSRegValue(rewriter, loc, sreg);
}

void storeDistributedToSharedWithStMatrix(
    RankedTensorType tensorTy, Type elemTy, SmallVector<Value> &inVals,
    Value smemBase, ArrayRef<unsigned> paddedRepShape,
    ArrayRef<unsigned> origRepShape, Location loc,
    ConversionPatternRewriter &rewriter) {
  auto shapePerCTA = getShapePerCTA(tensorTy);
  auto mmaLayout = tensorTy.getEncoding().cast<NvidiaMmaEncodingAttr>();
  auto order = triton::gpu::getOrder(mmaLayout);
  auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
  auto shapePerCTATile = getShapePerCTATile(mmaLayout);
  ArrayRef<unsigned> mmaShape = mmaLayout.getInstrShape();
  // 4xm8n8 matches exactly the size of 1 warp of wgmma layout for 16bit type
  // and has a shape of 16x16.
  int instrN = mmaShape[1] * warpsPerCTA[1];
  int instrM = mmaShape[0] * warpsPerCTA[0];
  std::array<int, 2> numRep = {ceil((int)origRepShape[0], instrM),
                               ceil((int)origRepShape[1], instrN)};

  Value thread = getThreadId(rewriter, loc);
  Value warp = udiv(thread, i32_val(32));
  Value lane = urem(thread, i32_val(32));

  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warp, warpsPerCTA);

  // Compute the relative offset for each lane.
  Value stMatrixLaneOffset =
      computeStMatrixAddr(lane, paddedRepShape[1], loc, rewriter);
  multiDimWarpId[0] = mul(multiDimWarpId[0], i32_val(mmaShape[0]));
  multiDimWarpId[1] = mul(multiDimWarpId[1], i32_val(mmaShape[1]));
  SmallVector<Value> multiDimOffsetWrapped =
      getWrappedMultiDimOffset(rewriter, loc, multiDimWarpId, origRepShape,
                               shapePerCTATile, shapePerCTA);
  Value relativeOffset =
      linearize(rewriter, loc, multiDimOffsetWrapped, paddedRepShape, order);
  relativeOffset = add(relativeOffset, stMatrixLaneOffset);
  int indexOffset = 0;
  int m8n8x4Stride = 16;
  int numNChunk = mmaShape[1] / m8n8x4Stride;
  for (int m = 0; m < numRep[0]; m++) {
    for (int n = 0; n < numRep[1]; n++) {
      for (int k = 0; k < numNChunk; k++) {
        Value addr =
            add(relativeOffset, i32_val(k * m8n8x4Stride + n * instrN +
                                        m * instrM * paddedRepShape[1]));
        stMatrixm8n8x4(addr, inVals, indexOffset, smemBase, elemTy, loc,
                       rewriter);
        indexOffset += 8;
      }
    }
  }
}

bool isStMatrixCompatible(RankedTensorType tensorTy) {
  auto mmaLayout = tensorTy.getEncoding().dyn_cast<NvidiaMmaEncodingAttr>();
  if (!mmaLayout || !mmaLayout.isHopper())
    return false;
  if (tensorTy.getElementType().getIntOrFloatBitWidth() != 16)
    return false;
  return true;
}

Value getSRegValue(OpBuilder &b, Location loc, const std::string &sRegStr) {
  PTXBuilder builder;
  auto &mov = builder.create("mov")->o("u32");
  auto *destOpr = builder.newOperand("=r");
  auto *sRegOpr = builder.newConstantOperand(sRegStr);
  mov(destOpr, sRegOpr);
  Value val = builder.launch(b, loc, b.getIntegerType(32), false);
  return val;
}
} // namespace NVIDIA
} // namespace LLVM
} // namespace mlir
