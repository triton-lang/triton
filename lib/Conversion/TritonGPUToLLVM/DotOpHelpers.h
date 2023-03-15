#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_DOT_OP_HELPERS_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_DOT_OP_HELPERS_H

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/PTXAsmFormat.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"

#include "Utility.h"

class TritonGPUToLLVMTypeConverter;

namespace mlir {
namespace LLVM {
using namespace mlir::triton;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

// Helper for conversion of DotOp with mma<version=1>, that is sm<80
struct DotOpMmaV1ConversionHelper {
  MmaEncodingAttr mmaLayout;
  ArrayRef<unsigned> wpt;
  static constexpr std::array<int, 3> fpw{{2, 2, 1}};

  using ValueTable = std::map<std::pair<int, int>, std::pair<Value, Value>>;

  explicit DotOpMmaV1ConversionHelper(MmaEncodingAttr mmaLayout)
      : mmaLayout(mmaLayout), wpt(mmaLayout.getWarpsPerCTA()) {}

  static ArrayRef<unsigned> getMmaInstrShape() { return instrShape; }

  static Type getMmaRetType(TensorType operand) {
    auto *ctx = operand.getContext();
    Type fp32Ty = type::f32Ty(ctx);
    // f16*f16+f32->f32
    return struct_ty(SmallVector<Type>{8, fp32Ty});
  }

  static Type getMatType(TensorType operand) {
    auto *ctx = operand.getContext();
    Type fp16Ty = type::f16Ty(ctx);
    Type vecTy = vec_ty(fp16Ty, 2);
    return struct_ty(SmallVector<Type>{vecTy});
  }

  // Loading $a from smem to registers, returns a LLVM::Struct.
  Value loadA(Value tensor, const SharedMemoryObject &smemObj, Value thread,
              Location loc, TritonGPUToLLVMTypeConverter *converter,
              ConversionPatternRewriter &rewriter, Type resultTy) const;

  // Loading $b from smem to registers, returns a LLVM::Struct.
  Value loadB(Value tensor, const SharedMemoryObject &smemObj, Value thread,
              Location loc, TritonGPUToLLVMTypeConverter *converter,
              ConversionPatternRewriter &rewriter, Type resultTy) const;

  static ArrayRef<unsigned> getOrder() { return mmaOrder; }

  // Compute the offset of the matrix to load.
  // Returns offsetAM, offsetAK, offsetBN, offsetBK.
  // NOTE, the information M(from $a) and N(from $b) couldn't be retrieved at
  // the same time in the usage in convert_layout[shared->dot_op], we leave
  // the noexist info to be 0 and only use the desired argument from the
  // composed result. In this way we want to retain the original code
  // structure in convert_mma884 method for easier debugging.
  std::tuple<Value, Value, Value, Value>
  computeOffsets(Value threadId, bool isARow, bool isBRow, ArrayRef<int> fpw,
                 ArrayRef<int> spw, ArrayRef<int> rep,
                 ConversionPatternRewriter &rewriter, Location loc) const;

  // Extract values belong to $a or $b from a LLVMStruct, the shape is n0xn1.
  DotOpMmaV1ConversionHelper::ValueTable extractLoadedOperand(
      Value llStruct, int NK, ConversionPatternRewriter &rewriter,
      TritonGPUToLLVMTypeConverter *typeConverter, Type type) const;

  using CoordTy = SmallVector<Value>;
  // Get the coordinates(m,n) of the elements emit by a thread in accumulator.
  static SmallVector<CoordTy>
  getMNCoords(Value thread, ConversionPatternRewriter &rewriter,
              ArrayRef<unsigned> wpt, const MmaEncodingAttr &mmaLayout,
              ArrayRef<int64_t> shape, bool isARow, bool isBRow, bool isAVec4,
              bool isBVec4);

  // \param elemId the offset of the element in a thread
  static CoordTy getCoord(int elemId, ArrayRef<CoordTy> coords) {
    return coords[elemId];
  }

private:
  static constexpr unsigned instrShape[] = {16, 16, 4};
  static constexpr unsigned mmaOrder[] = {0, 1};
};

// Helper for conversion of DotOp with mma<version=2>, that is sm>=80
struct DotOpMmaV2ConversionHelper {
  enum class TensorCoreType : uint8_t {
    // floating-point tensor core instr
    FP32_FP16_FP16_FP32 = 0, // default
    FP32_BF16_BF16_FP32,
    FP32_TF32_TF32_FP32,
    FP16_FP16_FP16_FP16,
    // integer tensor core instr
    INT32_INT1_INT1_INT32, // Not implemented
    INT32_INT4_INT4_INT32, // Not implemented
    INT32_INT8_INT8_INT32, // Not implemented
    //
    NOT_APPLICABLE,
  };

  MmaEncodingAttr mmaLayout;
  MLIRContext *ctx{};

  explicit DotOpMmaV2ConversionHelper(MmaEncodingAttr mmaLayout)
      : mmaLayout(mmaLayout) {
    ctx = mmaLayout.getContext();
  }

  void deduceMmaType(DotOp op) const { mmaType = getMmaType(op); }
  void deduceMmaType(Type operandTy) const {
    mmaType = getTensorCoreTypeFromOperand(operandTy);
  }

  // Get the M and N of mma instruction shape.
  static std::tuple<int, int> getInstrShapeMN() {
    // According to DotOpConversionHelper::mmaInstrShape, all the M,N are
    // {16,8}
    return {16, 8};
  }

  static std::tuple<int, int> getRepMN(const RankedTensorType &tensorTy);

  Type getShemPtrTy() const;

  // The type of matrix that loaded by either a ldmatrix or composed lds.
  Type getMatType() const;

  Type getLoadElemTy();

  Type getMmaRetType() const;

  int getMmaRetSize() const;

  ArrayRef<int> getMmaInstrShape() const {
    assert(mmaType != TensorCoreType::NOT_APPLICABLE &&
           "Unknown mma type found.");
    return mmaInstrShape.at(mmaType);
  }

  static ArrayRef<int> getMmaInstrShape(TensorCoreType tensorCoreType) {
    assert(tensorCoreType != TensorCoreType::NOT_APPLICABLE &&
           "Unknown mma type found.");
    return mmaInstrShape.at(tensorCoreType);
  }

  ArrayRef<int> getMmaMatShape() const {
    assert(mmaType != TensorCoreType::NOT_APPLICABLE &&
           "Unknown mma type found.");
    return mmaMatShape.at(mmaType);
  }

  // Deduce the TensorCoreType from either $a or $b's type.
  // TODO: both the input type ($a or $b) and output type ($c or $d) are
  // needed to differentiate TensorCoreType::FP32_FP16_FP16_FP32 from
  // TensorCoreType::FP16_FP16_FP16_FP16
  static TensorCoreType getTensorCoreTypeFromOperand(Type operandTy);

  int getVec() const {
    assert(mmaType != TensorCoreType::NOT_APPLICABLE &&
           "Unknown mma type found.");
    return mmaInstrVec.at(mmaType);
  }

  StringRef getMmaInstr() const {
    assert(mmaType != TensorCoreType::NOT_APPLICABLE &&
           "Unknown mma type found.");
    return mmaInstrPtx.at(mmaType);
  }

  static TensorCoreType getMmaType(triton::DotOp op);

private:
  mutable TensorCoreType mmaType{TensorCoreType::NOT_APPLICABLE};

  // Used on nvidia GPUs mma layout .version == 2
  // Refer to
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-storage
  // for more details.
  inline static const std::map<TensorCoreType, llvm::SmallVector<int>>
      mmaInstrShape = {
          {TensorCoreType::FP32_FP16_FP16_FP32, {16, 8, 16}},
          {TensorCoreType::FP32_BF16_BF16_FP32, {16, 8, 16}},
          {TensorCoreType::FP32_TF32_TF32_FP32, {16, 8, 8}},
          {TensorCoreType::FP16_FP16_FP16_FP16, {16, 8, 16}},

          {TensorCoreType::INT32_INT1_INT1_INT32, {16, 8, 256}},
          {TensorCoreType::INT32_INT4_INT4_INT32, {16, 8, 64}},
          {TensorCoreType::INT32_INT8_INT8_INT32, {16, 8, 32}},
  };

  // shape of matrices loaded by ldmatrix (m-n-k, for mxk & kxn matrices)
  // Refer to
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix
  // for more details.
  inline static const std::map<TensorCoreType, llvm::SmallVector<int>>
      mmaMatShape = {
          {TensorCoreType::FP32_FP16_FP16_FP32, {8, 8, 8}},
          {TensorCoreType::FP32_BF16_BF16_FP32, {8, 8, 8}},
          {TensorCoreType::FP32_TF32_TF32_FP32, {8, 8, 4}},
          {TensorCoreType::FP16_FP16_FP16_FP16, {8, 8, 8}},

          {TensorCoreType::INT32_INT1_INT1_INT32, {8, 8, 64}},
          {TensorCoreType::INT32_INT4_INT4_INT32, {8, 8, 32}},
          {TensorCoreType::INT32_INT8_INT8_INT32, {8, 8, 16}},
  };

  // Supported mma instruction in PTX.
  // Refer to
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma
  // for more details.
  inline static const std::map<TensorCoreType, std::string> mmaInstrPtx = {
      {TensorCoreType::FP32_FP16_FP16_FP32,
       "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"},
      {TensorCoreType::FP32_BF16_BF16_FP32,
       "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"},
      {TensorCoreType::FP32_TF32_TF32_FP32,
       "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"},
      {TensorCoreType::FP16_FP16_FP16_FP16,
       "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"},

      {TensorCoreType::INT32_INT1_INT1_INT32,
       "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc"},
      {TensorCoreType::INT32_INT4_INT4_INT32,
       "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32"},
      {TensorCoreType::INT32_INT8_INT8_INT32,
       "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32"},
  };

  // vector length per ldmatrix (16*8/element_size_in_bits)
  inline static const std::map<TensorCoreType, uint8_t> mmaInstrVec = {
      {TensorCoreType::FP32_FP16_FP16_FP32, 8},
      {TensorCoreType::FP32_BF16_BF16_FP32, 8},
      {TensorCoreType::FP32_TF32_TF32_FP32, 4},
      {TensorCoreType::FP16_FP16_FP16_FP16, 8},

      {TensorCoreType::INT32_INT1_INT1_INT32, 128},
      {TensorCoreType::INT32_INT4_INT4_INT32, 32},
      {TensorCoreType::INT32_INT8_INT8_INT32, 16},
  };
};

// Data loader for mma.16816 instruction.
class MMA16816SmemLoader {
public:
  MMA16816SmemLoader(int wpt, ArrayRef<uint32_t> order, uint32_t kOrder,
                     ArrayRef<Value> smemStrides, ArrayRef<int64_t> tileShape,
                     ArrayRef<int> instrShape, ArrayRef<int> matShape,
                     int perPhase, int maxPhase, int elemBytes,
                     ConversionPatternRewriter &rewriter,
                     TypeConverter *typeConverter, const Location &loc);

  // lane = thread % 32
  // warpOff = (thread/32) % wpt(0)
  llvm::SmallVector<Value> computeOffsets(Value warpOff, Value lane,
                                          Value cSwizzleOffset) {
    if (canUseLdmatrix)
      return computeLdmatrixMatOffs(warpOff, lane, cSwizzleOffset);
    else if (elemBytes == 4 && needTrans)
      return computeB32MatOffs(warpOff, lane, cSwizzleOffset);
    else if (elemBytes == 1 && needTrans)
      return computeB8MatOffs(warpOff, lane, cSwizzleOffset);
    else
      llvm::report_fatal_error("Invalid smem load config");

    return {};
  }

  int getNumPtrs() const { return numPtrs; }

  // Compute the offset to the matrix this thread(indexed by warpOff and lane)
  // mapped to.
  SmallVector<Value> computeLdmatrixMatOffs(Value warpId, Value lane,
                                            Value cSwizzleOffset);

  // Compute 32-bit matrix offsets.
  SmallVector<Value> computeB32MatOffs(Value warpOff, Value lane,
                                       Value cSwizzleOffset);

  // compute 8-bit matrix offset.
  SmallVector<Value> computeB8MatOffs(Value warpOff, Value lane,
                                      Value cSwizzleOffset);

  // Load 4 matrices and returns 4 vec<2> elements.
  std::tuple<Value, Value, Value, Value>
  loadX4(int mat0, int mat1, ArrayRef<Value> offs, ArrayRef<Value> ptrs,
         Type matTy, Type shemPtrTy) const;

private:
  SmallVector<uint32_t> order;
  int kOrder;
  SmallVector<int64_t> tileShape;
  SmallVector<int> instrShape;
  SmallVector<int> matShape;
  int perPhase;
  int maxPhase;
  int elemBytes;
  ConversionPatternRewriter &rewriter;
  const Location &loc;
  MLIRContext *ctx{};

  int cMatShape;
  int sMatShape;

  Value sStride;

  bool needTrans;
  bool canUseLdmatrix;

  int numPtrs;

  int pLoadStrideInMat;
  int sMatStride;

  int matArrStride;
  int warpOffStride;
};

// This class helps to adapt the existing DotOpConversion to the latest
// DotOpOperand layout design. It decouples the existing implementation to two
// parts:
// 1. loading the specific operand matrix(for $a, $b, $c) from smem
// 2. passing the loaded value and perform the mma codegen
struct MMA16816ConversionHelper {
  MmaEncodingAttr mmaLayout;
  ArrayRef<unsigned int> wpt;
  SmallVector<unsigned int> properWpt;

  Value thread, lane, warp;

  DotOpMmaV2ConversionHelper helper;
  ConversionPatternRewriter &rewriter;
  TritonGPUToLLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

  // dotOperand: type of either one operand of dotOp.
  MMA16816ConversionHelper(Type dotOperand, MmaEncodingAttr mmaLayout,
                           Value thread, ConversionPatternRewriter &rewriter,
                           TritonGPUToLLVMTypeConverter *typeConverter,
                           Location loc)
      : mmaLayout(mmaLayout), wpt(mmaLayout.getWarpsPerCTA()), thread(thread),
        helper(mmaLayout), rewriter(rewriter), typeConverter(typeConverter),
        loc(loc), ctx(mmaLayout.getContext()) {
    helper.deduceMmaType(dotOperand);

    Value _32 = i32_val(32);
    lane = urem(thread, _32);
    warp = udiv(thread, _32);
  }

  MMA16816ConversionHelper(DotOp op, MmaEncodingAttr mmaLayout, Value thread,
                           ConversionPatternRewriter &rewriter,
                           TritonGPUToLLVMTypeConverter *typeConverter,
                           Location loc)
      : mmaLayout(mmaLayout), wpt(mmaLayout.getWarpsPerCTA()), thread(thread),
        helper(mmaLayout), rewriter(rewriter), typeConverter(typeConverter),
        loc(loc), ctx(mmaLayout.getContext()) {
    helper.deduceMmaType(op);

    Value _32 = i32_val(32);
    lane = urem(thread, _32);
    warp = udiv(thread, _32);
  }

  // Get a warpId for M axis.
  Value getWarpM(int M) const {
    auto matInstrShape = helper.getMmaInstrShape();
    return urem(urem(warp, i32_val(wpt[0])), i32_val(M / matInstrShape[0]));
  }

  // Get a warpId for N axis.
  Value getWarpN(int N) const {
    auto matInstrShape = helper.getMmaInstrShape();
    Value warpMN = udiv(warp, i32_val(wpt[0]));
    return urem(urem(warpMN, i32_val(wpt[1])), i32_val(N / matInstrShape[1]));
  }

  // Get the mmaInstrShape deducing either from $a or $b.
  std::tuple<int, int, int> getMmaInstrShape(Type operand) const {
    helper.deduceMmaType(operand);
    auto mmaInstrShape = helper.getMmaInstrShape();
    int mmaInstrM = mmaInstrShape[0];
    int mmaInstrN = mmaInstrShape[1];
    int mmaInstrK = mmaInstrShape[2];
    return std::make_tuple(mmaInstrM, mmaInstrN, mmaInstrK);
  }

  // Get the mmaMatShape deducing either from $a or $b.
  std::tuple<int, int, int> getMmaMatShape(Type operand) const {
    helper.deduceMmaType(operand);
    auto matShape = helper.getMmaMatShape();
    int matShapeM = matShape[0];
    int matShapeN = matShape[1];
    int matShapeK = matShape[2];
    return std::make_tuple(matShapeM, matShapeN, matShapeK);
  }

  // \param operand is either $a or $b's type.
  inline int getNumRepM(Type operand, int M) const {
    return getNumRepM(operand, M, wpt[0]);
  }

  // \param operand is either $a or $b's type.
  inline int getNumRepN(Type operand, int N) const {
    return getNumRepN(operand, N, wpt[1]);
  }

  // \param operand is either $a or $b's type.
  inline int getNumRepK(Type operand, int K) const {
    return getNumRepK_(operand, K);
  }

  static int getNumRepM(Type operand, int M, int wpt) {
    auto tensorCoreType =
        DotOpMmaV2ConversionHelper::getTensorCoreTypeFromOperand(operand);
    int mmaInstrM =
        DotOpMmaV2ConversionHelper::getMmaInstrShape(tensorCoreType)[0];
    return std::max<int>(M / (wpt * mmaInstrM), 1);
  }

  static int getNumRepN(Type operand, int N, int wpt) {
    auto tensorCoreType =
        DotOpMmaV2ConversionHelper::getTensorCoreTypeFromOperand(operand);
    int mmaInstrN =
        DotOpMmaV2ConversionHelper::getMmaInstrShape(tensorCoreType)[1];
    return std::max<int>(N / (wpt * mmaInstrN), 1);
  }

  static int getNumRepK_(Type operand, int K) {
    auto tensorCoreType =
        DotOpMmaV2ConversionHelper::getTensorCoreTypeFromOperand(operand);
    int mmaInstrK =
        DotOpMmaV2ConversionHelper::getMmaInstrShape(tensorCoreType)[2];
    return std::max<int>(K / mmaInstrK, 1);
  }

  // Get number of elements per thread for $a operand.
  static size_t getANumElemsPerThread(RankedTensorType operand, int wpt) {
    auto shape = operand.getShape();
    int repM = getNumRepM(operand, shape[0], wpt);
    int repK = getNumRepK_(operand, shape[1]);
    return 4 * repM * repK;
  }

  // Get number of elements per thread for $b operand.
  static size_t getBNumElemsPerThread(RankedTensorType operand, int wpt) {
    auto shape = operand.getShape();
    int repK = getNumRepK_(operand, shape[0]);
    int repN = getNumRepN(operand, shape[1], wpt);
    return 4 * std::max(repN / 2, 1) * repK;
  }

  // Loading $a from smem to registers, returns a LLVM::Struct.
  Value loadA(Value tensor, const SharedMemoryObject &smemObj) const;

  // Loading $b from smem to registers, returns a LLVM::Struct.
  Value loadB(Value tensor, const SharedMemoryObject &smemObj);

  // Loading $c to registers, returns a Value.
  Value loadC(Value tensor, Value llTensor) const;

  // Conduct the Dot conversion.
  // \param a, \param b, \param c and \param d are DotOp operands.
  // \param loadedA, \param loadedB, \param loadedC, all of them are result of
  // loading.
  LogicalResult convertDot(Value a, Value b, Value c, Value d, Value loadedA,
                           Value loadedB, Value loadedC, DotOp op,
                           DotOpAdaptor adaptor) const;

private:
  std::function<void(int, int)>
  getLoadMatrixFn(Value tensor, const SharedMemoryObject &smemObj,
                  MmaEncodingAttr mmaLayout, int wpt, uint32_t kOrder,
                  SmallVector<int> instrShape, SmallVector<int> matShape,
                  Value warpId, ValueTable &vals, bool isA) const;

  // Compose a map of Values to a LLVM::Struct.
  // The layout is a list of Value with coordinate of (i,j), the order is as
  // the follows:
  // [
  //  (0,0), (0,1), (1,0), (1,1), # i=0, j=0
  //  (0,2), (0,3), (1,2), (1,3), # i=0, j=1
  //  (0,4), (0,5), (1,4), (1,5), # i=0, j=2
  //  ...
  //  (2,0), (2,1), (3,0), (3,1), # i=1, j=0
  //  (2,2), (2,3), (3,2), (3,3), # i=1, j=1
  //  (2,4), (2,5), (3,4), (3,5), # i=1, j=2
  //  ...
  // ]
  // i \in [0, n0) and j \in [0, n1)
  // There should be \param n0 * \param n1 elements in the output Struct.
  Value composeValuesToDotOperandLayoutStruct(const ValueTable &vals, int n0,
                                              int n1) const;

  ValueTable getValuesFromDotOperandLayoutStruct(Value value, int n0, int n1,
                                                 Type type) const;
};

// Helper for conversion of FMA DotOp.
struct DotOpFMAConversionHelper {
  Attribute layout;
  MLIRContext *ctx{};

  using ValueTable = std::map<std::pair<int, int>, Value>;

  explicit DotOpFMAConversionHelper(Attribute layout)
      : layout(layout), ctx(layout.getContext()) {}

  SmallVector<Value>
  getThreadIds(Value threadId, ArrayRef<unsigned> shapePerCTA,
               ArrayRef<unsigned> sizePerThread, ArrayRef<unsigned> order,
               ConversionPatternRewriter &rewriter, Location loc) const;

  Value loadA(Value A, Value llA, BlockedEncodingAttr dLayout, Value thread,
              Location loc, TritonGPUToLLVMTypeConverter *typeConverter,
              ConversionPatternRewriter &rewriter) const;

  Value loadB(Value B, Value llB, BlockedEncodingAttr dLayout, Value thread,
              Location loc, TritonGPUToLLVMTypeConverter *typeConverter,
              ConversionPatternRewriter &rewriter) const;

  ValueTable getValueTableFromStruct(
      Value val, int K, int n0, int shapePerCTA, int sizePerThread,
      ConversionPatternRewriter &rewriter, Location loc,
      TritonGPUToLLVMTypeConverter *typeConverter, Type type) const;

  Value getStructFromValueTable(ArrayRef<Value> vals,
                                ConversionPatternRewriter &rewriter,
                                Location loc,
                                TritonGPUToLLVMTypeConverter *typeConverter,
                                Type elemTy) const;

  // get number of elements per thread for $a or $b.
  static int getNumElemsPerThread(ArrayRef<int64_t> shape,
                                  DotOperandEncodingAttr dotOpLayout);

  // Get shapePerCTA for M or N axis.
  static int getShapePerCTAForMN(BlockedEncodingAttr layout, bool isM) {
    auto order = layout.getOrder();
    auto shapePerCTA = getShapePerCTA(layout);

    int mShapePerCTA =
        order[0] == 1 ? shapePerCTA[order[1]] : shapePerCTA[order[0]];
    int nShapePerCTA =
        order[0] == 0 ? shapePerCTA[order[1]] : shapePerCTA[order[0]];
    return isM ? mShapePerCTA : nShapePerCTA;
  }

  // Get sizePerThread for M or N axis.
  static int getSizePerThreadForMN(BlockedEncodingAttr layout, bool isM) {
    auto order = layout.getOrder();
    auto sizePerThread = getSizePerThread(layout);

    int mSizePerThread =
        order[0] == 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
    int nSizePerThread =
        order[0] == 0 ? sizePerThread[order[1]] : sizePerThread[order[0]];
    return isM ? mSizePerThread : nSizePerThread;
  }
};

} // namespace LLVM
} // namespace mlir

#endif
