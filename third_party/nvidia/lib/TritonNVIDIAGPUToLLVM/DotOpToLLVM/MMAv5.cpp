#include "Dialect/NVGPU/IR/Dialect.h"
#include "MMAHelpers.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/NvmmaSmemAttrs.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::NVIDIA;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

// Helper class to load tensor memory following MMAv5 layout.
class DotOpMmaV5TmemLoader : public DotOpMmaMemLoader {
public:
  static DotOpMmaV5TmemLoader build(Location loc, RewriterBase &rewriter,
                                    mlir::triton::gpu::MemDescType memTy,
                                    Value tmemBase,
                                    unsigned logicalElementBitWidth) {
    // We take the full layout even when it is a subview
    // We'll just iterate the real shape when calling tmemLoad tho
    unsigned storageElementBitWidth = memTy.getElementTypeBitWidth();
    assert(logicalElementBitWidth > 0 &&
           storageElementBitWidth % logicalElementBitWidth == 0 &&
           "logical element bit width must divide TMEM storage bit width");
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    Value address = tb.ptrtoint(i32_ty, tmemBase);

    auto llInv = toLinearLayout(memTy).pseudoinvert();
    return DotOpMmaV5TmemLoader(llInv, address, storageElementBitWidth,
                                logicalElementBitWidth);
  }

  MemDescOperand tmemLoad(int a, int b, ConversionPatternRewriter &rewriter,
                          Location loc) const {
    // MMAv5 supplies a logical K coordinate, while byte-backed FP4 memdescs
    // use packed K coordinates. This applies to both dense and fp4Padded FP4;
    // the memdesc layout handles any additional physical padding.
    unsigned packingFactor = storageElementBitWidth / logicalElementBitWidth;
    assert(b % packingFactor == 0 &&
           "logical K coordinate must be aligned to packed storage");
    b /= packingFactor;
    auto dims = to_vector(ll.getInDimNames());
    auto rowCol = ll.apply({{dims[0], a}, {dims[1], b}});
    int row = rowCol[0].second;
    int col = rowCol[1].second * storageElementBitWidth / 32;
    int offset = col | (row << 16);
    return {address, offset};
  }

  MemDescOperand memLoad(int a, int b, ConversionPatternRewriter &rewriter,
                         Location loc) const override {
    return tmemLoad(a, b, rewriter, loc);
  }

private:
  DotOpMmaV5TmemLoader(LinearLayout ll, Value address,
                       int storageElementBitWidth, int logicalElementBitWidth)
      : ll(std::move(ll)), address(address),
        storageElementBitWidth(storageElementBitWidth),
        logicalElementBitWidth(logicalElementBitWidth) {}

  LinearLayout ll;
  Value address;
  int storageElementBitWidth;
  int logicalElementBitWidth;
};

//===----------------------------------------------------------------------===//
// InstDescriptor
//===----------------------------------------------------------------------===//

enum class mxfpKind { mxf8f6f4 = 0, mxf4 = 1, mxf4nvf4 = 2 };
enum class scaleKind : uint32_t { ue4m3 = 0, e8m0 = 1, ue5m3 = 2 };

bool isTransposed(Value operand) {
  auto tensorTy = cast<MemDescType>(operand.getType());
  if (isa<ttng::TensorMemoryEncodingAttr>(tensorTy.getEncoding()))
    return false;

  auto attrs = ttng::getNvmmaSmemAttrs(tensorTy);
  assert(attrs && "expected MMAv5 shared operand to have NVMMA SMEM attrs");
  return attrs->transposed;
}

static int getScaleFactor(Type scaleType, int blockK) {
  auto shapedType = cast<ShapedType>(scaleType);
  int64_t scaleCols = shapedType.getShape().back();
  assert(blockK % scaleCols == 0);
  return blockK / scaleCols;
}

static int getScaleVecSize(ttng::TCGen5MMAScaledOp op) {
  return getScaleFactor(op.getAScale().getType(), op.getBlockK());
}

static bool isBlock16Scale(Type scaleType, int blockK) {
  auto shapedType = dyn_cast<ShapedType>(scaleType);
  if (!shapedType || !shapedType.hasRank())
    return false;
  return shapedType.getShape().back() * 16 == blockK;
}

inline mxfpKind getMXFPKind(ScaleDotElemType typeA, ScaleDotElemType typeB,
                            Type scaleAType, Type scaleBType, int blockK,
                            bool transpose) {
  if (typeA == ScaleDotElemType::E2M1 && typeB == ScaleDotElemType::E2M1) {
    auto scaleAElemType = cast<ShapedType>(scaleAType).getElementType();
    auto scaleBElemType = cast<ShapedType>(scaleBType).getElementType();
    bool isUE4M3 = llvm::isa<Float8E4M3FNType>(scaleAElemType) &&
                   llvm::isa<Float8E4M3FNType>(scaleBElemType);
    bool isUE5M3 = scaleAElemType.isInteger(8) && scaleBElemType.isInteger(8) &&
                   isBlock16Scale(scaleAType, blockK) &&
                   isBlock16Scale(scaleBType, blockK);
    if (isUE4M3 || isUE5M3) {
      assert(!transpose &&
             "MMAv5 with kind=mxf4nvf4 does not support transpose");
      return mxfpKind::mxf4nvf4;
    }
    if (!transpose)
      return mxfpKind::mxf4;
  }
  return mxfpKind::mxf8f6f4;
};

static scaleKind getScaleKind(ttng::TCGen5MMAScaledOp op, int blockK) {
  Type scaleType = op.getAScale().getType();
  Type elemType = cast<ShapedType>(scaleType).getElementType();
  if (llvm::isa<Float8E4M3FNType>(elemType))
    return scaleKind::ue4m3;
  if (elemType.isInteger(8) && isBlock16Scale(scaleType, blockK))
    return scaleKind::ue5m3;
  return scaleKind::e8m0;
}

static Value createInstDescriptor(ConversionPatternRewriter &rewriter,
                                  ttng::TCGen5MMAOp op, int M, int N,
                                  bool transposeA, bool transposeB, int kSize) {
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  union TCGen5InstructionDescriptor {
    uint32_t descriptor;
    struct {
      uint32_t sparsitySelector : 2;
      uint32_t sparsity : 1;
      uint32_t : 1;
      uint32_t dType : 2;
      uint32_t : 1;
      uint32_t aType : 3;
      uint32_t bType : 3;
      uint32_t negateA : 1;
      uint32_t negateB : 1;
      uint32_t transposeA : 1;
      uint32_t transposeB : 1;
      uint32_t N : 6;
      uint32_t : 1;
      uint32_t M : 5;
      uint32_t kSize : 1;
      uint32_t shift : 2;
    };
  };
  auto getTypeEncoding = [&](Type type) {
    if (type.isF16())
      return 0;
    if (type.isBF16())
      return 1;
    if (type.isF32())
      return 2;
    if (llvm::isa<Float8E4M3FNType>(type))
      return 0;
    if (llvm::isa<Float8E5M2Type>(type))
      return 1;
    // For 8-bit integer types, signed arithmetic is 1, unsigned arithmetic is
    // 0.
    if (type.isInteger(8))
      return op.getIsUnsigned() ? 0 : 1;
    llvm_unreachable("Unsupported type.");
  };
  static_assert(sizeof(TCGen5InstructionDescriptor) == 4,
                "instruction descriptor size should be 32 bits.");
  TCGen5InstructionDescriptor desc;
  desc.descriptor = 0;
  desc.transposeA = transposeA;
  desc.transposeB = transposeB;
  desc.M = M >> 4;
  desc.N = N >> 3;
  desc.aType = getTypeEncoding(op.getA().getType().getElementType());
  desc.bType = getTypeEncoding(op.getB().getType().getElementType());
  Type dstElType = op.getD().getType().getElementType();
  assert(dstElType.isF16() || dstElType.isF32() || dstElType.isInteger(32));
  if (dstElType.isInteger(32)) {
    desc.dType = 2;
  } else {
    desc.dType = dstElType.isF16() ? 0 : 1;
  }
  if (kSize == 64)
    desc.kSize = 1;

  return b.int_val(32, desc.descriptor);
}

static Value createScaleInstDescriptorFp8(ConversionPatternRewriter &rewriter,
                                          ttng::TCGen5MMAScaledOp op, int M,
                                          int N, bool transposeA,
                                          bool transposeB,
                                          int scaleFactorsubIdxA,
                                          int scaleFactorsubIdxB, int kSize) {
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  union TCGen5InstructionDescriptor {
    uint32_t descriptor;
    struct {
      uint32_t sparsitySelector : 2;
      uint32_t sparsity : 1;
      uint32_t : 1;
      uint32_t BScaleFactor : 2;
      uint32_t : 1;
      uint32_t aType : 3;
      uint32_t bType : 3;
      uint32_t negateA : 1;
      uint32_t negateB : 1;
      uint32_t transposeA : 1;
      uint32_t transposeB : 1;
      uint32_t N : 6;
      uint32_t scaleType : 2;
      uint32_t M : 4;
      uint32_t AScaleFactor : 2;
      uint32_t kSize : 1;
    };
  };
  auto getTypeEncoding = [](ScaleDotElemType type) {
    switch (type) {
    case ScaleDotElemType::E4M3:
      return 0;
    case ScaleDotElemType::E5M2:
      return 1;
    case ScaleDotElemType::E2M3:
      return 3;
    case ScaleDotElemType::E3M2:
      return 4;
    case ScaleDotElemType::E2M1:
      return 5;
    default:
      break;
    }
    llvm_unreachable("Unsupported type.");
  };
  static_assert(sizeof(TCGen5InstructionDescriptor) == 4,
                "instruction descriptor size should be 32 bits.");
  TCGen5InstructionDescriptor desc;
  desc.descriptor = 0;
  desc.transposeA = transposeA;
  desc.transposeB = transposeB;
  desc.M = M >> 5;
  desc.N = N >> 3;
  desc.aType = getTypeEncoding(op.getAType());
  desc.bType = getTypeEncoding(op.getBType());
  desc.AScaleFactor = scaleFactorsubIdxA;
  desc.BScaleFactor = scaleFactorsubIdxB;
  desc.scaleType = 1; // E8M0

  assert(kSize == 32 || kSize == 64);
  if (kSize == 64) {
    desc.kSize = 1;
    desc.AScaleFactor *= 2;
    desc.BScaleFactor *= 2;
  }

  return b.int_val(32, desc.descriptor);
}

static Value createScaleInstDescriptorFp4(
    ConversionPatternRewriter &rewriter, ttng::TCGen5MMAScaledOp op, int M,
    int N, bool transposeA, bool transposeB, int scaleFactorsubIdxA,
    int scaleFactorsubIdxB, mxfpKind mxfpInstKind, int blockK, int kSize) {
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  union TCGen5InstructionDescriptor {
    uint32_t descriptor;
    struct {
      uint32_t sparsitySelector : 2;
      uint32_t sparsity : 1;
      uint32_t kSizeUpper : 1;
      uint32_t BScaleFactor : 2;
      uint32_t : 1;
      uint32_t aType : 3;
      uint32_t bType : 2;
      uint32_t : 1;
      uint32_t negateA : 1;
      uint32_t negateB : 1;
      uint32_t transposeA : 1;
      uint32_t transposeB : 1;
      uint32_t N : 6;
      uint32_t scaleType : 2;
      uint32_t M : 4;
      uint32_t AScaleFactor : 2;
      uint32_t kSizeLower : 1;
    };
  };
  static_assert(sizeof(TCGen5InstructionDescriptor) == 4,
                "instruction descriptor size should be 32 bits.");
  TCGen5InstructionDescriptor desc;
  desc.descriptor = 0;
  desc.transposeA = transposeA;
  desc.transposeB = transposeB;
  desc.M = M >> 5;
  desc.N = N >> 3;
  desc.aType = 1;
  desc.bType = 1;
  desc.AScaleFactor = scaleFactorsubIdxA;
  desc.BScaleFactor = scaleFactorsubIdxB;
  desc.scaleType = static_cast<uint32_t>(getScaleKind(op, blockK));

  assert(kSize == 64 || kSize == 128);
  if (kSize == 128) {
    desc.kSizeUpper = 1;
  }

  assert(desc.AScaleFactor <= 1 && desc.BScaleFactor <= 1);
  assert(desc.transposeA == 0 &&
         "MMAv5 with kind=mxf4 does not support transpose");
  assert(desc.transposeB == 0 &&
         "MMAv5 with kind=mxf4 does not support transpose");

  int scaleVecSize = getScaleVecSize(op);
  if (mxfpInstKind == mxfpKind::mxf4 || (mxfpInstKind == mxfpKind::mxf4nvf4 &&
                                         scaleVecSize == 32 && kSize == 64)) {
    desc.AScaleFactor *= 2;
    desc.BScaleFactor *= 2;
    assert(desc.AScaleFactor == 0 || desc.AScaleFactor == 2 &&
                                         "MMAv5 with kind=mxf4 or "
                                         "kind=mxf4nvf4 and .block32 only "
                                         "supports SFA_ID 0 or 2");
    assert(desc.BScaleFactor == 0 || desc.BScaleFactor == 2 &&
                                         "MMAv5 with kind=mxf4 or "
                                         "kind=mxf4nvf4 and .block32 only "
                                         "supports SFB_ID 0 or 2");
  } else if (mxfpInstKind == mxfpKind::mxf4nvf4) {
    assert(desc.AScaleFactor == 0 &&
           "MMAv5 with kind=mxf4nvf4 and .block16 only supports SFA_ID 0");
    assert(desc.BScaleFactor == 0 &&
           "MMAv5 with kind=mxf4nvf4 and .block16 only supports SFB_ID 0");
  }

  return b.int_val(32, desc.descriptor);
}

//===----------------------------------------------------------------------===//
// tcgen05 instructions
//===----------------------------------------------------------------------===//

void createGen5MMA(ConversionPatternRewriter &rewriter, Location loc,
                   ttng::TCGen5MMAOp op, MemDescOperand a, Value b,
                   MemDescOperand d, Value pred, Value instDescriptor,
                   Value useInitAcc, bool aInTMem, bool twoCTAs,
                   std::string collectorB) {
  PTXBuilder ptxBuilder;
  std::string opcode =
      "tcgen05.mma.cta_group::" + std::to_string(twoCTAs ? 2 : 1) + ".kind::";
  Type srcElementTy = op.getA().getType().getElementType();
  if (srcElementTy.isF16() || srcElementTy.isBF16()) {
    opcode += "f16";
  } else if (srcElementTy.isF32()) {
    opcode += "tf32";
  } else if (llvm::isa<Float8E4M3FNType, Float8E5M2Type>(srcElementTy)) {
    opcode += "f8f6f4";
  } else if (op.getD().getType().getElementType().isInteger(32)) {
    // PTX uses "i8" for integer operations (both signed and unsigned)
    // The signed/unsigned distinction is encoded in the instruction descriptor
    opcode += "i8";
  } else {
    assert(0 && "Unsupported type.");
  }
  opcode += collectorB;
  auto *accOp = ptxBuilder.newAddrOperand(d.base, "r", *d.offset);
  assert(a.offset.has_value() == aInTMem);
  auto *aOp = aInTMem ? ptxBuilder.newAddrOperand(a.base, "r", *a.offset)
                      : ptxBuilder.newOperand(a.base, "l");
  auto *bOp = ptxBuilder.newOperand(b, "l");
  auto *instDescOp = ptxBuilder.newOperand(instDescriptor, "r");
  auto *useInitAccOp = ptxBuilder.newOperand(useInitAcc, "b");
  auto &mmaOp = *ptxBuilder.create(opcode);
  mmaOp({accOp, aOp, bOp, instDescOp, useInitAccOp}).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

void createScaledGen5MMA(ConversionPatternRewriter &rewriter, Location loc,
                         ttng::TCGen5MMAScaledOp op, MemDescOperand a, Value b,
                         MemDescOperand d, Value scaleA, Value scaleB,
                         Value pred, Value instDescriptor, Value useInitAcc,
                         bool aInTmem, mxfpKind mxfpInstKind, bool twoCTAs,
                         std::string collectorB) {
  PTXBuilder ptxBuilder;
  std::string opcode =
      "tcgen05.mma.cta_group::" + std::to_string(twoCTAs ? 2 : 1) + ".kind::";
  if (mxfpInstKind == mxfpKind::mxf8f6f4) {
    opcode += "mxf8f6f4.block_scale.block32";
  } else if (mxfpInstKind == mxfpKind::mxf4) {
    opcode += "mxf4.block_scale.block32";
  } else if (mxfpInstKind == mxfpKind::mxf4nvf4) {
    opcode += getScaleVecSize(op) == 32 ? "mxf4nvf4.block_scale.block32"
                                        : "mxf4nvf4.block_scale.block16";
  } else {
    assert(0 && "Unsupported mxfp kind.");
  }
  opcode += collectorB;
  auto *accOp = ptxBuilder.newAddrOperand(d.base, "r", *d.offset);
  assert(aInTmem == a.offset.has_value());
  auto *aOp = aInTmem ? ptxBuilder.newAddrOperand(a.base, "r", *a.offset)
                      : ptxBuilder.newOperand(a.base, "l");
  auto *bOp = ptxBuilder.newOperand(b, "l");
  auto *instDescOp = ptxBuilder.newOperand(instDescriptor, "r");
  auto *scaleAOp = ptxBuilder.newAddrOperand(scaleA, "r");
  auto *scaleBOp = ptxBuilder.newAddrOperand(scaleB, "r");
  auto *useInitAccOp = ptxBuilder.newOperand(useInitAcc, "b");
  auto &mmaOp = *ptxBuilder.create(opcode);
  mmaOp({accOp, aOp, bOp, instDescOp, scaleAOp, scaleBOp, useInitAccOp})
      .predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

void createMMACommit(ConversionPatternRewriter &rewriter, Location loc,
                     Value barrier, Value pred, bool twoCTAs,
                     ValueRange descs) {
  PTXBuilder ptxBuilder;
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value mask;
  for (uint16_t broadcastBits : ttng::getCTABroadcastMasks(twoCTAs, descs)) {
    Value descMask =
        LLVM::NVIDIA::createTMAMulticastMask(loc, rewriter, broadcastBits);
    mask = mask ? b.or_(descMask, mask) : descMask;
  }

  SmallVector<PTXBuilder::Operand *> ptxOperands;
  auto *predOperand = ptxBuilder.newOperand(pred, "b");
  ptxOperands.push_back(predOperand);
  barrier = b.ptrtoint(i32_ty, barrier);
  auto *barrierOperand = ptxBuilder.newOperand(barrier, "r");
  ptxOperands.push_back(barrierOperand);
  std::string opcode =
      "@$0 tcgen05.commit.cta_group::" + std::to_string(twoCTAs ? 2 : 1) +
      ".mbarrier::arrive::one.shared::cluster";
  if (mask)
    opcode += ".multicast::cluster";
  opcode += ".b64 [$1]";
  if (mask) {
    opcode += ", $2";
    auto *maskOperand = ptxBuilder.newOperand(mask, "h");
    ptxOperands.push_back(maskOperand);
  }
  opcode += ";";
  auto &barrierOp = *ptxBuilder.create(opcode);
  barrierOp(ptxOperands, /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

//===----------------------------------------------------------------------===//
// MMAv5 Conversion
//===----------------------------------------------------------------------===//

enum class ReuseB {
  None,
  Keep,
  Use,
  Lastuse,
};

// Information about how to lower a dot operation, shared between regular and
// scaled dot.
struct DotConversion {
  struct InstDesc {
    unsigned mmaSizeM;
    unsigned mmaSizeN;
    struct {
      int numRepM;
      int numRepN;
      int numRepK;
    } repShape;
    bool transA;
    bool transB;
    bool aInTmem;
  };

  using GetAccAddressFn = std::function<MemDescOperand(
      ConversionPatternRewriter &, Location, int, int, const InstDesc &)>;
  using CreateMMAInstFn = std::function<void(
      ConversionPatternRewriter &, Location, MemDescOperand, MemDescOperand,
      Value, Value, Value, const InstDesc &, int, int, int, ReuseB)>;

  struct {
    unsigned M;
    unsigned N;
    unsigned K;
  } shape;
  int mmaSizeK;
  SmallVector<int64_t> shapeA;
  SmallVector<int64_t> shapeB;
  int numBitsPerElementA;
  int numBitsPerElementB;
  GetAccAddressFn getAccAddress;
  CreateMMAInstFn createMMAInst;
};

LogicalResult convertDotImpl(const LLVMTypeConverter &typeConverter,
                             ConversionPatternRewriter &rewriter, Location loc,
                             Value a, Value b, Value loadedA, Value loadedB,
                             MemDescType dTensorTy, Value useDFlag, Value pred,
                             ValueRange barriers, ValueRange barrierPreds,
                             bool twoCTAs, ValueRange commitDescs,
                             bool opKindIsMXFP4,
                             const ttng::TargetFeatures &targetFeatures,
                             const DotConversion &op) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);

  // Only run mma on one thread. We currently use elect as ptxas is not able to
  // detect that tid.x == 0 is true only for 1 thread.
  Value warpId = mlir::triton::gpu::WarpIdOp::create(rewriter, loc);
  Value isWarp0 = tb.icmp_eq(warpId, tb.i32_val(0));
  if (twoCTAs) {
    Value cluster0 = LLVM::NVIDIA::createLeadCTAPredicate(loc, rewriter);
    pred = tb.and_(pred, cluster0);
  }
  pred = tb.and_(pred, isWarp0);

  // Synchronize the current partition before branching into the MMA block.
  if (!barriers.empty())
    BarrierOp::create(rewriter, loc, AddrSpace::Local);

  // Wrap the whole mma code sequence within a IF block.
  auto *curBlock = rewriter.getInsertionBlock();
  auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
  auto *mmaBlock = rewriter.createBlock(curBlock->getParent(),
                                        std::next(Region::iterator(curBlock)));
  rewriter.setInsertionPointToEnd(curBlock);
  LLVM::CondBrOp::create(rewriter, loc, pred, mmaBlock, endBlock);
  // Emit the rest in mmaBlock
  rewriter.setInsertionPointToEnd(mmaBlock);

  Value elect = LLVM::NVIDIA::createElectPredicate(loc, rewriter);

  auto aTensorTy = cast<MemDescType>(a.getType());
  auto bTensorTy = cast<MemDescType>(b.getType());
  bool aInTmem = isa<ttng::TensorMemoryEncodingAttr>(aTensorTy.getEncoding());

  Value baseA = loadedA;
  if (!aInTmem) {
    baseA = getOffsetedBase(loadedA, aTensorTy, &typeConverter, rewriter, loc);
  }
  Value baseB =
      getOffsetedBase(loadedB, bTensorTy, &typeConverter, rewriter, loc);

  auto [M, N, K] = op.shape;

  auto tensorMemAttr =
      cast<ttng::TensorMemoryEncodingAttr>(dTensorTy.getEncoding());
  unsigned mmaSizeM = tensorMemAttr.getBlockM();
  // Account for subslices
  unsigned mmaSizeN = std::min<unsigned>(tensorMemAttr.getBlockN(), N);
  // Checked in the verifier
  assert(mmaSizeN <= 256 &&
         "The maximum size of an MMA instruction is 128x256");
  unsigned mmaSizeK = op.mmaSizeK;
  int numRepM = ceil<unsigned>(M, mmaSizeM);
  int numRepN = ceil<unsigned>(N, mmaSizeN);
  assert((!twoCTAs || numRepN == 1) &&
         "grep for [Note: numRepN > 1 and two_ctas]");
  int numRepK = ceil<unsigned>(K, mmaSizeK);

  SmallVector<int64_t> shapeA = op.shapeA;
  SmallVector<int64_t> shapeB = op.shapeB;
  // In A * B = C
  // For M=64 twoCTAs, B and C have the same split and A has a split half of C
  // along M.
  SmallVector<unsigned> aOperandShape = {mmaSizeM, mmaSizeK};
  // For M=128 twoCTAs, A and C have the same split and B has a split half of C
  // along N.
  SmallVector<unsigned> bOperandShape = {mmaSizeK,
                                         mmaSizeN / (twoCTAs ? 2 : 1)};

  std::unique_ptr<DotOpMmaMemLoader> aLoader;
  bool transA = false;
  auto isFp4a = op.numBitsPerElementA == 4;
  if (aInTmem) {
    aLoader =
        std::make_unique<DotOpMmaV5TmemLoader>(DotOpMmaV5TmemLoader::build(
            loc, rewriter, aTensorTy, baseA, op.numBitsPerElementA));
  } else {
    auto loader = DotOpMmaSmemLoader::build(loc, rewriter, aTensorTy, baseA,
                                            aOperandShape, 0, 5, isFp4a);
    if (failed(loader)) {
      return mlir::emitError(loc, "failed to find valid tcgen05.mma layout for "
                                  "operand A in shared memory ")
             << aTensorTy << " for MMAv5 instruction shape [" << mmaSizeM
             << ", " << mmaSizeK << "]";
    }
    aLoader = std::make_unique<DotOpMmaSmemLoader>(std::move(*loader));
    transA = ((DotOpMmaSmemLoader *)aLoader.get())->getDescriptor().transposed;
  }

  auto isFp4b = op.numBitsPerElementB == 4;
  auto bLoader = DotOpMmaSmemLoader::build(loc, rewriter, bTensorTy, baseB,
                                           bOperandShape, 1, 5, isFp4b);
  if (failed(bLoader)) {
    return mlir::emitError(loc, "failed to find valid tcgen05.mma layout for "
                                "operand B in shared memory ")
           << bTensorTy << " for MMAv5 instruction shape [" << mmaSizeK << ", "
           << mmaSizeN << "]";
  }
  bool transB = !bLoader->getDescriptor().transposed;

  if (aTensorTy.getElementType().isF32() && (transA || transB)) {
    return mlir::emitError(loc, "tcgen05.mma does not support transposed "
                                "float32 operands in shared memory");
  }

  DotConversion::InstDesc desc{mmaSizeM, mmaSizeN, {numRepM, numRepN, numRepK},
                               transA,   transB,   aInTmem};

  // B reuse requires M = 128 for 1CTA or 256 for 2CTA, which corresponds to
  // the condition mmaSizeM == 128 here
  if (numRepM == 2 && mmaSizeM == 128 && targetFeatures.supportsReuseB()) {
    for (int n = 0; n < numRepN; n++) {
      Value useInitAcc = useDFlag;
      for (int k = 0; k < numRepK; k++) {
        Value b = bLoader->smemLoad(k * bOperandShape[0], n * bOperandShape[1],
                                    rewriter, loc);
        for (int m = 0; m < 2; m++) {
          MemDescOperand accAddress =
              op.getAccAddress(rewriter, loc, m, n, desc);
          MemDescOperand a =
              aLoader->memLoad(m * mmaSizeM, k * mmaSizeK, rewriter, loc);
          ReuseB reuseB = m == 0 ? ReuseB::Keep : ReuseB::Lastuse;
          op.createMMAInst(rewriter, loc, accAddress, a, b, elect, useInitAcc,
                           desc, m, n, k, reuseB);
        }
        useInitAcc = tb.i1_val(1);
      }
    }
  } else {
    for (int m = 0; m < numRepM; m++) {
      for (int n = 0; n < numRepN; n++) {
        Value useInitAcc = useDFlag;
        MemDescOperand accAddress = op.getAccAddress(rewriter, loc, m, n, desc);
        for (int k = 0; k < numRepK; k++) {
          MemDescOperand a = aLoader->memLoad(
              m * aOperandShape[0], k * aOperandShape[1], rewriter, loc);
          Value b = bLoader->smemLoad(k * bOperandShape[0],
                                      n * bOperandShape[1], rewriter, loc);
          op.createMMAInst(rewriter, loc, accAddress, a, b, elect, useInitAcc,
                           desc, m, n, k, ReuseB::None);
          useInitAcc = tb.i1_val(1);
        }
      }
    }
  }

  for (auto [barrier, barrierPred] : llvm::zip(barriers, barrierPreds)) {
    Value commitPred = tb.and_(barrierPred, elect);
    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, barrier, i64_ty, rewriter);
    createMMACommit(rewriter, loc, smemObj.getBase(), commitPred, twoCTAs,
                    commitDescs);
  }
  LLVM::BrOp::create(rewriter, loc, endBlock);
  return success();
}

std::string getCollectorBModifer(ReuseB reuseB) {
  if (reuseB == ReuseB::Keep) {
    return ".collector::b::fill";
  } else if (reuseB == ReuseB::Use) {
    return ".collector::b::use";
  } else if (reuseB == ReuseB::Lastuse) {
    return ".collector::b::lastuse";
  }
  return "";
}

LogicalResult convertDot(const LLVMTypeConverter &typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         ttng::TCGen5MMAOp op,
                         ttng::TCGen5MMAOpAdaptor &adaptor,
                         const ttng::TargetFeatures &targetFeatures) {
  MemDescType aTensorTy = op.getA().getType();
  MemDescType bTensorTy = op.getB().getType();
  MemDescType dTensorTy = op.getD().getType();
  bool twoCTAs = ttng::getModuleTwoCTAs(op);
  assert(twoCTAs == op.getTwoCtas());
  SmallVector<Value> commitDescs = op.getCompletionDescs();

  DotConversion dot;

  SmallVector<int64_t> dstPerCTA = triton::gpu::getShapePerCTA(dTensorTy);
  dot.shape.M = dstPerCTA[0];
  dot.shape.N = dstPerCTA[1];
  dot.shape.K = aTensorTy.getDimSize(1);
  dot.mmaSizeK = 256 / aTensorTy.getElementTypeBitWidth();

  auto tensorMemAttr =
      cast<ttng::TensorMemoryEncodingAttr>(dTensorTy.getEncoding());
  unsigned mmaSizeM = tensorMemAttr.getBlockM();

  // 2xfp8 requires M = 128 for 1CTA or 256 for 2CTA, which corresponds to
  // the condition mmaSizeM == 128 here
  if (targetFeatures.supports2xFp8Tcgen05MMA() &&
      aTensorTy.getElementTypeBitWidth() == 8 &&
      aTensorTy.getDimSize(1) >= 64 && mmaSizeM == 128) {
    dot.mmaSizeK = 64;
  }

  dot.shapeA = getShapePerCTA(aTensorTy);
  dot.shapeB = getShapePerCTA(bTensorTy);
  dot.numBitsPerElementA = aTensorTy.getElementTypeBitWidth();
  dot.numBitsPerElementB = bTensorTy.getElementTypeBitWidth();

  DotOpMmaV5TmemLoader dLoader =
      DotOpMmaV5TmemLoader::build(loc, rewriter, dTensorTy, adaptor.getD(),
                                  dTensorTy.getElementTypeBitWidth());
  dot.getAccAddress = [&](ConversionPatternRewriter &rewriter, Location loc,
                          int m, int n, const DotConversion::InstDesc &desc) {
    return dLoader.tmemLoad(m * desc.mmaSizeM, n * desc.mmaSizeN, rewriter,
                            loc);
  };

  dot.createMMAInst = [&](ConversionPatternRewriter &rewriter, Location loc,
                          MemDescOperand accAddress, MemDescOperand a, Value b,
                          Value pred, Value useInitAcc,
                          const DotConversion::InstDesc &desc, int m, int n,
                          int k, ReuseB reuseB) {
    // mmaSizeM/N is the per-cta size M/N, while the 2CTA instruction expects
    // the 2CTA size mmaSize is always 64 / 128 so we double it for 2CTA
    auto mmaSizeM = twoCTAs ? desc.mmaSizeM * 2 : desc.mmaSizeM;
    auto mmaSizeN = desc.mmaSizeN;
    assert(desc.mmaSizeM == 64 || desc.mmaSizeM == 128);
    Value instDescriptor =
        createInstDescriptor(rewriter, op, mmaSizeM, mmaSizeN, desc.transA,
                             desc.transB, dot.mmaSizeK);
    auto collectorB = getCollectorBModifer(reuseB);
    createGen5MMA(rewriter, loc, op, a, b, accAddress, pred, instDescriptor,
                  useInitAcc, desc.aInTmem, twoCTAs, collectorB);
  };

  return convertDotImpl(
      typeConverter, rewriter, loc, op.getA(), op.getB(), adaptor.getA(),
      adaptor.getB(), dTensorTy, adaptor.getUseD(), adaptor.getPred(),
      adaptor.getBarriers(), adaptor.getBarrierPreds(), twoCTAs, commitDescs,
      /*opKindIsMXFP4=*/false, targetFeatures, dot);
}

int64_t getFormatBitSize(ScaleDotElemType type) {
  switch (type) {
  case ScaleDotElemType::E4M3:
    return 8;
  case ScaleDotElemType::E5M2:
    return 8;
  case ScaleDotElemType::E2M3:
    return 6;
  case ScaleDotElemType::E3M2:
    return 6;
  case ScaleDotElemType::E2M1:
    return 4;
  default:
    llvm_unreachable("Unsupported type.");
  }
}

int getScaleFactorColsPerSet(mxfpKind kind, ttng::TCGen5MMAScaledOp op,
                             int kSize) {
  switch (kind) {
  case mxfpKind::mxf8f6f4:
    return kSize == 32 ? 1 : 2;
  case mxfpKind::mxf4:
    return kSize == 64 ? 2 : 4;
  case mxfpKind::mxf4nvf4:
    return getScaleVecSize(op) == 32 && kSize == 64 ? 2 : 4;
  default:
    llvm_unreachable("Unsupported mxfp kind.");
  }
};

bool isFp4Padded(MemDescType operand) {
  if (auto tmemLayout =
          dyn_cast<ttng::TensorMemoryEncodingAttr>(operand.getEncoding()))
    return tmemLayout.getFp4Padded();
  auto attrs = ttng::getNvmmaSmemAttrs(operand);
  assert(attrs && "expected MMAv5 shared operand to have NVMMA SMEM attrs");
  return attrs->fp4Padded;
}

int linearizeScaleBlockIdx(Value scale, int mnIdx, int wordIdx, int numRepMn,
                           int numRepKWords) {
  auto enc = cast<ttng::TensorMemoryScalesEncodingAttr>(
      cast<MemDescType>(scale.getType()).getEncoding());
  return enc.getBlockRepOrder() ==
                 ttng::TensorMemoryScalesBlockRepOrder::K_THEN_MN
             ? (wordIdx + mnIdx * numRepKWords)
             : (mnIdx + wordIdx * numRepMn);
}

LogicalResult convertScaledDot(const LLVMTypeConverter &typeConverter,
                               ConversionPatternRewriter &rewriter,
                               Location loc, ttng::TCGen5MMAScaledOp op,
                               ttng::TCGen5MMAScaledOpAdaptor &adaptor,
                               const ttng::TargetFeatures &targetFeatures) {
  MemDescType aTensorTy = op.getA().getType();
  MemDescType bTensorTy = op.getB().getType();
  MemDescType dTensorTy = op.getD().getType();
  int blockK = op.getBlockK();

  mxfpKind mxfpInstKind =
      getMXFPKind(op.getAType(), op.getBType(), op.getAScale().getType(),
                  op.getBScale().getType(), blockK,
                  isTransposed(op.getA()) || !isTransposed(op.getB()));
  bool opKindIsMXFP4 = mxfpInstKind != mxfpKind::mxf8f6f4;

  DotConversion dot;

  auto tensorMemAttr =
      cast<ttng::TensorMemoryEncodingAttr>(dTensorTy.getEncoding());
  unsigned mmaSizeM = tensorMemAttr.getBlockM();
  unsigned mmaSizeN = tensorMemAttr.getBlockN();

  // Use per-CTA shape to correctly handle 2-CTA mode. getBlockM/N() returns
  // the full block shape (e.g., 256 for 2-CTA), but we need the per-CTA shape
  // (e.g., 128) to compute the correct number of MMA repetitions.
  SmallVector<int64_t> dstPerCTA = triton::gpu::getShapePerCTA(dTensorTy);
  dot.shape.M = dstPerCTA[0];
  dot.shape.N = dstPerCTA[1];
  dot.shape.K = blockK; // K is not split across CTAs

  dot.shapeA = triton::gpu::getAllocationShapePerCTA(aTensorTy);
  dot.shapeB = triton::gpu::getAllocationShapePerCTA(bTensorTy);
  if (opKindIsMXFP4) {
    dot.shapeA[1] *= 2;
    dot.shapeB[0] *= 2;
  }

  bool hasFp4PaddedOperand = isFp4Padded(aTensorTy) || isFp4Padded(bTensorTy);

  if (!targetFeatures.supports4xFp4Tcgen05MMA() || hasFp4PaddedOperand ||
      // M = 64 for 1CTA or 128 for 2CTA not supported for 2xfp8 / 4xfp4
      mmaSizeM == 64 ||
      // We need to disable MMA_K = 128 for NVFP4 on Rubin when MMA_N < 128,
      // which requires a special unpacked layout for scales B in TMEM
      // (currently undocumented). tensorMemoryScalesToLinearLayout and lowering
      // of tcgen05.cp need to be updated for this.
      (mxfpInstKind == mxfpKind::mxf4nvf4 && mmaSizeN < 128)) {
    dot.mmaSizeK = opKindIsMXFP4 ? 64 : 32;
  } else {
    dot.mmaSizeK = opKindIsMXFP4 ? std::min(blockK, 128) : std::min(blockK, 64);
  }

  dot.numBitsPerElementA = getFormatBitSize(op.getAType());
  dot.numBitsPerElementB = getFormatBitSize(op.getBType());

  TritonLLVMOpBuilder tb(loc, rewriter);
  DotOpMmaV5TmemLoader dLoader =
      DotOpMmaV5TmemLoader::build(loc, rewriter, dTensorTy, adaptor.getD(),
                                  dTensorTy.getElementTypeBitWidth());
  Value baseScaleA = tb.ptrtoint(i32_ty, adaptor.getAScale());
  Value baseScaleB = tb.ptrtoint(i32_ty, adaptor.getBScale());
  bool twoCTAs = ttng::getModuleTwoCTAs(op);
  SmallVector<Value> commitDescs = op.getCompletionDescs();

  dot.getAccAddress = [&](ConversionPatternRewriter &rewriter, Location loc,
                          int m, int n, const DotConversion::InstDesc &desc) {
    return dLoader.tmemLoad(m * desc.mmaSizeM, n * desc.mmaSizeN, rewriter,
                            loc);
  };

  dot.createMMAInst = [&](ConversionPatternRewriter &rewriter, Location loc,
                          MemDescOperand accAddress, MemDescOperand a, Value b,
                          Value pred, Value useInitAcc,
                          const DotConversion::InstDesc &desc, int m, int n,
                          int k, ReuseB reuseB) {
    auto [numRepM, numRepN, numRepK] = desc.repShape;
    int scaleFactorColsPerSet =
        getScaleFactorColsPerSet(mxfpInstKind, op, dot.mmaSizeK);
    int numRepKWords = ceil<int>(numRepK, 4 / scaleFactorColsPerSet);
    int numColPerScaleBlockA = ceil<int>(
        ttng::getTmemAllocSizes(cast<MemDescType>(op.getAScale().getType()))
            .numCols,
        numRepM * numRepKWords);
    int numColPerScaleBlockB = ceil<int>(
        ttng::getTmemAllocSizes(cast<MemDescType>(op.getBScale().getType()))
            .numCols,
        numRepN * numRepKWords);
    numColPerScaleBlockB = std::max(numColPerScaleBlockB, 2);
    int subWordIdx = k % (4 / scaleFactorColsPerSet);
    int wordIdx = k / (4 / scaleFactorColsPerSet);
    int scaleIdxA = linearizeScaleBlockIdx(op.getAScale(), m, wordIdx, numRepM,
                                           numRepKWords);
    int scaleIdxB = linearizeScaleBlockIdx(op.getBScale(), n, wordIdx, numRepN,
                                           numRepKWords);
    Value scaleA =
        tb.add(baseScaleA, tb.i32_val(scaleIdxA * numColPerScaleBlockA));
    Value scaleB =
        tb.add(baseScaleB, tb.i32_val(scaleIdxB * numColPerScaleBlockB));
    Value instDescriptor;
    // For 2CTA mode, the M dimension in the instruction descriptor must be
    // doubled to match the hardware's expectation for cta_group::2 operations.
    int mmaM = twoCTAs ? desc.mmaSizeM * 2 : desc.mmaSizeM;
    if (mxfpInstKind == mxfpKind::mxf8f6f4) {
      instDescriptor = createScaleInstDescriptorFp8(
          rewriter, op, mmaM, desc.mmaSizeN, desc.transA, desc.transB,
          subWordIdx, subWordIdx, dot.mmaSizeK);
    } else {
      instDescriptor = createScaleInstDescriptorFp4(
          rewriter, op, mmaM, desc.mmaSizeN, desc.transA, desc.transB,
          subWordIdx, subWordIdx, mxfpInstKind, blockK, dot.mmaSizeK);
    }

    auto collectorB = getCollectorBModifer(reuseB);
    createScaledGen5MMA(rewriter, loc, op, a, b, accAddress, scaleA, scaleB,
                        pred, instDescriptor, useInitAcc, desc.aInTmem,
                        mxfpInstKind, twoCTAs, collectorB);
  };

  return convertDotImpl(
      typeConverter, rewriter, loc, op.getA(), op.getB(), adaptor.getA(),
      adaptor.getB(), dTensorTy, adaptor.getUseD(), adaptor.getPred(),
      adaptor.getBarriers(), adaptor.getBarrierPreds(), twoCTAs, commitDescs,
      opKindIsMXFP4, targetFeatures, dot);
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

struct TCGen5MMAOpConversion
    : public ConvertOpToLLVMPattern<ttng::TCGen5MMAOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  TCGen5MMAOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                        const ttng::TargetFeatures &targetFeatures)
      : ConvertOpToLLVMPattern<ttng::TCGen5MMAOp>(converter, benefit),
        targetFeatures(targetFeatures) {}

  LogicalResult
  matchAndRewrite(ttng::TCGen5MMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(convertDot(*getTypeConverter(), rewriter, op.getLoc(), op,
                          adaptor, targetFeatures)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }

  ttng::TargetFeatures targetFeatures;
};

struct TCGen5MMAScaledOpConversion
    : public ConvertOpToLLVMPattern<ttng::TCGen5MMAScaledOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  TCGen5MMAScaledOpConversion(LLVMTypeConverter &converter,
                              PatternBenefit benefit,
                              const ttng::TargetFeatures &targetFeatures)
      : ConvertOpToLLVMPattern<ttng::TCGen5MMAScaledOp>(converter, benefit),
        targetFeatures(targetFeatures) {}

  LogicalResult
  matchAndRewrite(ttng::TCGen5MMAScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(convertScaledDot(*getTypeConverter(), rewriter, op.getLoc(), op,
                                adaptor, targetFeatures)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }

private:
  ttng::TargetFeatures targetFeatures;
};

struct TCGen5CommitOpConversion
    : public ConvertOpToLLVMPattern<ttng::TCGen5CommitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttng::TCGen5CommitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);

    // Because this operation can signal other partitions we need to synchronize
    // the current partition first.
    BarrierOp::create(rewriter, loc, AddrSpace::Local);

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getBarrier(), rewriter.getI64Type(), rewriter);
    Value pred = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);

    if (adaptor.getPred())
      pred = b.and_(adaptor.getPred(), pred);

    bool twoCTAs = ttng::getModuleTwoCTAs(op);
    if (twoCTAs) {
      Value cluster0 = LLVM::NVIDIA::createLeadCTAPredicate(loc, rewriter);
      pred = b.and_(pred, cluster0);
    }

    createMMACommit(rewriter, op.getLoc(), smemObj.getBase(), pred, twoCTAs,
                    op.getDescs());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace NVIDIA {

void populateTCGen5MMAOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      PatternBenefit benefit,
                                      const TargetInfo &targetInfo) {
  patterns.add<TCGen5MMAOpConversion, TCGen5MMAScaledOpConversion>(
      typeConverter, benefit, targetInfo.getTargetFeatures());
  patterns.add<TCGen5CommitOpConversion>(typeConverter, benefit);
}

} // namespace NVIDIA
} // namespace triton
} // namespace mlir
