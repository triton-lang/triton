#include "Dialect/NVGPU/IR/Dialect.h"
#include "MMAHelpers.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::NVIDIA;
namespace ttng = mlir::triton::nvidia_gpu;

using ::mlir::triton::gpu::NVMMASharedEncodingAttr;

//===----------------------------------------------------------------------===//
// DotOpMmaV5TmemLoader
//===----------------------------------------------------------------------===//

mlir::triton::NVIDIA::DotOpMmaV5TmemLoader::DotOpMmaV5TmemLoader(
    Value tensor, Value base, SmallVector<unsigned int> instrShape,
    bool interleaved, bool trans)
    : base(base), instrShape(instrShape), interleaved(interleaved),
      trans(trans) {
  auto ty = cast<MemDescType>(tensor.getType());
  auto tmemEncoding = cast<ttng::TensorMemoryEncodingAttr>(ty.getEncoding());
  int elTyWidth = ty.getElementTypeBitWidth();
  unpacked = tmemEncoding.getColStride() != 1;
  // When using TMEM to store operands mma operands the TMEM block size may be
  // smaller than mma k block. Therefore we need to adjust the offset
  // calculation.
  numSlicePerBlockN = tmemEncoding.getBlockN() / instrShape[1];
  numElementsPer32b = 32 / (elTyWidth * tmemEncoding.getColStride());
  auto shapePerCTA = triton::gpu::getShapePerCTA(ty);
  numRepM = ceil<unsigned>(shapePerCTA[0], instrShape[0]);
}

MemDescOperand mlir::triton::NVIDIA::DotOpMmaV5TmemLoader::tmemLoad(
    int a, int b, ConversionPatternRewriter &rewriter, Location loc) const {
  int numRows = 64;
  if (interleaved || instrShape[0] >= 128)
    numRows = 128;
  int numColPerBlock =
      ((instrShape[0] * numSlicePerBlockN * instrShape[1]) / numRows) /
      numElementsPer32b;
  int blockId = a + (b / numSlicePerBlockN) * numRepM;
  int offset;
  if (!interleaved) {
    offset = numColPerBlock * blockId;
  } else {
    int blockIdIsOdd = blockId & 1;
    int blockIdPrevEven = blockId - blockIdIsOdd;
    offset = numColPerBlock * blockIdPrevEven + ((16 * blockIdIsOdd) << 16);
  }
  offset += (b % numSlicePerBlockN) * (instrShape[1] / numElementsPer32b);
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  Value address = tb.ptrtoint(i32_ty, base);
  return {address, offset};
}

//===----------------------------------------------------------------------===//
// InstDescriptor
//===----------------------------------------------------------------------===//

namespace {

enum class mxfpKind { mxf8f6f4 = 0, mxf4 = 1, mxf4nvf4 = 2 };

inline mxfpKind getMXFPKind(ScaleDotElemType typeA, ScaleDotElemType typeB,
                            Type scaleAType, Type scaleBType, bool transpose) {
  if (typeA == ScaleDotElemType::E2M1 && typeB == ScaleDotElemType::E2M1) {
    if (llvm::isa<Float8E4M3FNType>(scaleAType) &&
        llvm::isa<Float8E4M3FNType>(scaleBType)) {
      assert(!transpose &&
             "MMAv5 with kind=mxf4nvf4 does not support transpose");
      return mxfpKind::mxf4nvf4;
    }
    if (!transpose)
      return mxfpKind::mxf4;
  }
  return mxfpKind::mxf8f6f4;
};

static Value createInstDescriptor(ConversionPatternRewriter &rewriter,
                                  ttng::TCGen5MMAOp op, int M, int N,
                                  bool transposeA, bool transposeB) {
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
      uint32_t : 1;
      uint32_t shift : 2;
    };
  };
  auto getTypeEncoding = [](Type type) {
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
  assert(dstElType.isF16() || dstElType.isF32());
  desc.dType = dstElType.isF16() ? 0 : 1;
  return b.int_val(32, desc.descriptor);
}

static Value createScaleInstDescriptor(ConversionPatternRewriter &rewriter,
                                       ttng::TCGen5MMAScaledOp op, int M, int N,
                                       bool transposeA, bool transposeB,
                                       int scaleFactorsubIdxA,
                                       int scaleFactorsubIdxB,
                                       mxfpKind mxfpInstKind) {
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
      uint32_t scaleType : 1;
      uint32_t M : 5;
      uint32_t AScaleFactor : 2;
      uint32_t : 1;
    };
  };
  auto getTypeEncoding = [](ScaleDotElemType type, bool isMXF4) {
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
      return !isMXF4 ? 5 : 1;
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
  desc.M = M >> 4;
  desc.N = N >> 3;
  desc.aType =
      getTypeEncoding(op.getAType(), mxfpInstKind != mxfpKind::mxf8f6f4);
  desc.bType =
      getTypeEncoding(op.getBType(), mxfpInstKind != mxfpKind::mxf8f6f4);
  desc.AScaleFactor = scaleFactorsubIdxA;
  desc.BScaleFactor = scaleFactorsubIdxB;
  // Hardcoded UE8M0 scale type.
  desc.scaleType = 1;

  if (mxfpInstKind != mxfpKind::mxf8f6f4) {
    assert(desc.aType == 1 && desc.bType == 1);
    assert(desc.AScaleFactor <= 1 && desc.BScaleFactor <= 1);
    assert(desc.transposeA == 0 &&
           "MMAv5 with kind=mxf4 does not support transpose");
    assert(desc.transposeB == 0 &&
           "MMAv5 with kind=mxf4 does not support transpose");
    if (mxfpInstKind == mxfpKind::mxf4) {
      desc.AScaleFactor *= 2;
      desc.BScaleFactor *= 2;
      assert(desc.AScaleFactor == 0 ||
             desc.AScaleFactor == 2 &&
                 "MMAv5 with kind=mxf4 only supports SFA_ID 0 or 2");
      assert(desc.BScaleFactor == 0 ||
             desc.BScaleFactor == 2 &&
                 "MMAv5 with kind=mxf4 only supports SFB_ID 0 or 2");
    } else if (mxfpInstKind == mxfpKind::mxf4nvf4) {
      desc.scaleType = 0; // UE4M3
      assert(desc.AScaleFactor == 0 &&
             "MMAv5 with kind=mxf4nvf4 currently only supports SFA_ID 0");
      assert(desc.BScaleFactor == 0 &&
             "MMAv5 with kind=mxf4nvf4 currently only supports SFB_ID 0");
    }
  }

  return b.int_val(32, desc.descriptor);
}

//===----------------------------------------------------------------------===//
// tcgen05 instructions
//===----------------------------------------------------------------------===//

static void createGen5MMA(ConversionPatternRewriter &rewriter, Location loc,
                          ttng::TCGen5MMAOp op, MemDescOperand a, Value b,
                          MemDescOperand d, Value pred, Value instDescriptor,
                          Value useInitAcc, bool aInTMem, bool twoCTAs) {
  PTXBuilder ptxBuilder;
  std::string opcode =
      "tcgen05.mma.cta_group::" + std::to_string(twoCTAs ? 2 : 1) + ".kind::";
  Type srcElementTy = op.getA().getType().getElementType();
  if (srcElementTy.isF16() || srcElementTy.isBF16())
    opcode += "f16";
  else if (srcElementTy.isF32())
    opcode += "tf32";
  else if (llvm::isa<Float8E4M3FNType, Float8E5M2Type>(srcElementTy))
    opcode += "f8f6f4";
  else
    assert(0 && "Unsupported type.");
  auto *accOp = ptxBuilder.newAddrOperand(d.base, "r", *d.offset);
  assert(a.offset.has_value() == aInTMem);
  auto *aOp = aInTMem ? ptxBuilder.newAddrOperand(a.base, "r", *a.offset)
                      : ptxBuilder.newOperand(a.base, "l");
  auto *bOp = ptxBuilder.newOperand(b, "l");
  auto *instDescOp = ptxBuilder.newOperand(instDescriptor, "r");
  auto *useInitAccOp = ptxBuilder.newOperand(useInitAcc, "b");
  auto &mmaOp = *ptxBuilder.create<PTXInstr>(opcode);
  mmaOp({accOp, aOp, bOp, instDescOp, useInitAccOp}).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static void createScaledGen5MMA(ConversionPatternRewriter &rewriter,
                                Location loc, ttng::TCGen5MMAScaledOp op,
                                MemDescOperand a, Value b, MemDescOperand d,
                                Value scaleA, Value scaleB, Value pred,
                                Value instDescriptor, Value useInitAcc,
                                bool aInTmem, mxfpKind mxfpInstKind) {
  PTXBuilder ptxBuilder;
  std::string opcode;
  if (mxfpInstKind == mxfpKind::mxf8f6f4) {
    opcode =
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X";
  } else if (mxfpInstKind == mxfpKind::mxf4) {
    opcode = "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X";
  } else if (mxfpInstKind == mxfpKind::mxf4nvf4) {
    opcode =
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X";
  } else {
    assert(0 && "Unsupported mxfp kind.");
  }
  auto *accOp = ptxBuilder.newAddrOperand(d.base, "r", *d.offset);
  assert(aInTmem == a.offset.has_value());
  auto *aOp = aInTmem ? ptxBuilder.newAddrOperand(a.base, "r", *a.offset)
                      : ptxBuilder.newOperand(a.base, "l");
  auto *bOp = ptxBuilder.newOperand(b, "l");
  auto *instDescOp = ptxBuilder.newOperand(instDescriptor, "r");
  auto *scaleAOp = ptxBuilder.newAddrOperand(scaleA, "r");
  auto *scaleBOp = ptxBuilder.newAddrOperand(scaleB, "r");
  auto *useInitAccOp = ptxBuilder.newOperand(useInitAcc, "b");
  auto &mmaOp = *ptxBuilder.create<PTXInstr>(opcode);
  mmaOp({accOp, aOp, bOp, instDescOp, scaleAOp, scaleBOp, useInitAccOp})
      .predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static void createMMACommit(ConversionPatternRewriter &rewriter, Location loc,
                            Value barrier, Value pred, bool twoCTAs = false) {
  PTXBuilder ptxBuilder;
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<PTXBuilder::Operand *> ptxOperands;
  auto *predOperand = ptxBuilder.newOperand(pred, "b");
  ptxOperands.push_back(predOperand);
  auto *barrierOperand = ptxBuilder.newOperand(barrier, "l");
  ptxOperands.push_back(barrierOperand);
  std::string opcode;
  if (twoCTAs) {
    // .multicast::cluster and mask 0x3 means the completion of UTCMMA.2CTA will
    // be broadcasted into CTAid 0 and 1
    auto *ctaMask = ptxBuilder.newOperand(b.int_val(16, 0x3), "h");
    ptxOperands.push_back(ctaMask);
    opcode = "@$0 "
             "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::"
             "cluster.multicast::cluster.b64 [$1], $2;";
  } else {
    opcode = "@$0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$1];";
  }
  auto &barrierOp = *ptxBuilder.create<PTXInstr>(opcode);
  barrierOp(ptxOperands, /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

//===----------------------------------------------------------------------===//
// MMAv5 Conversion
//===----------------------------------------------------------------------===//

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
    bool interleaved;
    bool aInTmem;
  };

  using GetAccAddressFn = std::function<MemDescOperand(
      ConversionPatternRewriter &, Location, int, int, const InstDesc &)>;
  using CreateMMAInstFn = std::function<void(
      ConversionPatternRewriter &, Location, MemDescOperand, MemDescOperand,
      Value, Value, Value, const InstDesc &, int, int, int)>;

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

static bool isTransposed(Value operand) {
  auto tensorTy = cast<MemDescType>(operand.getType());
  if (auto shared = dyn_cast<NVMMASharedEncodingAttr>(tensorTy.getEncoding()))
    return shared.getTransposed();
  return false;
}

void convertDotImpl(const LLVMTypeConverter &typeConverter,
                    ConversionPatternRewriter &rewriter, Location loc, Value a,
                    Value b, Value loadedA, Value loadedB,
                    MemDescType dTensorTy, Value useDFlag, Value pred,
                    ValueRange barriers, ValueRange barrierPreds, bool twoCTAs,
                    bool opKindIsMXFP4, const DotConversion &op) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);

  // Only run mma on one thread. We currently use elect as ptxas is not able to
  // detect that tid.x == 0 is true only for 1 thread.
  Value warpId = rewriter.create<nvgpu::WarpIdOp>(loc);
  Value isWarp0 = tb.icmp_eq(warpId, tb.i32_val(0));
  if (twoCTAs) {
    // TODO: we have to sync the two CTAs because we currently don't use remove
    // barriers for the copies.
    rewriter.create<ttng::ClusterArriveOp>(loc, false);
    rewriter.create<ttng::ClusterWaitOp>(loc);

    Value clusterId = rewriter.create<nvgpu::ClusterCTAIdOp>(loc);
    Value cluster0 = tb.icmp_eq(clusterId, tb.i32_val(0));
    pred = tb.and_(pred, cluster0);
  }
  pred = tb.and_(pred, isWarp0);

  // Wrap the whole mma code sequence within a IF block.
  auto *curBlock = rewriter.getInsertionBlock();
  auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
  auto *mmaBlock = rewriter.createBlock(curBlock->getParent(),
                                        std::next(Region::iterator(curBlock)));
  rewriter.setInsertionPointToEnd(curBlock);
  rewriter.create<LLVM::CondBrOp>(loc, pred, mmaBlock, endBlock);
  // Emit the rest in mmaBlock
  rewriter.setInsertionPointToEnd(mmaBlock);

  Value elect = LLVM::NVIDIA::createElectPredicate(loc, rewriter);

  auto aTensorTy = cast<MemDescType>(a.getType());
  auto bTensorTy = cast<MemDescType>(b.getType());
  bool aInTmem = isa<ttng::TensorMemoryEncodingAttr>(aTensorTy.getEncoding());
  bool transA = isTransposed(a);
  bool transB = !isTransposed(b);

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
  unsigned mmaSizeN = tensorMemAttr.getBlockN();
  unsigned mmaSizeK = op.mmaSizeK;
  int numRepM = ceil<unsigned>(M, mmaSizeM);
  int numRepN = ceil<unsigned>(N, mmaSizeN);
  int numRepK = ceil<unsigned>(K, mmaSizeK);
  bool interleaved = (mmaSizeM == 64 && (numRepM > 1 || numRepN > 1));

  assert((!aTensorTy.getElementType().isF32() || !(transA || transB)) &&
         "Currently don't support transpose for F32.");

  Value zero = tb.i32_val(0);
  SmallVector<int64_t> shapeA = op.shapeA;
  SmallVector<int64_t> shapeB = op.shapeB;
  SmallVector<unsigned> aOperandShape = {mmaSizeM, mmaSizeK};

  auto getAllocShape = [&](MemDescType tensorTy, int kDim) {
    // allocationShape uses the shape, not the `allocShape`?
    auto fullAllocShape = triton::gpu::getAllocationShapePerCTA(
        tensorTy.getEncoding(), tensorTy.getAllocShape());
    auto ret = to_vector(ArrayRef<int64_t>(fullAllocShape).take_back(2));

    if (opKindIsMXFP4) {
      ret[kDim] *= 2;
    }
    return ret;
  };

  std::unique_ptr<DotOpMmaMemLoader> aLoader;
  if (aInTmem) {
    aLoader = std::make_unique<DotOpMmaV5TmemLoader>(a, baseA, aOperandShape,
                                                     interleaved, transA);
  } else {
    auto allocShapeA = getAllocShape(aTensorTy, 1);
    aLoader = std::make_unique<DotOpMmaSmemLoader>(DotOpMmaSmemLoader::build(
        loc, rewriter, aTensorTy, baseA, aOperandShape, 5));
  }

  auto allocShapeB = getAllocShape(bTensorTy, 0);
  DotOpMmaSmemLoader bLoader = DotOpMmaSmemLoader::build(
      loc, rewriter, bTensorTy, baseB, {mmaSizeK, mmaSizeN}, 5);

  DotConversion::InstDesc desc{mmaSizeM, mmaSizeN, {numRepM, numRepN, numRepK},
                               transA,   transB,   interleaved,
                               aInTmem};
  for (int m = 0; m < numRepM; m++) {
    for (int n = 0; n < numRepN; n++) {
      Value useInitAcc = useDFlag;
      MemDescOperand accAddress = op.getAccAddress(rewriter, loc, m, n, desc);
      for (int k = 0; k < numRepK; k++) {
        MemDescOperand a = aLoader->memLoad(m, k, rewriter, loc);
        Value b = bLoader.smemLoad(k, n, rewriter, loc);
        op.createMMAInst(rewriter, loc, accAddress, a, b, elect, useInitAcc,
                         desc, m, n, k);
        useInitAcc = tb.i1_val(1);
      }
    }
  }

  for (auto [barrier, barrierPred] : llvm::zip(barriers, barrierPreds)) {
    Value commitPred = tb.and_(barrierPred, elect);
    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, barrier, i64_ty, rewriter);
    createMMACommit(rewriter, loc, smemObj.getBase(), commitPred, twoCTAs);
  }
  rewriter.create<LLVM::BrOp>(loc, endBlock);
}

void convertDot(const LLVMTypeConverter &typeConverter,
                ConversionPatternRewriter &rewriter, Location loc,
                ttng::TCGen5MMAOp op, ttng::TCGen5MMAOpAdaptor &adaptor) {
  MemDescType aTensorTy = op.getA().getType();
  MemDescType bTensorTy = op.getB().getType();
  MemDescType dTensorTy = op.getD().getType();
  bool twoCTAs = op.getTwoCtas();

  DotConversion dot;

  SmallVector<int64_t> dstPerCTA = triton::gpu::getShapePerCTA(dTensorTy);
  dot.shape.M = dstPerCTA[0];
  dot.shape.N = dstPerCTA[1];
  dot.shape.K = aTensorTy.getDimSize(1);
  dot.mmaSizeK = 256 / aTensorTy.getElementTypeBitWidth();

  dot.shapeA = getShapePerCTA(aTensorTy);
  dot.shapeB = getShapePerCTA(bTensorTy);
  dot.numBitsPerElementA = aTensorTy.getElementTypeBitWidth();
  dot.numBitsPerElementB = bTensorTy.getElementTypeBitWidth();

  dot.getAccAddress = [&](ConversionPatternRewriter &rewriter, Location loc,
                          int m, int n, const DotConversion::InstDesc &desc) {
    DotOpMmaV5TmemLoader dLoader = DotOpMmaV5TmemLoader(
        op.getD(), adaptor.getD(), {desc.mmaSizeM, desc.mmaSizeN},
        desc.interleaved, /*trans=*/false);
    return dLoader.tmemLoad(m, n, rewriter, loc);
  };

  dot.createMMAInst = [&](ConversionPatternRewriter &rewriter, Location loc,
                          MemDescOperand accAddress, MemDescOperand a, Value b,
                          Value pred, Value useInitAcc,
                          const DotConversion::InstDesc &desc, int m, int n,
                          int k) {
    Value instDescriptor = createInstDescriptor(
        rewriter, op, twoCTAs ? desc.mmaSizeM * 2 : desc.mmaSizeM,
        desc.mmaSizeN, desc.transA, desc.transB);
    createGen5MMA(rewriter, loc, op, a, b, accAddress, pred, instDescriptor,
                  useInitAcc, desc.aInTmem, twoCTAs);
  };

  convertDotImpl(typeConverter, rewriter, loc, op.getA(), op.getB(),
                 adaptor.getA(), adaptor.getB(), dTensorTy, adaptor.getUseD(),
                 adaptor.getPred(), adaptor.getBarriers(),
                 adaptor.getBarrierPreds(), twoCTAs, /*opKindIsMXFP4=*/false,
                 dot);
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

int getScaleFactorColsPerSet(mxfpKind kind) {
  switch (kind) {
  case mxfpKind::mxf8f6f4:
    return 1;
  case mxfpKind::mxf4:
    return 2;
  case mxfpKind::mxf4nvf4:
    return 4;
  default:
    llvm_unreachable("Unsupported mxfp kind.");
  }
};

void convertScaledDot(const LLVMTypeConverter &typeConverter,
                      ConversionPatternRewriter &rewriter, Location loc,
                      ttng::TCGen5MMAScaledOp op,
                      ttng::TCGen5MMAScaledOpAdaptor &adaptor) {
  MemDescType aTensorTy = op.getA().getType();
  MemDescType bTensorTy = op.getB().getType();
  MemDescType dTensorTy = op.getD().getType();

  mxfpKind mxfpInstKind = getMXFPKind(
      op.getAType(), op.getBType(), op.getAScale().getType().getElementType(),
      op.getBScale().getType().getElementType(),
      isTransposed(op.getA()) || !isTransposed(op.getB()));
  bool opKindIsMXFP4 = mxfpInstKind != mxfpKind::mxf8f6f4;

  DotConversion dot;

  dot.shape.M = op.getBlockM();
  dot.shape.N = op.getBlockN();
  dot.shape.K = op.getBlockK();
  dot.mmaSizeK = !opKindIsMXFP4 ? 32 : 64;

  dot.shapeA = triton::gpu::getAllocationShapePerCTA(aTensorTy);
  dot.shapeB = triton::gpu::getAllocationShapePerCTA(bTensorTy);
  if (opKindIsMXFP4) {
    dot.shapeA[1] *= 2;
    dot.shapeB[0] *= 2;
  }

  dot.numBitsPerElementA = opKindIsMXFP4 ? getFormatBitSize(op.getAType())
                                         : aTensorTy.getElementTypeBitWidth();
  dot.numBitsPerElementB = opKindIsMXFP4 ? getFormatBitSize(op.getBType())
                                         : bTensorTy.getElementTypeBitWidth();

  TritonLLVMOpBuilder tb(loc, rewriter);
  Value baseD = tb.ptrtoint(i32_ty, adaptor.getD());
  Value baseScaleA = tb.ptrtoint(i32_ty, adaptor.getAScale());
  Value baseScaleB = tb.ptrtoint(i32_ty, adaptor.getBScale());

  int numRows = 128;
  int colSizeInBits = 32;
  dot.getAccAddress = [&](ConversionPatternRewriter &rewriter, Location loc,
                          int m, int n, const DotConversion::InstDesc &desc) {
    int numColPerBlock = ceil<int>(desc.mmaSizeM * desc.mmaSizeN *
                                       dTensorTy.getElementTypeBitWidth(),
                                   numRows * colSizeInBits);
    int blockId = m + n * desc.repShape.numRepM;
    return MemDescOperand{baseD, numColPerBlock * blockId};
  };

  dot.createMMAInst = [&](ConversionPatternRewriter &rewriter, Location loc,
                          MemDescOperand accAddress, MemDescOperand a, Value b,
                          Value pred, Value useInitAcc,
                          const DotConversion::InstDesc &desc, int m, int n,
                          int k) {
    auto [numRepM, numRepN, numRepK] = desc.repShape;
    int scaleFactorColsPerSet = getScaleFactorColsPerSet(mxfpInstKind);
    int numColPerScaleBlockA = ceil<int>(
        ttng::getTmemAllocSizes(cast<MemDescType>(op.getAScale().getType()))
            .numCols,
        numRepM * (ceil<int>(numRepK, 4 / scaleFactorColsPerSet)));
    int numColPerScaleBlockB = ceil<int>(
        ttng::getTmemAllocSizes(cast<MemDescType>(op.getBScale().getType()))
            .numCols,
        numRepN * (ceil<int>(numRepK, 4 / scaleFactorColsPerSet)));
    numColPerScaleBlockB = std::max(numColPerScaleBlockB, 2);
    int subWordIdx = k % (4 / scaleFactorColsPerSet);
    int wordIdx = k / (4 / scaleFactorColsPerSet);
    Value scaleA = tb.add(
        baseScaleA, tb.i32_val((m + wordIdx * numRepM) * numColPerScaleBlockA));
    Value scaleB = tb.add(
        baseScaleB, tb.i32_val((n + wordIdx * numRepN) * numColPerScaleBlockB));
    Value instDescriptor = createScaleInstDescriptor(
        rewriter, op, desc.mmaSizeM, desc.mmaSizeN, desc.transA, desc.transB,
        subWordIdx, subWordIdx, mxfpInstKind);
    createScaledGen5MMA(rewriter, loc, op, a, b, accAddress, scaleA, scaleB,
                        pred, instDescriptor, useInitAcc, desc.aInTmem,
                        mxfpInstKind);
  };

  convertDotImpl(typeConverter, rewriter, loc, op.getA(), op.getB(),
                 adaptor.getA(), adaptor.getB(), dTensorTy, adaptor.getUseD(),
                 adaptor.getPred(), adaptor.getBarriers(),
                 adaptor.getBarrierPreds(), /*twoCTAs=*/false, opKindIsMXFP4,
                 dot);
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

struct TCGen5MMAOpConversion
    : public ConvertOpToLLVMPattern<ttng::TCGen5MMAOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttng::TCGen5MMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto AEnc = op.getA().getType().getEncoding();
    auto BEnc = op.getB().getType().getEncoding();
    assert(
        (isa<NVMMASharedEncodingAttr, ttng::TensorMemoryEncodingAttr>(AEnc)) &&
        "Operand A should use Shared or Tensor memory layout.");
    assert(isa<NVMMASharedEncodingAttr>(BEnc) &&
           "Operand B should use Shared layout.");
    convertDot(*getTypeConverter(), rewriter, op.getLoc(), op, adaptor);
    rewriter.eraseOp(op);
    return success();
  }
};

struct TCGen5MMAScaledOpConversion
    : public ConvertOpToLLVMPattern<ttng::TCGen5MMAScaledOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttng::TCGen5MMAScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    convertScaledDot(*getTypeConverter(), rewriter, op.getLoc(), op, adaptor);
    rewriter.eraseOp(op);
    return success();
  }
};

struct TCGen5CommitOpConversion
    : public ConvertOpToLLVMPattern<ttng::TCGen5CommitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttng::TCGen5CommitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getBarrier(), rewriter.getI64Type(), rewriter);
    Value pred = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);

    if (adaptor.getPred())
      pred = b.and_(adaptor.getPred(), pred);

    createMMACommit(rewriter, op.getLoc(), smemObj.getBase(), pred,
                    op.getTwoCtas());
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
                                      PatternBenefit benefit) {
  patterns.add<TCGen5MMAOpConversion, TCGen5MMAScaledOpConversion,
               TCGen5CommitOpConversion>(typeConverter, benefit);
}

} // namespace NVIDIA
} // namespace triton
} // namespace mlir
