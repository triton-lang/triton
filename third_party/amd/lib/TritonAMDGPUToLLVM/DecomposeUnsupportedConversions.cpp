#include "TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <numeric>

using namespace mlir;
using namespace mlir::triton;
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DECOMPOSEUNSUPPORTEDAMDCONVERSIONS
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

constexpr int LDSSize = 65536;
constexpr int kPtrBitWidth = 64;

// pass ws related named attrs.
static void addAttrs(Operation *op, ArrayRef<mlir::NamedAttribute> attrs) {
  for (const NamedAttribute attr : attrs)
    op->setAttr(attr.getName(), attr.getValue());
}

static void promoteReduceOpResult(OpBuilder &builder, triton::ReduceOp op,
                                  Value result, Type promotedType) {
  // save original type
  auto originalType = result.getType();
  auto elemType = originalType.isa<RankedTensorType>()
                      ? originalType.cast<RankedTensorType>().getElementType()
                      : originalType;

  // promote result type
  result.setType(promotedType);

  // set insertion point after reduce op
  builder.setInsertionPointAfter(op);

  // truncate result back to original type
  mlir::Operation *truncResult = nullptr;
  if (elemType.isInteger(16) || elemType.isInteger(8)) {
    truncResult = builder.create<mlir::arith::TruncIOp>(result.getLoc(),
                                                        originalType, result);
  } else if (elemType.isF16()) {
    truncResult = builder.create<mlir::arith::TruncFOp>(result.getLoc(),
                                                        originalType, result);
  }

  // replace all uses except for the truncOp above
  if (truncResult != nullptr) {
    result.replaceAllUsesWith(truncResult->getResult(0));
    truncResult->setOperand(0, result);
  }
}

static int getCvtOpLDSUsage(triton::gpu::ConvertLayoutOp &cvtOp) {
  unsigned inVec = 0;
  unsigned outVec = 0;
  auto smemShape = getScratchConfigForCvtLayout(cvtOp, inVec, outVec);
  unsigned elems =
      std::accumulate(smemShape.begin(), smemShape.end(), 1, std::multiplies{});
  auto srcType = cvtOp.getSrc().getType();
  auto bytes =
      srcType.getElementType().isa<triton::PointerType>()
          ? elems * kPtrBitWidth / 8
          : elems * std::max<int>(8, srcType.getElementTypeBitWidth()) / 8;

  return bytes;
}

bool isPowerOfTwo(unsigned x) { return x && (x & (x - 1)) == 0; }

static std::vector<std::pair<int, int>> factorizePowerOf2(int n) {
  assert(isPowerOfTwo(n));
  int x = log2(n);
  std::vector<std::pair<int, int>> pairs;

  for (int i = 0; i <= x / 2; ++i) {
    int j = x - i;
    pairs.push_back({pow(2, i), pow(2, j)});
    pairs.push_back({pow(2, j), pow(2, i)});
  }

  return pairs;
}

static std::pair<triton::gpu::ConvertLayoutOp, triton::gpu::ConvertLayoutOp>
createNewConvertOps(ModuleOp &mod, OpBuilder &builder,
                    triton::gpu::ConvertLayoutOp &cvtOp,
                    std::pair<unsigned, unsigned> warpsPerCta) {
  unsigned warpsPerCtaX = warpsPerCta.first;
  unsigned warpsPerCtaY = warpsPerCta.second;
  auto srcType = cvtOp.getSrc().getType();
  auto dstType = cvtOp.getType();

  auto srcMfma =
      srcType.getEncoding().dyn_cast<triton::gpu::AMDMfmaEncodingAttr>();
  auto newMfmaEnc = triton::gpu::AMDMfmaEncodingAttr::get(
      mod.getContext(), srcMfma.getVersionMajor(), srcMfma.getVersionMinor(),
      {warpsPerCtaX, warpsPerCtaY}, srcMfma.getMDim(), srcMfma.getNDim(),
      srcMfma.getIsTransposed(), srcMfma.getCTALayout());

  auto newDstType = RankedTensorType::get(
      dstType.getShape(), dstType.getElementType(), dstType.getEncoding());
  auto newSrcType = RankedTensorType::get(srcType.getShape(),
                                          srcType.getElementType(), newMfmaEnc);

  auto tmpCvt = builder.create<triton::gpu::ConvertLayoutOp>(
      cvtOp.getLoc(), newSrcType, cvtOp.getSrc());
  auto newEpilogueCvt = builder.create<triton::gpu::ConvertLayoutOp>(
      cvtOp.getLoc(), newDstType, tmpCvt);

  return std::make_pair(tmpCvt, newEpilogueCvt);
}

struct DecomposeUnsupportedAMDConversions
    : public mlir::triton::impl::DecomposeUnsupportedAMDConversionsBase<
          DecomposeUnsupportedAMDConversions> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    /* ---------------- */
    // Convert Fp8E4B15
    /* ---------------- */
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      if (!getElementTypeOrSelf(cvtOp)
               .isa<mlir::Float8E4M3B11FNUZType, mlir::Float8E4M3FNType>())
        return;
      auto shape = cvtOp.getType().cast<RankedTensorType>().getShape();
      auto argEncoding =
          cvtOp.getSrc().getType().cast<RankedTensorType>().getEncoding();
      auto cvtEncoding = cvtOp.getType().cast<RankedTensorType>().getEncoding();
      if (argEncoding.isa<triton::gpu::DotOperandEncodingAttr>() ||
          cvtEncoding.isa<triton::gpu::DotOperandEncodingAttr>())
        return;
      auto F16Ty = builder.getF16Type();

      auto newArgType = RankedTensorType::get(shape, F16Ty, argEncoding);
      auto newCvtType = RankedTensorType::get(shape, F16Ty, cvtEncoding);
      auto newArg = builder.create<mlir::triton::FpToFpOp>(
          cvtOp.getLoc(), newArgType, cvtOp.getSrc());
      addAttrs(newArg, cvtOp->getAttrs());
      auto newCvt = builder.create<mlir::triton::gpu::ConvertLayoutOp>(
          cvtOp.getLoc(), newCvtType, newArg);
      addAttrs(newCvt, cvtOp->getAttrs());
      auto newRet = builder.create<mlir::triton::FpToFpOp>(
          cvtOp.getLoc(), cvtOp.getType(), newCvt.getResult());
      newRet.setRounding(
          triton::RoundingMode::RTNE); // Downcast requires rounding mode
      addAttrs(newRet, cvtOp->getAttrs());
      cvtOp.replaceAllUsesWith(newRet.getResult());
      cvtOp.erase();
    });
    /* -------------------------------- */
    // Replace `splat -> shared
    // with `splat -> blocked -> shared
    /* -------------------------------- */
    mod.walk([&](triton::SplatOp splatOp) -> void {
      auto dstType = splatOp.getType();
      auto shared =
          dstType.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
      if (shared) {
        OpBuilder builder(splatOp);
        SmallVector<unsigned, 4> sizePerThread(dstType.getRank(), 1);
        auto newType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), dstType.getShape(), sizePerThread,
                getOrder(shared), numWarps, threadsPerWarp, numCTAs));
        auto newSplat = builder.create<triton::SplatOp>(
            splatOp.getLoc(), newType, splatOp.getSrc());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            splatOp.getLoc(), dstType, newSplat.getResult());
        splatOp.replaceAllUsesWith(newConvert.getResult());
        splatOp.erase();
      }
    });
    /* -------------------------------- */
    // Replace `mfma -> dot_op` with `mfma -> blocked -> dot_op`
    // unless certain conditions are met
    /* -------------------------------- */
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getSrc().getType();
      auto dstType = cvtOp.getType();
      auto srcMfma =
          srcType.getEncoding().dyn_cast<triton::gpu::AMDMfmaEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcMfma && dstDotOp && !isMfmaToDotShortcut(srcType, dstType)) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), srcType.getShape(), getSizePerThread(srcMfma),
                getOrder(srcMfma), numWarps, threadsPerWarp, numCTAs));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getSrc());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
    // Try to reduce LDS usage of cvt(mfma->blocked) op by changing the shape of
    // WarpsPerCta attribute in mfma layout. The implicit LDS usage of
    // cvt(mfma->blocked) op depends on the number of warps per CTA that mfma
    // layout uses along x dimension and block layout uses across y dimension.
    //
    // clang-format off
    //
    // LDS usage of this op is roughly calculated as:
    // LDS_USAGE = getShapePerCTA(mfma_layout)[0] * getShapePerCTA(blocked_layout)[1] * sizeof(data_type)
    // LDS_USAGE = warpsPerCTA(mfma_layout)[0] * warpsPerCta(blocked_layout)[1] * C,
    // where C = 32 * sizePerWarp(blocked_layout)[1] * threadsPerWarp(blocked_layout)[1] * sizeof(data_type)
    //
    // clang-format on
    //
    // When LDS_USAGE exceeds the size of LDS, try to lower LDS usage by
    // decomposing cvt(mfma->blocked) op into 2 conversions: cvt(mfma->mfma_tmp)
    // and cvt(mfma_tmp->blocked), where mfma_tmp has WarpsPerCta attribute that
    // minimizes uses of LDS for these conversions.
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);

      auto srcType = cvtOp.getSrc().getType();
      auto dstType = cvtOp.getType();

      auto srcMfma =
          srcType.getEncoding().dyn_cast<triton::gpu::AMDMfmaEncodingAttr>();
      auto dstBlocked =
          dstType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();

      if (!srcMfma || !dstBlocked) {
        return;
      }

      auto currLDSUsage = getCvtOpLDSUsage(cvtOp);
      if (currLDSUsage <= LDSSize) {
        return;
      }

      unsigned numWarps =
          srcMfma.getWarpsPerCTA()[0] * srcMfma.getWarpsPerCTA()[1];

      triton::gpu::ConvertLayoutOp tmpCvt;
      triton::gpu::ConvertLayoutOp newEpilogueCvt;

      // Find all possible shapes of WarpsPerCTA by finding all possible
      // factorizations of numWarps. Pick shape for which both conversions in
      // decomposition use LDS less than LDSSize and for which sum of LDS usage
      // is minimal. If no such shape exists, do not decompose.
      unsigned minLDSUsage = 2 * LDSSize;
      int minIdx = -1;
      auto factorizedNumWarps = factorizePowerOf2(numWarps);

      for (int i = 0; i < factorizedNumWarps.size(); i++) {
        auto warpsPerCTAPair = factorizedNumWarps[i];
        std::tie(tmpCvt, newEpilogueCvt) =
            createNewConvertOps(mod, builder, cvtOp, warpsPerCTAPair);

        int tmpCvtLDS = getCvtOpLDSUsage(tmpCvt);
        int newCvtLDS = getCvtOpLDSUsage(newEpilogueCvt);
        if (tmpCvtLDS <= LDSSize && newCvtLDS <= LDSSize) {
          int LDSUsage = tmpCvtLDS + newCvtLDS;
          if (LDSUsage < minLDSUsage) {
            minLDSUsage = LDSUsage;
            minIdx = i;
          }
        }
        newEpilogueCvt.erase();
        tmpCvt.erase();
      }

      if (minIdx == -1) {
        return;
      }

      assert(minIdx >= 0 && minIdx < factorizedNumWarps.size());
      auto warpsPerCTAPair = factorizedNumWarps[minIdx];
      std::tie(tmpCvt, newEpilogueCvt) =
          createNewConvertOps(mod, builder, cvtOp, warpsPerCTAPair);

      cvtOp.replaceAllUsesWith(newEpilogueCvt.getResult());
      cvtOp.erase();
    });
    /* -------------------------------- */
    // Replace `blocked -> dot_op` with `blocked -> shared -> dot_op`
    // because the codegen doesn't handle `blocked -> dot_op` directly
    /* -------------------------------- */
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getSrc().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcBlocked =
          srcType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcBlocked && dstDotOp) {
        auto tmpType = MemDescType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::SharedEncodingAttr::get(
                mod.getContext(), dstDotOp, srcType.getShape(),
                srcBlocked.getOrder(), srcBlocked.getCTALayout(),
                srcType.getElementType()));
        auto tmp = builder.create<triton::gpu::LocalAllocOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getSrc());
        addAttrs(tmp, cvtOp->getAttrs());
        auto newConvert = builder.create<triton::gpu::LocalLoadOp>(
            cvtOp.getLoc(), dstType, tmp);
        addAttrs(newConvert, cvtOp->getAttrs());
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });

    // promote reduce ops
    mod.walk([&](triton::ReduceOp op) -> void {
      OpBuilder builder(op);

      // promote operands
      SmallVector<Value> newOperands;
      for (OpOperand &operand : op->getOpOperands()) {
        auto val = operand.get();
        auto oldType = val.getType().cast<RankedTensorType>();
        auto elemType = oldType.getElementType();
        if (elemType.isInteger(16) || elemType.isInteger(8)) {
          auto newType =
              oldType.cloneWith(std::nullopt, builder.getIntegerType(32));
          auto promotedVal =
              builder.create<mlir::arith::ExtSIOp>(op->getLoc(), newType, val);
          newOperands.push_back(promotedVal);
        } else if (elemType.isF16()) {
          auto newType = oldType.cloneWith(std::nullopt, builder.getF32Type());
          auto promotedVal =
              builder.create<mlir::arith::ExtFOp>(op->getLoc(), newType, val);
          newOperands.push_back(promotedVal);
        } else {
          newOperands.push_back(val);
        }
      }
      op->setOperands(newOperands);

      // promote results
      for (Value result : op.getResults()) {
        auto type = result.getType();
        if (type.isInteger(16) || type.isInteger(8)) {
          promoteReduceOpResult(builder, op, result,
                                builder.getIntegerType(32));
        } else if (type.isF16()) {
          promoteReduceOpResult(builder, op, result, builder.getF32Type());
        } else if (type.isa<RankedTensorType>()) {
          auto oldType = type.cast<RankedTensorType>();
          auto elemType = oldType.getElementType();
          if (elemType.isInteger(16) || elemType.isInteger(8)) {
            promoteReduceOpResult(
                builder, op, result,
                oldType.cloneWith(std::nullopt, builder.getIntegerType(32)));
          } else if (elemType.isF16()) {
            promoteReduceOpResult(
                builder, op, result,
                oldType.cloneWith(std::nullopt, builder.getF32Type()));
          }
        }
      }

      // promote combine op
      for (Block &oldBlock : op.getCombineOp().getBlocks()) {
        // update block args
        for (auto arg : oldBlock.getArguments()) {
          auto type = arg.getType();
          if (type.isInteger(16) || type.isInteger(8)) {
            arg.setType(builder.getIntegerType(32));
          } else if (type.isF16()) {
            arg.setType(builder.getF32Type());
          }
        }

        for (Operation &oldOp : oldBlock.getOperations()) {
          // update operands
          for (OpOperand &operand : oldOp.getOpOperands()) {
            auto val = operand.get();
            auto type = val.getType();
            if (type.isInteger(16) || type.isInteger(8)) {
              val.setType(builder.getIntegerType(32));
            } else if (type.isF16()) {
              val.setType(builder.getF32Type());
            }
          }

          // update results
          for (Value result : oldOp.getResults()) {
            auto type = result.getType();
            if (type.isInteger(16) || type.isInteger(8)) {
              result.setType(builder.getIntegerType(32));
            } else if (type.isF16()) {
              result.setType(builder.getF32Type());
            }
          }
        }
      }
    });
  }
};

} // namespace

namespace mlir {

namespace triton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>>
createDecomposeUnsupportedAMDConversionsPass() {
  return std::make_unique<DecomposeUnsupportedAMDConversions>();
}

} // namespace gpu

} // namespace triton

} // namespace mlir
