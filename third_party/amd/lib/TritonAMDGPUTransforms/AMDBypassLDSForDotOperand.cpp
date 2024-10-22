#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

//===----------------------------------------------------------------------===//
// AMDBypassLDSForDotOperandPass Overview
//
// The AMDBypassLDSForDotOperandPass implements a strategy to bypass using the
// Local Data Share (LDS) for one of the operands in an MFMA dot operation.
//
//===----------------------------------------------------------------------===//
//
// Standard Data Flow for MFMA Dot Operations:
//
// Typically, the data flow for operands in a dot operation involves three main
// steps:
//
// 1. Load Tensor from HBM to VGPRs:
//    The tensor is initially loaded into the VGPRs using a blocked (coalesced)
//    layout.
//
// 2. Write Tensor to Shared Memory (LDS):
//    This step is used for data rearrangement across threads.
//
// 3. Read Tensor from Shared Memory:
//    The tensor is read from shared memory using the dot layout, which is
//    optimized for MFMA instructions.
//
//===----------------------------------------------------------------------===//
//
// Coalescing in Triton:
//
// Coalescing in Triton is managed by configuring parameters for the blocked
// layout during the Coalesce pass. There are two primary levels of
// coalescing to consider:
//
// 1. Maximizing Load Width:
//    Achieving the widest possible loads ensures that elements are grouped into
//    larger memory transactions by the VMEM unit. This reduces the number of
//    instructions needed to load data, minimizing instruction queue size and
//    reducing wait times.
//
// 2. Ensuring Consecutive Thread Access:
//    When consecutive threads access sequential memory addresses, the TA unit
//    in the VMEM can merge multiple requests into a single, larger transaction.
//    This approach significantly boosts memory bandwidth utilization.
//
//===----------------------------------------------------------------------===//
//
// Optimizing the Data Flow:
//
// Under certain conditions, the dot layout of one of the operands allows direct
// loading from HBM to VGPRs in the MFMA dot layout, without losing level 1
// coalescing efficiency or increasing the number of global loads due to shared
// data between threads.
//
// The required conditions are:
//
// 1. K-Major (K dimension is continuous) Tensor Layout :
//    The operand we want to bypass LDS for must be K-major (i.e., row-major for
//    operand 0 or column-major for operand 1). This supports vectorized global
//    load instructions, as MFMA instructions require each thread to hold B
//    operand elements along the K dimension.
//
// 2. kWidth * sizeof(dataType) == 128:
//    Using the maximum kWidth for a specific data type ensures optimal global
//    load vectorization (e.g., using global_load_dwordx4 instructions).
//
// 3. Single Warp per CTA Dimension:
//    Either warpsPerCTA[ndim] == 1 for operand A bypass or warpsPerCTA[mDim] ==
//    1 for operand B bypass. This guarantees that each tensor element is
//    handled by exactly one thread, maintaining the same number of global loads
//    as in the blocked layout (i.e., each element is loaded only once).
//
//===----------------------------------------------------------------------===//
// Current Limitations:
// These limitations are temporary and will be addressed in future updates:
//
// 1. Support is limited to bypassing LDS for operand 1 (e.g., weights in
//    MoE-like kernels). Bypassing for operand 0 is not yet implemented.
//
// 2. LDS bypass is only supported for the fp16 data type due to the
//    kWidth == 8 condition. Other data types will be supported in the future.
//===----------------------------------------------------------------------===//

using namespace mlir;
namespace ttg = triton::gpu;

// Find all tt.load instructions that are involved in computation of a tensor
// for operand that is getting converted to dot layout.
SmallVector<triton::LoadOp> getAllLoadOpsReachingOp(Operation *op,
                                                    ModuleOp &mod) {
  SmallVector<triton::LoadOp> loadOpsVec;

  mod.walk([&](triton::LoadOp loadOp) {
    SetVector<Operation *> forwardSlices;
    getForwardSlice((Operation *)loadOp, &forwardSlices);
    if (std::find(forwardSlices.begin(), forwardSlices.end(), op) !=
        forwardSlices.end()) {
      loadOpsVec.push_back(loadOp);
    }
  });

  return loadOpsVec;
}

struct TritonAMDGPUBypassLDSForDotOperandPass
    : public TritonAMDGPUBypassLDSForDotOperandBase<
          TritonAMDGPUBypassLDSForDotOperandPass> {

  TritonAMDGPUBypassLDSForDotOperandPass() = default;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto convertOps = collectConvertOps(module);

    module.dump();

    for (ttg::ConvertLayoutOp &convertOp : convertOps) {
      auto loadInsts = getAllLoadOpsReachingOp(convertOp, module);
      assert(!loadInsts.empty());

      // Convert load instructions to dot layout.
      for (auto loadInst : loadInsts) {
        auto loadType =
            dyn_cast<RankedTensorType>(loadInst.getResult().getType());
        if (!loadType)
          return;

        auto dstType = llvm::cast<RankedTensorType>(convertOp.getType());
        auto dstDotOp =
            llvm::cast<ttg::DotOperandEncodingAttr>(dstType.getEncoding());
        convertOpEncoding(dstDotOp, loadInst);
      }
    }
  }

  SmallVector<ttg::ConvertLayoutOp> collectConvertOps(ModuleOp &module) {
    SmallVector<ttg::ConvertLayoutOp> convertOps;

    module.walk([&](ttg::ConvertLayoutOp cvtOp) {
      if (isEligibleConvertOp(cvtOp))
        convertOps.push_back(cvtOp);
    });

    return convertOps;
  }

  // Check if the required conditions and current limitations from the above doc
  // are met.
  bool isEligibleConvertOp(ttg::ConvertLayoutOp convertOp) {
    auto srcType = dyn_cast<RankedTensorType>(convertOp.getOperand().getType());
    auto dstType = dyn_cast<RankedTensorType>(convertOp.getType());

    if (!srcType || !dstType || srcType.getShape().size() != 2)
      return false;

    auto srcBlocked = dyn_cast<ttg::BlockedEncodingAttr>(srcType.getEncoding());
    auto dstDotOp =
        dyn_cast<ttg::DotOperandEncodingAttr>(dstType.getEncoding());
    if (!srcBlocked || !dstDotOp)
      return false;

    // srcBlocked.getOrder[0] == 0 is the requirement for opIdx 1 tensor to be K
    // major (required condition 1) from the above doc).
    auto mfmaLayout = dyn_cast<ttg::AMDMfmaEncodingAttr>(dstDotOp.getParent());
    return mfmaLayout && dstDotOp.getKWidth() == 8 &&
           mfmaLayout.getWarpsPerCTA()[0] == 1 && dstDotOp.getOpIdx() == 1 &&
           srcBlocked.getOrder()[0] == 0;
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUBypassLDSForDotOperand() {
  return std::make_unique<TritonAMDGPUBypassLDSForDotOperandPass>();
}
