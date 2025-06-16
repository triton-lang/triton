/*
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <memory>

#define GEN_PASS_CLASSES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-aref-async-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::nvws;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
using namespace ttg;

void arefReuse(triton::FuncOp funcOp) {
  std::map<std::string, SmallVector<scf::ForOp>> forOps;
  funcOp.walk([&](ttng::WarpGroupOp wgOp) {
    auto group = getGroup(wgOp);
    if (group == "nvws.group.mma1") {
      wgOp.walk([&](scf::ForOp forOp) { forOps[group].push_back(forOp); });
    } else {
      return WalkResult::advance();
    }
    for (auto [group, forOps] : forOps) {
      if (forOps.size() != 2)
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (forOps.empty())
    return;
  if (forOps.at("nvws.group.mma1").size() != 2)
    return;

  SmallVector<Value> arefs_mma1[2];
  for (auto [i, forOp] : llvm::enumerate(forOps.at("nvws.group.mma1"))) {
    auto idx = i;
    forOp.walk([&](ttng::ArefPutEnterOp enterOp) {
      arefs_mma1[idx].push_back(enterOp.getAref());
    });
  }
  if (arefs_mma1[0].size() != 1)
    return;
  if (arefs_mma1[1].size() != 1)
    return;
  assert(arefs_mma1[0].size() == 1);
  assert(arefs_mma1[1].size() == 1);

  // reuse arefs from 1st loop in the 2nd loop
  arefs_mma1[1][0].replaceAllUsesWith(arefs_mma1[0][0]);

  arefs_mma1[1][0].getDefiningOp()->erase();
}
void subsliceFACorrection(tt::FuncOp funcOp) {
  SmallVector<SmallVector<Operation *>> tileList;
  
  // Find load-multiply-store patterns
  funcOp.walk([&](ttng::TMEMLoadOp loadOp) {
    // Skip if no token or token has multiple uses
    if (!loadOp.getToken() || !loadOp.getToken().hasOneUse())
      return;
      
    // Get store op from token user
    auto storeOp = dyn_cast<ttng::TMEMStoreOp>(*loadOp.getToken().getUsers().begin());
    if (!storeOp || !storeOp.getToken() || !storeOp.getToken().use_empty())
      return;

    // Get multiply op from load result user
    auto val = loadOp.getResult();
    if (!val.hasOneUse())
      return;
    auto mulf = dyn_cast<arith::MulFOp>(*val.getUsers().begin());
    if (!mulf || !mulf.getResult().hasOneUse())
      return;

    // Check multiply result feeds into store
    if (*mulf.getResult().getUsers().begin() == storeOp) {
      tileList.push_back({loadOp, mulf, storeOp});
    }
  });

  // Process each tile
  for (auto tile : tileList) {
    auto load = cast<ttng::TMEMLoadOp>(tile[0]);
    auto mulf = cast<arith::MulFOp>(tile[1]); 
    auto store = cast<ttng::TMEMStoreOp>(tile[2]);

    auto inpShape = load.getSrc().getType().getShape();
    assert(inpShape.size() == 2);

    OpBuilder b(load);
    const int splitNSize = 16;
    Value tmem = load.getSrc();
    assert(inpShape[1] % splitNSize == 0);

    // Create new shapes
    SmallVector<int64_t> shape1(inpShape);
    shape1[1] = splitNSize;

    int mDim = getShapePerCTA(load.getSrc().getType())[0];
    if (mDim != 128)
      return;

    // Get types for load and broadcast
    auto numWarps = ttg::lookupNumWarps(load);
    int threadsPerWarp = ttg::lookupThreadsPerWarp(b);
    RankedTensorType loadType, bcastType;
    {
      auto encoding = getDefaultBlockedEncoding(load.getContext(), inpShape,
                                              numWarps, threadsPerWarp, 1);
      RankedTensorType tensorType = RankedTensorType::get(
          shape1, load.getType().getElementType(), encoding);
      Attribute newDistributedEncoding =
          ttng::getTmemCompatibleLayout(mDim, shape1[1], tensorType, numWarps);
      loadType = RankedTensorType::get(shape1, load.getType().getElementType(),
                                      newDistributedEncoding);
      shape1[1] = 1;
      bcastType = RankedTensorType::get(shape1, load.getType().getElementType(),
                                       newDistributedEncoding);
    }

    // Get broadcast op
    auto scaleOpnd = mulf.getLhs() == load.getResult() ? 1 : 0;
    auto bcast = dyn_cast<tt::BroadcastOp>(mulf.getOperand(scaleOpnd).getDefiningOp());
    if (!bcast)
      return;

    // Create new ops
    auto cvt0 = b.create<ttg::ConvertLayoutOp>(bcast.getLoc(), bcastType,
                                              bcast.getSrc());
    auto bcast1 = b.create<tt::BroadcastOp>(bcast.getLoc(), loadType, cvt0.getResult());

    // Process each slice
    for (int i = 0; i < inpShape[1]; i += splitNSize) {
      Value slice = b.create<ttng::TMEMSubSliceOp>(load.getLoc(), tmem, i, splitNSize);
      auto load1 = b.create<ttng::TMEMLoadOp>(load.getLoc(), loadType, slice);
      auto mulf1 = b.create<arith::MulFOp>(mulf.getLoc(), load1, bcast1);
      b.create<ttng::TMEMStoreOp>(store.getLoc(), b.getType<AsyncTokenType>(),
                                 slice, load1.getToken(), mulf1.getResult(),
                                 store.getPred());
    }

    // Cleanup old ops
    store->erase();
    mulf->erase();
    load->erase();
    bcast->erase();
  }
}

class NVWSBlackwellFA : public NVWSBlackwellFABase<NVWSBlackwellFA> {

public:
  void runOnFunc(triton::FuncOp funcOp) {
    arefReuse(funcOp);
    LLVM_DEBUG({ DBGS() << "after::arefReuse:\n" << funcOp << "\n"; });

    subsliceFACorrection(funcOp);
    LLVM_DEBUG({ DBGS() << "after::subsliceFACorrection:\n" << funcOp << "\n"; });
  }

  void runOnOperation() override {
    auto mod = getOperation();
    mod.walk([&](triton::FuncOp funcOp) { runOnFunc(funcOp); });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createNVWSBlackwellFAPass() {
  return std::make_unique<NVWSBlackwellFA>();
}
