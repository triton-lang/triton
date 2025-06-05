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
#include "mlir/IR/Verifier.h"
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

// #define GEN_PASS_CLASSES
// #include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define GEN_PASS_CLASSES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-aref-copy-elimination"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::gpu;
using namespace triton::nvws;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

Operation *getParentInSameBlock(Block *targetBlock, Operation *op) {
  Operation *current = op;

  while (current) {
    if (current->getBlock() == targetBlock)
      return current;
    current = current->getParentOp();
  }

  return nullptr; // op2 is not nested within op1's block
}

SetVector<Operation *> findCopyConsumers(Operation *op) {
  // if user output is not memDescType, we assume it is consumer of copyOp
  // otherwise traverse to the next op
  SetVector<Operation *> opConsumers;
  for (auto user : op->getUsers()) {
    if (llvm::count_if(user->getResults(), [](auto res) {
          return isa<MemDescType>(res.getType());
        }) == 0) {
      opConsumers.insert(user);
    } else {
      auto consumers = findCopyConsumers(user);
      opConsumers.insert(consumers.begin(), consumers.end());
    }
  }
  return opConsumers;
}

enum class BlockScope {
  UNSUPPORTED,
  SAME_BLOCK,
  NESTED_INSIDE,
};

BlockScope getBlockScope(Block *from, Block *to) {
  if (from == to)
    return BlockScope::SAME_BLOCK;

  auto block = to;
  while (block && block != from)
    block = block->getParentOp()->getBlock();
  if (block == from)
    return BlockScope::NESTED_INSIDE;

  return BlockScope::UNSUPPORTED;
}

ttng::ArefConsumer getConsumerKind(Operation *copyOrCloneOp) {
  if (auto copyOp = dyn_cast<ttng::ArefCopyOp>(copyOrCloneOp)) {
    auto consumer = *copyOp.getToken().getUsers().begin();
    if (isa<ttng::TMEMLoadOp>(consumer)) {
      return ttng::ArefConsumer::LDTM;
    } else if (auto mmav5 = dyn_cast<ttng::MMAv5OpInterface>(consumer)) {
      return ttng::ArefConsumer::UMMA;
    }
  } else {
    auto cloneOp = cast<ttng::ArefCloneOp>(copyOrCloneOp);
    auto consumers = findCopyConsumers(cloneOp);
    assert(!consumers.empty());
    auto consumer = consumers.front();
    if (isa<ttg::LocalLoadOp>(consumer)) {
      return ttng::ArefConsumer::LDS;
    } else if (isa<ttng::WarpGroupDotOp>(consumer)) {
      return ttng::ArefConsumer::WGMMA;
    } else if (auto mmav5 = dyn_cast<ttng::MMAv5OpInterface>(consumer)) {
      return ttng::ArefConsumer::UMMA;
    }
  }
  // consumer is unknown to copy elimination
  return ttng::ArefConsumer::NONE;
}

ttng::ArefProducer getProducerKind(ttng::ArefCopyOp &copyOp) {
  if (copyOp.getSrcDep()) {
    if (auto producer = getTokenProducerOp(copyOp.getSrcDep()).producerOp) {
      if (isa<ttng::TMEMStoreOp>(producer)) {
        return ttng::ArefProducer::STTM;
      } else if (isa<ttng::MMAv5OpInterface>(producer)) {
        return ttng::ArefProducer::UMMA;
      }
    }
  } else {
    auto src = copyOp.getSrc();
    auto producer = src.getDefiningOp();
    if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(producer)) {
      // only handle tmem_alloc with an operand for now as it implies tmem_Store
      // tmem_alloc w/o operand doesn't have a store semantics and will be
      // caught the NONE case so we know something is off
      if (tmemAlloc.getSrc())
        return ttng::ArefProducer::STTM;
    }
    Value localStoreSrc;
    if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(producer)) {
      localStoreSrc = localAlloc.getSrc();
    } else if (auto localStore = dyn_cast<ttg::LocalStoreOp>(producer)) {
      localStoreSrc = localStore.getSrc();
    }
    if (localStoreSrc) {
      if (isa<triton::DescriptorLoadOp, triton::DescriptorGatherOp>(
              localStoreSrc.getDefiningOp())) {
        return ttng::ArefProducer::TMALDG;
      } else if (isa<tt::LoadOp>(localStoreSrc.getDefiningOp())) {
        return ttng::ArefProducer::LDGSTS;
      }
      return ttng::ArefProducer::STS;
    }
  }
  // the producer is unknown to copy elimination
  return ttng::ArefProducer::NONE;
}

void removeConsumerCopyOp(Operation *copyOrCloneOp, Operation *lastConsumer) {
  assert((isa<ttng::ArefCopyOp, ttng::ArefCloneOp>(copyOrCloneOp)));
  auto getEnterOp =
      copyOrCloneOp->getOperand(0).getDefiningOp<ttng::ArefGetEnterOp>();
  auto arefTag = getEnterOp->getAttrOfType<StringAttr>("aref_tag").str();
  ttng::ArefGetExitOp getExitOp;
  for (auto user : getEnterOp.getAref().getUsers())
    if (auto exitOp = dyn_cast<ttng::ArefGetExitOp>(user))
      if (exitOp->getAttrOfType<StringAttr>("aref_tag").str() == arefTag) {
        getExitOp = exitOp;
        break;
      }
  assert(getExitOp);
  auto consumerKind = getConsumerKind(copyOrCloneOp);

  // if we don't know consumer, we don't removeConsumerCopyOP
  if (consumerKind == ttng::ArefConsumer::NONE)
    return;

  SmallVector<Attribute> consumerAttr{
      ttng::ArefConsumerAttr::get(getExitOp.getContext(), consumerKind)};
  getExitOp.setConsumersAttr(
      ArrayAttr::get(getExitOp.getContext(), consumerAttr));
  auto moveGetExitAfter = [](ttng::ArefGetExitOp getExitOp,
                             Operation *lastConsumer) {
    getExitOp->moveAfter(lastConsumer);
    OpBuilder builder(getExitOp);
    getExitOp.getIndexMutable().assign(
        mkConstant(builder, getExitOp.getLoc(), 0, 32, getGroups(getExitOp)));
  };

  auto consumerScope =
      getBlockScope(copyOrCloneOp->getBlock(), lastConsumer->getBlock());
  Operation *allocOp = {};
  if (BlockScope::SAME_BLOCK == consumerScope) {
    // if consumer and copy are in the same block, can move getExitOp
    // after consumer and reuse buffer

    //      %srcBuf = aref_get.enter %aref
    //      %copyBuf = aref_clone %srcBuf
    //      aref_get.exit %aref
    //         .. use %copyBuf
    //
    //   becomes
    //
    //      %srcBuf = aref_get.enter %aref
    //         .. use %srcBuf
    //      aref_get.exit %aref

    if (auto copyOp = dyn_cast<ttng::ArefCopyOp>(copyOrCloneOp)) {
      allocOp = copyOp.getDst().getDefiningOp();
      assert(isa<ttng::TMEMAllocOp>(allocOp));

      // Skip copy elimination from get.enter for MMAv5 operations since this op
      // both reads and updates its accumulator. This must be coordinated with
      // its corresponding put operations. For example, in Flash Attention,
      // MMAv5 operations have paired put/get operations that must be processed
      // together. This is currently handled as a specialized optimization in
      // the BlackwellFA pass.
      // TODO: Develop a more general approach for handling MMAv5 copy
      // elimination.
      if (isa<ttng::MMAv5OpInterface>(lastConsumer))
        return;

      // update token/buffer returned with get
      // XXX: this could lead in interesting behaviour, when copyOp is
      //      is nested in a region_op, and tmem_alloc buffer used outside
      //   %buf, %tok = tmem_alloc ()
      //   region_op {
      //      %srcBuf, %srcTok = aref_get.enter %aref
      //      %copyTok = aref_copy %srcBuf[%srcTok], %buf
      //      aref_get.exit %aref
      //      .. use %buf
      //   }
      //   .. use %buf
      //
      //  it becomes after removeCopyOp
      //   region_op {
      //      %srcBuf, %srcTok = aref_get.enter %aref
      //       .. %use %srcBuf
      //      aref_get.exit %aref
      //   }
      //   .. use %srcBuf
      //
      //   2nd use of %srcBuf outside region_op is invalid
      //   correct way would be to propagate %srcBuf outside region, and
      //   find different place to put aref_get.exit %aref, if possible
      //   if not, copy cannot be eliminated, and we just keep it
      //
      //   we don't have use cases like that today
      copyOp.getToken().replaceAllUsesWith(copyOp.getSrcDep());
      copyOp.getDst().replaceUsesWithIf(
          copyOp.getSrc(), [&](OpOperand &use) -> bool {
            auto blockScopeCopyToUse =
                getBlockScope(copyOp->getBlock(), use.getOwner()->getBlock());
            if (blockScopeCopyToUse == BlockScope::SAME_BLOCK) {
              // use is either last consumer, or is in between copy and last
              // consumer
              return use.getOwner() == lastConsumer ||
                     (use.getOwner()->isBeforeInBlock(lastConsumer) &&
                      copyOp->isBeforeInBlock(use.getOwner()));
            } else if (blockScopeCopyToUse == BlockScope::NESTED_INSIDE) {
              // if nested inside block, ensure that use is between enter & exit
              auto parentOp =
                  getParentInSameBlock(getEnterOp->getBlock(), use.getOwner());
              return parentOp->isBeforeInBlock(getExitOp) &&
                     getEnterOp->isBeforeInBlock(parentOp);
            }
            return false;
          });
    } else {
      auto cloneOp = cast<ttng::ArefCloneOp>(copyOrCloneOp);
      cloneOp.getResult().replaceAllUsesWith(cloneOp.getSrc());
    }
    moveGetExitAfter(getExitOp, lastConsumer);
  } else if (BlockScope::NESTED_INSIDE == consumerScope) {
    // When the consumer is nested inside a block containing the copy op,
    // we need to move getExitOp after the outermost parent operation in
    // the copy's block that contains the consumer.

    //      %srcBuf = aref_get.enter %aref
    //      %copyBuf = aref_clone srcBuf
    //      aref_get.exit %aref
    //      region_op {
    //         .. use %copyBuf
    //      }
    //
    //   becomes
    //
    //      %srcBuf = aref_get.enter %aref
    //      region_op {
    //         .. use %srcBuf
    //      }
    //      aref_get.exit %aref

    // find region_op of last consumer that is in the same block as copyOp
    auto regionOp = lastConsumer->getParentOp();
    while (regionOp->getBlock() != copyOrCloneOp->getBlock())
      regionOp = regionOp->getParentOp();

    if (auto copyOp = dyn_cast<ttng::ArefCopyOp>(copyOrCloneOp)) {
      allocOp = copyOp.getDst().getDefiningOp();
      assert(isa<ttng::TMEMAllocOp>(allocOp));
      copyOp.getToken().replaceAllUsesWith(copyOp.getSrcDep());
      copyOp.getDst().replaceAllUsesWith(copyOp.getSrc());
    } else {
      auto cloneOp = cast<ttng::ArefCloneOp>(copyOrCloneOp);
      cloneOp.getResult().replaceAllUsesWith(cloneOp.getSrc());
    }
    // move exitOp right after the region_op
    moveGetExitAfter(getExitOp, regionOp);
  } else {
    llvm_unreachable("unsupported copy elimination case for consumer side");
  }

  // remove copyOp and allocOp
  copyOrCloneOp->erase();
  if (allocOp && allocOp->use_empty())
    allocOp->erase();
}

void removeProducerCopyOp(ttng::ArefCopyOp &copyOp) {
  auto putEnterOp = cast<ttng::ArefPutEnterOp>(copyOp.getDst().getDefiningOp());
  ttng::ArefPutExitOp putExitOp;
  auto arefTag = putEnterOp->getAttrOfType<StringAttr>("aref_tag").str();
  for (auto user : putEnterOp.getAref().getUsers())
    if (auto exitOp = dyn_cast<ttng::ArefPutExitOp>(user))
      if (exitOp->getAttrOfType<StringAttr>("aref_tag").str() == arefTag) {
        putExitOp = exitOp;
        break;
      }

  assert(putExitOp);
  auto producerKind = getProducerKind(copyOp);
  if (producerKind == ttng::ArefProducer::NONE)
    return;

  SmallVector<Attribute> producerAttr{
      ttng::ArefProducerAttr::get(putExitOp.getContext(), producerKind)};
  putExitOp.setProducersAttr(
      ArrayAttr::get(putExitOp.getContext(), producerAttr));
  auto movePutEnterBefore = [](ttng::ArefPutEnterOp putEnterOp,
                               Operation *producer) {
    putEnterOp->moveBefore(producer);
    OpBuilder builder(putEnterOp);
    putEnterOp.setIndex(
        mkConstant(builder, putEnterOp.getLoc(), 0, 32, getGroups(putEnterOp)));
  };

  if (auto tok = copyOp.getToken()) {
    if (!tok.use_empty()) {
      assert(copyOp.getSrcDep());
      if (!tok.hasOneUse())
        return;
      // If there is a use of the token, try to move that use before copyOp
      // prior to eliminating copyOp. This ensures exitOp will be after the use.
      auto &use = *tok.getUses().begin();
      auto user = use.getOwner();
      if (getBlockScope(copyOp->getBlock(), user->getBlock()) !=
          BlockScope::SAME_BLOCK) {
        return;
      }
      if (auto tmemLoadOp = dyn_cast<ttng::TMEMLoadOp>(user)) {
        tmemLoadOp.getDepMutable().assign(copyOp.getSrcDep());
        tmemLoadOp->moveBefore(copyOp);
      } else if (auto tmemStoreOp = dyn_cast<ttng::TMEMStoreOp>(user)) {
        tmemStoreOp.getDepMutable().assign(copyOp.getSrcDep());
        tmemStoreOp->moveBefore(copyOp);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        yieldOp.setOperand(use.getOperandNumber(), copyOp.getSrcDep());
      } else {
        // unsupport use of token, we don't eliminate this copy
        return;
      }
    }
  }

  if (copyOp.getSrcDep()) {
    // we store to mutable memory,
    // assume that buffer comes from tmem_alloc
    auto tmemAllocOp = cast<ttng::TMEMAllocOp>(copyOp.getSrc().getDefiningOp());
    assert(tmemAllocOp.getToken() && "token expected");
    if (!tmemAllocOp.getToken().hasOneUse())
      return;
    assert(tmemAllocOp.getToken().hasOneUse() &&
           "token is expected to have one use");

    auto producer = *tmemAllocOp.getToken().getUsers().begin();
    auto copyScope = getBlockScope(producer->getBlock(), copyOp->getBlock());
    if (BlockScope::SAME_BLOCK == copyScope) {
      // if same block, put putEnterOp just before producer
      //
      //    %tok2  = store/update %buf[%tok1]
      //    ..
      //    %dstBuf, %dstTok =  aref_put.enter %aref
      //    aref_copy %buif[%tok2], %dstBuf[%dstTok]
      //    aref_put.exit %aref
      //
      //  becomes
      //
      //    %dstBuf, %dstTok =  aref_put.enter %aref
      //    %tok2  = store/update %dstBuf[%dstTok]
      //    aref_put.exit %aref
      //
      //  or the following, despite update is done inside region_op
      //  the first producer is actually iterArgs of for-op
      //
      //   %buf, %tok = tmem_alloc ()
      //   %tok2 = region_op .., %tok1 = %tok{
      //        ..
      //       %tok2 = store/update %buf[%tok1]
      //       ..
      //       yield .. %tok2
      //    }
      //    %dstBuf, %dstTok = aref_put.enter %aref
      //    aref_copy %buf[%tok2], %dstBuf[%dstTok]
      //    aref_put.exit %aref
      //
      //  becomes
      //
      //    %dstBuf, %dstTok = aref_put.enter %aref
      //

      movePutEnterBefore(putEnterOp, producer);
      assert(copyOp.getDstDep());
      tmemAllocOp.getToken().replaceAllUsesWith(copyOp.getDstDep());
      copyOp.getSrc().replaceUsesWithIf(
          copyOp.getDst(), [&](OpOperand &use) -> bool {
            auto blockScopeCopyToUse =
                getBlockScope(copyOp->getBlock(), use.getOwner()->getBlock());
            if (blockScopeCopyToUse == BlockScope::SAME_BLOCK) {
              // use is either producer, or is in between producer and copy
              return use.getOwner() == producer ||
                     (producer->isBeforeInBlock(use.getOwner()) &&
                      use.getOwner()->isBeforeInBlock(copyOp));
            } else if (blockScopeCopyToUse == BlockScope::NESTED_INSIDE) {
              // if nested inside block, sure that use is between enter & exit
              auto parentOp =
                  getParentInSameBlock(putEnterOp->getBlock(), use.getOwner());
              return parentOp->isBeforeInBlock(putExitOp) &&
                     putEnterOp->isBeforeInBlock(parentOp);
            }
            return false;
          });
    } else {
      return;
    }
    copyOp->erase();
    if (tmemAllocOp->use_empty())
      tmemAllocOp->erase();
  } else {
    // if there is no token, we assume producer comes from
    // local/tmem_alloc with src operand, immutable buffer

    auto rewriteProducer = [&](Operation *producer) {
      if (auto localAllocOp = dyn_cast<LocalAllocOp>(producer)) {
        // if immutable producer is localAllocOp, we replace it with
        // localStore
        OpBuilder builder(localAllocOp);
        assert(localAllocOp.getSrc());
        builder.create<LocalStoreOp>(localAllocOp.getLoc(),
                                     localAllocOp.getSrc(), copyOp.getDst());
      } else if (auto tmemAllocOp = dyn_cast<ttng::TMEMAllocOp>(producer)) {
        // if immutable producer is tmemAllocOp, we replace it with
        // tmemStore
        OpBuilder builder(tmemAllocOp);
        assert(tmemAllocOp.getSrc());
        auto trueVal = mkConstant(builder, tmemAllocOp->getLoc(), 1, 1, {});
        builder.create<ttng::TMEMStoreOp>(tmemAllocOp.getLoc(), Type{},
                                          copyOp.getDst(), Value{},
                                          tmemAllocOp.getSrc(), trueVal);

      } else {
        llvm_unreachable("unsupported immutable producer");
      }
    };

    auto producer = copyOp.getSrc().getDefiningOp();
    auto copyScope = getBlockScope(producer->getBlock(), copyOp->getBlock());
    if (BlockScope::SAME_BLOCK == copyScope) {
      // if producer and copy are in the same block, move putEnterOp
      // before producer
      //
      //   %buf = alloc %value
      //   ..
      //   %dst = aref_put.enter %aref
      //   aref_copy %buf, %dst
      //   aref_put.exit %aref
      //
      //  becomes
      //
      //   %dst = aref_put.enter %aref
      //   store %value, %dst
      //   ..
      //   aref_put.exit %aref
      //
      movePutEnterBefore(putEnterOp, producer);
      rewriteProducer(producer);
      copyOp->erase();
      producer->erase();
    } else {
      llvm_unreachable("unsupported immutable copy elimination case for "
                       "producer side");
    }
  }
}

void removeCopyOrCloneOp(Operation *op,
                         DenseMap<Operation *, int> const &opOrdering) {

  auto mod = op->getParentOfType<ModuleOp>();
  if (isa<ttng::ArefGetEnterOp>(op->getOperand(0).getDefiningOp())) {
    auto allConsumers = findCopyConsumers(op);
    // Filter consumers that are in the same block or nested blocks
    SetVector<Operation *> validConsumers;
    for (auto consumer : allConsumers) {
      if (getBlockScope(op->getBlock(), consumer->getBlock()) !=
          BlockScope::UNSUPPORTED)
        validConsumers.insert(consumer);
    }
    if (validConsumers.empty())
      return;

    auto lastConsumer = *llvm::max_element(validConsumers, [&](auto a, auto b) {
      return opOrdering.at(a) < opOrdering.at(b);
    });
    removeConsumerCopyOp(op, lastConsumer);
  } else {
    auto copyOp = cast<ttng::ArefCopyOp>(op);
    removeProducerCopyOp(copyOp);
  }
}

void removeArefCopiesFromWg(ttng::WarpGroupOp wg) {
  SmallVector<Operation *> ops;
  wg->walk([&](Operation *op) {
    if (isa<ttng::ArefCopyOp, ttng::ArefCloneOp>(op))
      ops.push_back(op);
  });

  DenseMap<Operation *, int> opOrdering;
  wg->walk([&](Operation *op) { opOrdering[op] = opOrdering.size(); });

  for (auto op : ops)
    removeCopyOrCloneOp(op, opOrdering);
}

class NVWSArefCopyElimination
    : public NVWSArefCopyEliminationBase<NVWSArefCopyElimination> {
public:
  void runOnFunc(triton::FuncOp funcOp) {
    SmallVector<ttng::WarpGroupOp> wgOps;
    funcOp.walk([&](ttng::WarpGroupOp wgOp) { wgOps.push_back(wgOp); });

    for (auto wg : wgOps)
      removeArefCopiesFromWg(wg);
  }
  void runOnOperation() override {
    auto mod = getOperation();
    mod.walk([&](triton::FuncOp funcOp) { runOnFunc(funcOp); });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createNVWSArefCopyEliminationPass() {
  return std::make_unique<NVWSArefCopyElimination>();
}
