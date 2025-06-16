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
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <memory>

#define GEN_PASS_CLASSES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-aref-insertion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::gpu;
using namespace triton::nvws;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

using GroupId = std::string;
using GroupMap = std::map<GroupId, WSGroup>;
using GroupSet = std::set<GroupId>;

// ----------------------------------------------------------------------------

/* introduces aref_create/put/get to preserve def-use across groups

   value semantics:
   ----------------
   if there is producer in @group1 and consumer in @group2, we insert arefs,
    e.g.
         %val = ..    @group1
          ..  = %val  @group2
   becomes
         ..
         %aref_val_buf = local_alloc
         %aref_val = aref_create %aref_val_buf
         ..
         %val = ..   @group1

         %dst_buf = aref_put.enter %aref_val    @group1
         local_store %val, %dst_buf             @group1
         aref_put.exit %aref_val                @group1

         %src_buf = aref_get.enter %aref_val   @group2
         %val1 =  local_load %src_buf          @group2
         aref_get.exit %aref_val               @group2

         update all uses of %val with %val1 in what follows
          ..  = %val1@group2A

   immutable smem semantics, immutable not support (no use-cases)
   --------------------------------------------------------------
     %a = local_alloc %val : memdesc<.. immutable > @group1
      .. = op %a                                     @group2

   becomes:
      %a = local_alloc %val              @group1
      %dst_A = aref_put.enter %aref_A    @group1
      aref_copy %dst_a, %a               @group1
      aref_put.exit %aref_a              @group1

      %src_a = aref_get.enter %aref_A    @group2
      %a1 = aref_copy %src_a             @group2 // %a1 & %a type is same
                                                 // immutable in this case
                                                 // doens't create
                                                 // type-mismatch downstream

      aref_get.exit %aref_A              @group2

      // update all uses of %a with %a1
       .. = op %a1                          @group2

   immutable tmem semantics, similar to local_alloc
   -------------------------------------------------

       %buf = tmem_alloc %val : memdesc<.. immutable > @group1
       = .. op %buf                                    @group2

   becomes:
      %a = tmem_alloc %val               @group1
      %dst_A = aref_put.enter %aref_A    @group1
      aref_copy %dst_a, %a               @group1
      aref_put.exit %aref_a              @group1

      %src_a = aref_get.enter %aref_A    @group2
      %a1 = aref_clone %src_a             @group2
      aref_get.exit %aref_A              @group2

      // update all uses of %a with %a1
       .. = op %a1                          @group2



   mutable tmem semantics
   -----------------------

      %buf, %tok = tmem_alloc      @group1, @group2 (must be in both groups)
      ..
      %tok1 = .. store_op %buf[%tok]      @group1
      %tok2 = .. load_op %buf[%tok1]     @group2

    becomes

       %aref_buf = tmem_alloc; %aref = aref_create %tmem_buf
       %buf, %tok = tmem_alloc     @group1, @group2

       %tok1 = .. store_op %buf[%tok]          @group1
       %dst, %tokDst = aref_put.enter %aref    @group1
       aref_copy %buf[%tok1], %dst[tokDst]     @group1
       aref_put.exit %aref                     @group1

       %src, %tokSrc = aref_get.enter %aref         @group2
       %tok1a = aref_copy %src[%tokSrc], %buf       @group2
       aref_get.exit %aref                          @group2

       // update all uses of %tok1 with %tok1a
       %tok2 = tmem_load %buf[%tok1a]      @group2



*/

struct ProducedValueInfo {
  GroupSet group; // group where result is being used
  Value result;   // result being produced
};

SmallVector<ProducedValueInfo> getProducedValues(Operation *op) {
  // specially handle ForOp/IfOp, as they have group.<idx> annotation for
  // results
  SmallVector<ProducedValueInfo> producedValues;
  if (isa<scf::ForOp, scf::IfOp>(op)) {
    for (auto result : op->getResults()) {
      auto groups = getGroups(result);
      producedValues.push_back({groups, result});
    }
  } else {
    // Handle remaining ops uniformly, all results produced by the same groups
    for (auto result : op->getResults()) {
      producedValues.push_back({getGroups(op), result});
    }
  }
  return producedValues;
};

ttng::ArefCreateOp createAref(OpBuilder &builder,
                              ProducedValueInfo &producedValue) {
  MemDescType arefBufType;

  auto result = producedValue.result;
  assert(!isa<AsyncTokenType>(result.getType()));

  if (auto memDescType = dyn_cast<MemDescType>(result.getType())) {
    arefBufType = getArefbufMemDescType(memDescType, 1);
  } else if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
    // if result is a value, create memdesctype for location where value will
    // be stored
    MemDescType memDescType;
    Attribute SharedMemorySpace =
        SharedMemorySpaceAttr::get(tensorType.getContext());
    if (auto load = result.getDefiningOp<DescriptorOpInterface>()) {
      auto encoding =
          ttng::getEncodingFromDescriptor(load, tensorType, load.getDesc());
      memDescType =
          MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                           encoding, SharedMemorySpace);
    } else {
      for (auto user : producedValue.result.getUsers()) {
        // if user is localAlloc/localStore, uses their memDescType
        if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user)) {
          memDescType = cast<MemDescType>(localAlloc.getResult().getType());
          break;
        } else if (auto localStore = dyn_cast<ttg::LocalStoreOp>(user)) {
          memDescType = cast<MemDescType>(localStore.getDst().getType());
          break;
        }
      }
    }
    if (!memDescType) {
      // The right smem encoding cannot be inferred from the IR. This can
      // happen, for example, in an attention kernel where smem is used to
      // communicate between different warp groups. We need to pick a new
      // encoding for such cases. For now, use a non-swizzled layout.
      // TODO: Use a swizzled one when possible
      auto CTALayout = getCTALayout(tensorType.getEncoding());
      auto newOrder = getOrderForMemory(tensorType);
      auto encoding = SwizzledSharedEncodingAttr::get(
          tensorType.getContext(), 1, 1, 1, newOrder, CTALayout);
      memDescType =
          MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                           encoding, SharedMemorySpace);
    }
    arefBufType = getArefbufMemDescType(memDescType, 1);
  } else {
    // need to support scalar types, similarly  to ranked tensor types
    llvm_unreachable("unsupported type");
  }
  auto loc = producedValue.result.getLoc();

  auto arefTy = triton::nvidia_gpu::ArefType::get(
      builder.getContext(),
      ttg::TypeArrayAttr::get(builder.getContext(), arefBufType));
  assert((isa<SharedMemorySpaceAttr>(arefBufType.getMemorySpace())));
  auto alloc = createAlloc(builder, loc, arefBufType, Value());
  alloc->setAttr("aref_buffer", builder.getUnitAttr());
  auto aref = builder.create<triton::nvidia_gpu::ArefCreateOp>(
      loc, arefTy, alloc->getResult(0));
  return aref;
};

bool isDescLoadAndAlloc(Value result) {
  auto alloc = result.getDefiningOp<LocalAllocOp>();
  if (!alloc)
    return false;
  return alloc.getSrc().getDefiningOp<DescriptorOpInterface>() != nullptr;
}

bool isGlobalLoadAndAlloc(Value result) {
  auto alloc = result.getDefiningOp<LocalAllocOp>();
  if (!alloc)
    return false;
  return alloc.getSrc().getDefiningOp<tt::LoadOp>() != nullptr;
}

void createNVWSDescriptorLoadOp(OpBuilder &builder, Operation *ttDescLoadOp,
                                Value dataBuf, GroupId producerGroup,
                                Location loc) {
  auto txCount = ttng::getTxCount(ttDescLoadOp);
  if (auto descLoad = dyn_cast<tt::DescriptorLoadOp>(ttDescLoadOp)) {
    auto newDescLoad = builder.create<triton::nvws::DescriptorLoadOp>(
        loc, descLoad.getDesc(), descLoad.getIndices(), txCount, dataBuf,
        descLoad.getCache(), descLoad.getEvict());
    setGroups(newDescLoad, {producerGroup});
  } else if (auto descGather = dyn_cast<tt::DescriptorGatherOp>(ttDescLoadOp)) {
    auto newDescGather = builder.create<triton::nvws::DescriptorGatherOp>(
        loc, descGather.getDesc(), descGather.getXOffsets(),
        descGather.getYOffset(), txCount, dataBuf);
    setGroups(newDescGather, {producerGroup});
  } else {
    llvm_unreachable("unknown descriptor op.");
  }
}

SmallVector<Operation *>
createArefPut(OpBuilder &builder, ttng::ArefCreateOp aref, std::string arefTag,
              ProducedValueInfo producedValue, GroupId producerGroup) {
  auto loc = producedValue.result.getLoc();
  auto arefBufType = cast<MemDescType>(aref.getOperand(0).getType());
  Value result = producedValue.result;
  auto dataBufType = getDataMemDescType(arefBufType, true);

  SmallVector<Type> buffers{dataBufType};
  SmallVector<Type> tokens{builder.getType<NoneType>()};

  auto putEnterOp = builder.create<ttng::ArefPutEnterOp>(
      loc, buffers, tokens, aref,
      mkConstant(builder, loc, 0, 32, {producerGroup}));
  setGroups(putEnterOp, {producerGroup});
  putEnterOp->setAttr("aref_tag", builder.getStringAttr(arefTag));
  auto dataBuf = putEnterOp.getBuffers()[0];

  auto producerKind = ttng::ArefProducer::NONE;
  SmallVector<Operation *> staleOps;

  if (isDescLoadAndAlloc(result)) {
    auto alloc = result.getDefiningOp<LocalAllocOp>();
    auto descOp = alloc.getSrc().getDefiningOp();
    createNVWSDescriptorLoadOp(builder, descOp, dataBuf, producerGroup, loc);
    producerKind = ttng::ArefProducer::TMALDG;
    staleOps.push_back(alloc);
    staleOps.push_back(descOp);
  } else if (isGlobalLoadAndAlloc(result)) {
    auto alloc = result.getDefiningOp<LocalAllocOp>();
    auto loadOp = alloc.getSrc().getDefiningOp<tt::LoadOp>();
    assert(loadOp);
    auto newLoad = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
        loc, loadOp.getPtr(), dataBuf, loadOp.getMask(), loadOp.getOther(),
        loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    setGroups(newLoad, {producerGroup});
    producerKind = ttng::ArefProducer::LDGSTS;
    staleOps.push_back(alloc);
    staleOps.push_back(loadOp);
  } else if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
    auto op = result.getDefiningOp();
    if (op && isa<DescriptorOpInterface>(op)) {
      createNVWSDescriptorLoadOp(builder, op, dataBuf, producerGroup, loc);
      producerKind = ttng::ArefProducer::TMALDG;
    } else if (op && isa<tt::LoadOp>(op)) {
      auto loadOp = cast<tt::LoadOp>(op);
      auto newLoad = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
          loc, loadOp.getPtr(), dataBuf, loadOp.getMask(), loadOp.getOther(),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
      setGroups(newLoad, {producerGroup});
      producerKind = ttng::ArefProducer::LDGSTS;
    } else {
      auto storeOp = builder.create<ttg::LocalStoreOp>(loc, result, dataBuf);
      setGroups(storeOp, {producerGroup});
      producerKind = ttng::ArefProducer::STS;
    }
  } else {
    // Note: need to support scalar types
    llvm_unreachable("unsupported type");
  }
  auto putExitOp = builder.create<ttng::ArefPutExitOp>(
      loc, aref, mkConstant(builder, loc, 0, 32, {producerGroup}),
      builder.getArrayAttr(SmallVector<Attribute>{
          ttng::ArefProducerAttr::get(aref.getContext(), producerKind)}));
  putExitOp->setAttr("aref_tag", builder.getStringAttr(arefTag));
  setGroups(putExitOp, {producerGroup});

  return staleOps;
};

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

SetVector<Operation *> getTransitiveConsumers(Operation *op) {
  SetVector<Operation *> opConsumers;
  for (auto user : op->getUsers()) {
    if (llvm::count_if(user->getResults(), [](auto res) {
          return isa<MemDescType>(res.getType());
        }) == 0) {
      opConsumers.insert(user);
    } else {
      auto consumers = getTransitiveConsumers(user);
      opConsumers.insert(consumers.begin(), consumers.end());
    }
  }
  return opConsumers;
}

ttng::ArefConsumer getConsumerKind(const SetVector<Operation *> &consumers) {
  assert(!consumers.empty());
  auto consumer = consumers.front();
  if (isa<ttg::LocalLoadOp>(consumer)) {
    return ttng::ArefConsumer::LDS;
  } else if (isa<ttng::WarpGroupDotOp>(consumer)) {
    return ttng::ArefConsumer::WGMMA;
  } else if (auto mmav5 = dyn_cast<ttng::MMAv5OpInterface>(consumer)) {
    return ttng::ArefConsumer::UMMA;
  }
  return ttng::ArefConsumer::NONE;
}

Operation *getExitInsertPoint(Block *producerBlock,
                              const SetVector<Operation *> &consumers) {
  DenseMap<Operation *, int> opOrdering;
  producerBlock->walk(
      [&](Operation *op) { opOrdering[op] = opOrdering.size(); });

  SetVector<Operation *> validConsumers;
  for (auto consumer : consumers) {
    if (getBlockScope(producerBlock, consumer->getBlock()) !=
        BlockScope::UNSUPPORTED)
      validConsumers.insert(consumer);
  }
  assert(!validConsumers.empty());

  auto lastConsumer = *llvm::max_element(validConsumers, [&](auto a, auto b) {
    return opOrdering.at(a) < opOrdering.at(b);
  });

  auto consumerScope = getBlockScope(producerBlock, lastConsumer->getBlock());
  if (BlockScope::SAME_BLOCK == consumerScope) {
    return lastConsumer;
  } else if (BlockScope::NESTED_INSIDE == consumerScope) {
    auto regionOp = lastConsumer->getParentOp();
    while (regionOp->getBlock() != producerBlock) {
      regionOp = regionOp->getParentOp();
    }
    return regionOp;
  } else {
    llvm_unreachable("unsupported consumer scope");
  }
  return nullptr;
}

void createArefGet(OpBuilder &builder, ttng::ArefCreateOp aref,
                   std::string arefTag, ProducedValueInfo producedValue,
                   SetVector<Operation *> users, GroupSet consumerGroups,
                   bool needArefPhi) {
  OpBuilder::InsertionGuard g(builder);
  auto loc = producedValue.result.getLoc();
  auto arefBufType = cast<MemDescType>(aref.getOperand(0).getType());

  Value result = producedValue.result;

  SmallVector<Type> buffers{getDataMemDescType(arefBufType, false)};
  SmallVector<Type> tokens{builder.getType<NoneType>()};
  auto getEnterOp = builder.create<ttng::ArefGetEnterOp>(
      loc, buffers, tokens, aref,
      mkConstant(builder, loc, 0, 32, consumerGroups));
  Value dataBuf = getEnterOp.getBuffers()[0];
  setGroups(getEnterOp, consumerGroups);
  getEnterOp->setAttr("aref_tag", builder.getStringAttr(arefTag));

  auto createExit = [&](ttng::ArefConsumer consumerKind) {
    SmallVector<Attribute> consumerAttr{
        ttng::ArefConsumerAttr::get(aref.getContext(), consumerKind)};
    auto consumersAttr = builder.getArrayAttr(consumerAttr);
    auto getExitOp = builder.create<ttng::ArefGetExitOp>(
        loc, aref, mkConstant(builder, loc, 0, 32, consumerGroups),
        consumersAttr);
    getExitOp->setAttr("aref_tag", builder.getStringAttr(arefTag));
    setGroups(getExitOp, consumerGroups);
  };

  Value newOperand;
  if (auto memDescType = dyn_cast<MemDescType>(result.getType())) {
    newOperand = dataBuf;
    auto consumers = getTransitiveConsumers(result.getDefiningOp());
    auto insertPoint =
        getExitInsertPoint(result.getDefiningOp()->getBlock(), consumers);
    builder.setInsertionPointAfter(insertPoint);
    createExit(getConsumerKind(consumers));
  } else if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
    auto localLoadOp =
        builder.create<ttg::LocalLoadOp>(loc, tensorType, dataBuf);
    newOperand = localLoadOp.getResult();
    setGroups(localLoadOp, consumerGroups);
    createExit(ttng::ArefConsumer::LDS);
  } else {
    // Note: need to support scalar types
    llvm_unreachable("unsupported type");
  }

  /* see if we need to use aref_phi op
     we use are_phi if we have this situation

      %val = .. @gr1
       ..  = %val @gr1, @gr2, @gr3

      we'll have
      %val = ..                     @gr1
      %dst = aref_put.enter %aref   @gr1
      local_store %val, %dst        @gr1
      aref_put.exit %aref           @gr1

      %src = aref_put.enter %aref   @gr2,@gr3
      %val2 = local_load %src       @gr2,@gr3
      aref_get.exit %aref           @gr2,@gr3

      %val3 = aref_phi %val, %val2

      ..  = %val3 @gr1, @gr2, @gr3
  */

  if (needArefPhi) {
    newOperand = builder.create<ttng::ArefPhiOp>(loc, result.getType(), result,
                                                 newOperand);
  }

  // update result operand with newOperand
  for (auto user : users) {
    bool updatedOperand = false;
    for (auto [i, operand] : llvm::enumerate(user->getOperands())) {
      if (result == operand) {
        user->setOperand(i, newOperand);
        updatedOperand = true;
      }
    }
    assert(updatedOperand);
  }
};

bool insertArefs(OpBuilder &builder, tt::FuncOp funcOp,
                 ProducedValueInfo producedValue, int arefTag) {
  GroupSet consumerGroups;
  auto [producerGroups, result] = producedValue;
  assert(!producerGroups.empty());

  bool needArefPhi = false;
  for (auto &useOpnd : result.getUses()) {
    GroupSet userGroups;
    if (auto forOp = dyn_cast<scf::ForOp>(useOpnd.getOwner())) {
      auto idx = useOpnd.getOperandNumber();
      if (idx < 3) {
        auto groups = getGroups(forOp);
        userGroups.insert(groups.begin(), groups.end());
      } else {
        auto groups = getGroupsIdx(forOp, idx - 3);
        userGroups.insert(groups.begin(), groups.end());
      }
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(useOpnd.getOwner())) {
      auto parentOp = yieldOp->getParentOp();
      auto groups = getGroupsIdx(parentOp, useOpnd.getOperandNumber());
      userGroups.insert(groups.begin(), groups.end());
    } else {
      userGroups = getGroups(useOpnd.getOwner());
    }

    for (auto group : userGroups) {
      if (producerGroups.count(group) == 0) {
        consumerGroups.insert(group);
      } else {
        needArefPhi = true;
      }
    }
  }
  if (consumerGroups.empty())
    return false;

  // for now we enforce that there is only one consumer group, but in
  // future we may want to support multiple consumer groups
  assert(consumerGroups.size() == 1);

  // we also enforce that there is at least one user of the result
  assert(llvm::count_if(result.getUsers(), [](auto) { return true; }) >= 1);

  // if there are multitiple consumer groups, we need to generate as
  // separate aref_get per consumer group

  // if there are mutlilpe producer groups, we just pick first group to
  // generate aref_put,

  // if there are multiple producers, it is possible zip them in
  // round-robin way, e.g.
  //      (p1,p2,p3)x(c1,c2,c3,c4,c5,c6) -> (p1,c1) (p2,c2), (p3,c3),
  //      (p1,c4), (p2,c5) (p3,c6) but it will require multilpe aref
  //      buffers

  SetVector<Operation *> users(result.getUsers().begin(),
                               result.getUsers().end());

  ttng::ArefCreateOp aref;
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    aref = createAref(builder, producedValue);
  }

  // for now we set put/get right after producing op, e.g.

  //     %val = ..  @gr1
  //       ..
  //      .. = val  @gr2

  //      %val = ..   @gr1
  //      put %val    @gr2
  //      %val = get  @gr2
  //        ..
  //      .. = val    @gr2
  //
  // However, in future we may want to consider where consumer is, e.g.

  //   for  {
  //     %val = ..  @gr1
  //      if {
  //           .. = %val @gr2
  //         }
  //    }

  //  we may want to have put/get next to consumer, e.g.

  //   for  {
  //     %val = ..      @gr1
  //      if {
  //            put %val   @gr1
  //            %val = get %gr2
  //            .. = %val @gr2
  //         }
  //   }

  // that can be important if we'd want to support epilogue decoupling
  // in flattened loops.

  auto tag = std::string("aref_") + std::to_string(arefTag);
  auto staleOps =
      createArefPut(builder, aref, tag, producedValue, *producerGroups.begin());
  createArefGet(builder, aref, tag, producedValue, users, consumerGroups,
                needArefPhi);

  for (auto op : staleOps) {
    op->erase();
  }

  return true;
}

void runArefInsertionOnFunc(triton::FuncOp funcOp) {
  OpBuilder builder(funcOp);

  // Gather all operations from the function body before inserting any arefs.
  SmallVector<Operation *> opsToArefy;
  funcOp.walk([&](Operation *op) {
    if (isa<ttng::ArefCreateOp, ttng::TMEMAllocOp, ttng::ArefPutEnterOp,
            ttng::ArefGetEnterOp, ttng::TMEMLoadOp, ttng::TMEMStoreOp,
            ttng::ArefPutExitOp, ttng::ArefGetExitOp, scf::YieldOp,
            triton::FuncOp, triton::ReturnOp>(op))
      return;

    opsToArefy.push_back(op);
  });

  int arefTag = 0;

  // Visit all for-ops to check if we need to insert arefs between block
  // arguments and their users. This handles cases where a value is produced in
  // one group but consumed in another across loop iterations, e.g.:
  //
  //     %val0 = .. @gr1
  //     %ret = for .. iter_args(%val = %val0) {
  //       ..  =  %val   @gr2
  //       %val1 = ..     @gr1
  //       yield %val1
  //     } groups=[@gr1,@gr2], groups.0=[@gr1]
  //
  //  will insert aref between block argument and its use
  //
  //     %val0 = .. @gr1
  //     %ret = for .. iter_args(%val = %val0) {
  //        aref_put %val    @gr1
  //        %val' = aref_get %gr2
  //          ..  =  %val'   @gr2
  //        %val1 = ..       @gr1
  //        yield %val1
  //     } groups=[@gr1,@gr2], groups.0=[@gr1]
  //
  funcOp.walk([&](scf::ForOp forOp) {
    auto body = forOp.getBody();
    OpBuilder builder(forOp);
    for (auto arg : body->getArguments()) {
      if (arg.getArgNumber() > 0 && !isa<AsyncTokenType>(arg.getType())) {
        // insert arefs only for loop-carried values
        auto groups = getGroupsIdx(forOp, arg.getArgNumber() - 1);
        auto producedValue = ProducedValueInfo{groups, arg};
        builder.setInsertionPointToStart(body);
        if (insertArefs(builder, funcOp, producedValue, arefTag))
          arefTag++;
      }
    }
  });

  // Handle ops in the function body
  for (auto op : opsToArefy) {
    // otherwise we need to place put/get
    auto producedValues = getProducedValues(op);
    for (auto producedValue : producedValues) {
      OpBuilder builder(op);
      builder.setInsertionPointAfter(op);
      if (insertArefs(builder, funcOp, producedValue, arefTag))
        arefTag++;
    }
  }
}

class NVWSArefInsertion : public NVWSArefInsertionBase<NVWSArefInsertion> {
public:
  void runOnOperation() override {
    auto mod = getOperation();
    mod.walk([&](triton::FuncOp funcOp) { runArefInsertionOnFunc(funcOp); });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createNVWSArefInsertionPass() {
  return std::make_unique<NVWSArefInsertion>();
}
