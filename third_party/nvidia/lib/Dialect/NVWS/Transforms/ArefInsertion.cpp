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
#include "mlir/Dialect/UB/IR/UBOps.h"
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
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/WSUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <memory>

// #define GEN_PASS_CLASSES
// #include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define GEN_PASS_CLASSES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-aref-insertion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::gpu;
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
  // tmemAlloc handled specially, if there is src, tmem is immutable
  // and we track result, otherwise tmem is mutable and we track token
  if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(op)) {
    auto groups = getGroups(tmemAlloc);
    if (tmemAlloc.getToken())
      return {{groups, tmemAlloc.getToken()}};
    else
      return {{groups, tmemAlloc.getResult()}};
  }

  // specially handle ForOp/IfOp, as they have group.<idx> annotation for
  // results
  if (isa<scf::ForOp, scf::IfOp>(op)) {
    SmallVector<ProducedValueInfo> producedValues;
    for (auto result : op->getResults()) {
      auto groups = getGroups(result);
      producedValues.push_back({groups, result});
    }
    return producedValues;
  }

  // Handle all other operations uniformly by collecting their results and
  // groups
  //  Note: tmem_load produce both a token and value would cause issues
  //  with epilogue decoupling in flattened loops. This is not currently
  //  supported but could be added in the future.
  SmallVector<ProducedValueInfo> producedValues;
  for (auto result : op->getResults()) {
    auto groups = getGroups(op);
    producedValues.push_back({groups, result});
  }
  return producedValues;
};

ttng::ArefCreateOp createAref(OpBuilder &builder,
                              ProducedValueInfo &producedValue) {
  MemDescType arefBufType;

  auto result = producedValue.result;

  if (isa<AsyncTokenType>(result.getType())) {
    // if result is async-token, op does something on mutable tmem
    // locate tmem buffer associated witht this token
    auto buffer = getTokenProducerOp(result).buffer;
    auto memDescType = cast<MemDescType>(buffer.getType());
    arefBufType = getArefbufMemDescType(memDescType, 1);
  } else if (auto memDescType = dyn_cast<MemDescType>(result.getType())) {
    arefBufType = getArefbufMemDescType(memDescType, 1);
  } else if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
    // if result is a value, create memdesctype for location where value will
    // be stored
    MemDescType memDescType;
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
    if (!memDescType) {
      Attribute SharedMemorySpace =
          SharedMemorySpaceAttr::get(tensorType.getContext());
      auto CTALayout = getCTALayout(tensorType.getEncoding());
      // No swizzling for scale for now
      auto newOrder = getOrderForMemory(tensorType);
      auto newLayout = SwizzledSharedEncodingAttr::get(
          tensorType.getContext(), 1, 1, 1, newOrder, CTALayout);
      memDescType =
          MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                           newLayout, SharedMemorySpace);
    }
    arefBufType = getArefbufMemDescType(memDescType, 1);
  } else {
    // need to support scalar types, similarly  to ranked tensor types
    llvm_unreachable("unsupported type");
  }
  auto loc = producedValue.result.getLoc();

  auto arefTy = triton::nvidia_gpu::ArefType::get(
      builder.getContext(),
      TypeArrayAttr::get(builder.getContext(), arefBufType));
  assert((isa<SharedMemorySpaceAttr, ttng::TensorMemorySpaceAttr>(
      arefBufType.getMemorySpace())));
  auto alloc = ttg::createAlloc(builder, loc, arefBufType, Value());
  alloc->setAttr("aref_buffer", builder.getUnitAttr());
  auto aref = builder.create<triton::nvidia_gpu::ArefCreateOp>(
      loc, arefTy, alloc->getResult(0));
  return aref;
};

void createArefPut(OpBuilder &builder, ttng::ArefCreateOp aref,
                   std::string arefTag, ProducedValueInfo producedValue,
                   GroupId producerGroup) {
  auto loc = producedValue.result.getLoc();
  auto arefBufType = cast<MemDescType>(aref.getOperand(0).getType());
  Value result = producedValue.result;
  auto dataBufType = getDataMemDescType(arefBufType, true);

  SmallVector<Type> buffers{dataBufType};
  SmallVector<Type> tokens;
  if (isa<AsyncTokenType>(result.getType()))
    tokens.push_back(builder.getType<AsyncTokenType>());
  else
    tokens.push_back(builder.getType<NoneType>());

  auto putEnterOp = builder.create<ttng::ArefPutEnterOp>(
      loc, buffers, tokens, aref,
      mkConstant(builder, loc, 0, 32, {producerGroup}));
  setGroups(putEnterOp, {producerGroup});
  putEnterOp->setAttr("aref_tag", builder.getStringAttr(arefTag));
  auto dataBuf = putEnterOp.getBuffers()[0];

  auto producerKind = ttng::ArefProducer::NONE;
  if (isa<AsyncTokenType>(result.getType())) {
    // when result is token, find associated buffer were result will be copied
    auto buffer = getTokenProducerOp(result).buffer;
    auto putToken = putEnterOp.getTokens()[0];
    assert(isa<AsyncTokenType>(putToken.getType()));

    auto copyOp = builder.create<ttng::ArefCopyOp>(loc, Type{}, buffer, dataBuf,
                                                   result, putToken);
    setGroups(copyOp, {producerGroup});
  } else if (auto memDescType = dyn_cast<MemDescType>(result.getType())) {
    auto copyOp = builder.create<ttng::ArefCopyOp>(loc, Type{}, result, dataBuf,
                                                   Value(), Value());
    setGroups(copyOp, {producerGroup});
  } else if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
    auto storeOp = builder.create<ttg::LocalStoreOp>(loc, result, dataBuf);
    setGroups(storeOp, {producerGroup});
    if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(
            result.getDefiningOp()))
      producerKind = ttng::ArefProducer::TMALDG;
    else if (isa<tt::LoadOp>(result.getDefiningOp()))
      producerKind = ttng::ArefProducer::LDGSTS;
    else
      producerKind = ttng::ArefProducer::STS;
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
};

void createArefGet(OpBuilder &builder, ttng::ArefCreateOp aref,
                   std::string arefTag, ProducedValueInfo producedValue,
                   SmallVector<Operation *> users, GroupSet consumerGroups, bool needArefPhi) {
  OpBuilder::InsertionGuard g(builder);
  auto loc = producedValue.result.getLoc();
  auto arefBufType = cast<MemDescType>(aref.getOperand(0).getType());

  Value result = producedValue.result;

  SmallVector<Type> buffers, tokens;
  if (isa<AsyncTokenType>(result.getType())) {
    // if there is token, we assume it is mutable memory
    buffers.push_back(getDataMemDescType(arefBufType, true));
    tokens.push_back(builder.getType<AsyncTokenType>());
  } else {
    // otherwise it is immutable, which somes from local/tmem_allo with src
    buffers.push_back(getDataMemDescType(arefBufType, false));
    tokens.push_back(builder.getType<NoneType>());
  }
  auto getEnterOp = builder.create<ttng::ArefGetEnterOp>(
      loc, buffers, tokens, aref,
      mkConstant(builder, loc, 0, 32, consumerGroups));
  Value dataBuf = getEnterOp.getBuffers()[0];
  setGroups(getEnterOp, consumerGroups);
  getEnterOp->setAttr("aref_tag", builder.getStringAttr(arefTag));

  Value newOperand;
  auto consumerKind = ttng::ArefConsumer::NONE;
  if (isa<AsyncTokenType>(result.getType())) {
    auto buffer = getTokenProducerOp(result).buffer;

    // for now we assume mutable buffer comes from TMEMAllocOp
    auto alloc = buffer.getDefiningOp<ttng::TMEMAllocOp>();
    assert(alloc);
    assert(!alloc.getSrc());
    auto srcToken = getEnterOp.getTokens()[0];
    assert(isa<AsyncTokenType>(srcToken.getType()));

    auto copyOp =
        builder.create<ttng::ArefCopyOp>(loc, builder.getType<AsyncTokenType>(),
                                         dataBuf, buffer, srcToken, Value());
    newOperand = copyOp.getToken();
    setGroups(copyOp, consumerGroups);
  } else if (auto memDescType = dyn_cast<MemDescType>(result.getType())) {
    auto copyOp =
        builder.create<ttng::ArefCloneOp>(loc, result.getType(), dataBuf);
    newOperand = copyOp.getResult();
    setGroups(copyOp, consumerGroups);
  } else if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
    auto localLoadOp =
        builder.create<ttg::LocalLoadOp>(loc, tensorType, dataBuf);
    newOperand = localLoadOp.getResult();
    setGroups(localLoadOp, consumerGroups);
    consumerKind = ttng::ArefConsumer::LDS;
  } else {
    // Note: need to support scalar types
    llvm_unreachable("unsupported type");
  }

  SmallVector<Attribute> consumerAttr{
      ttng::ArefConsumerAttr::get(aref.getContext(), consumerKind)};
  auto consumersAttr = builder.getArrayAttr(consumerAttr);
  auto getExitOp = builder.create<ttng::ArefGetExitOp>(
      loc, aref, mkConstant(builder, loc, 0, 32, consumerGroups),
      consumersAttr);
  getExitOp->setAttr("aref_tag", builder.getStringAttr(arefTag));
  setGroups(getExitOp, consumerGroups);

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
        assert(!updatedOperand);
        user->setOperand(i, newOperand);
        updatedOperand = true;
      }
    }
    assert(updatedOperand);
  }
};

void runArefInsertionOnFunc(triton::FuncOp funcOp) {
  OpBuilder builder(funcOp);

  SmallVector<Operation *> opsToArefy;
  funcOp.walk([&](Operation *op) {
    if (isa<scf::YieldOp, triton::FuncOp, triton::ReturnOp>(op))
      return;

    opsToArefy.push_back(op);
  });

  int arefTag = 0;

  for (auto op : opsToArefy) {

    // otherwise we need to place put/get
    auto producedValues = getProducedValues(op);

    for (auto producedValue : producedValues) {
      GroupSet consumerGroups;
      bool needArefPhi = false;
      auto [producerGroups, result] = producedValue;

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
        continue;

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

      SmallVector<Operation *> users(result.getUsers().begin(),
                                     result.getUsers().end());

      builder.setInsertionPointToStart(&funcOp.getBody().front());
      auto aref = createAref(builder, producedValue);

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

      // in nested loop, the put/get will be placed after for-loop, because thi
      // is where final tmem produce token will be, e.g.

      // %tok0 = ..
      // %tok1 = for  %tok0 = %tok  {
      //           %tok1 = mma .. %tok0 ..  @gr1
      //           yield %tok1
      //          }
      //  ..
      //  tmem_load .. %tok1  @gr2

      // so aref will be placed after %tok1 producer, which is for-loop

      // %tok0 = ..
      // %tok1 = for  %tok0 = %tok  {
      //           %tok1 = mma .. %tok0 .. @gr1
      //           yield %tok1
      //          }
      //  put %buf  @gr1
      //  get %buf  @gr2
      //  ..
      //  tmem_load .. %tok1 @gr2
      //
      // That also help with perf in MOE kernel, where we want to have
      // aref_get before bias-load, so that convert_layout + tmem_load could be
      // intearleaved

      builder.setInsertionPointAfter(op);
      auto tag = std::string("aref_") + std::to_string(arefTag);
      createArefPut(builder, aref, tag, producedValue, *producerGroups.begin());
      createArefGet(builder, aref, tag, producedValue, users, consumerGroups, needArefPhi);
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
