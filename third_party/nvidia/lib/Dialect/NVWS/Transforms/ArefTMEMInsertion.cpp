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
#include "mlir/IR/OpDefinition.h"
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

#define DEBUG_TYPE "nvws-aref-tmem-insertion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::gpu;
using namespace triton::nvws;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

// Find the position of a value in any range (e.g., loop init args, yield
// operands)
template <typename Range>
inline int findValuePos(const Range &range, mlir::Value v) {
  int pos = -1;
  for (auto [i, arg] : llvm::enumerate(range)) {
    if (arg == v) {
      pos = i;
      break;
    }
  }
  assert(pos != -1 && "Value not found in range");
  return pos;
}

// Get the only element from a set (with assert)
template <typename T> inline T getOne(const std::set<T> &s) {
  assert(s.size() == 1);
  return *s.begin();
}

// Helper function to create Aref operations
ttng::ArefCreateOp createAref(Value arefBuffer, bool first_get,
                              DenseSet<ttng::ArefCreateOp> &scopedArefs) {
  OpBuilder builder(arefBuffer.getContext());
  builder.setInsertionPointAfter(arefBuffer.getDefiningOp());
  auto arefBufType = cast<MemDescType>(arefBuffer.getType());
  auto arefTy = triton::nvidia_gpu::ArefType::get(
      builder.getContext(),
      ttg::TypeArrayAttr::get(builder.getContext(), arefBufType));
  assert((isa<ttng::TensorMemorySpaceAttr>(arefBufType.getMemorySpace())));
  auto aref = builder.create<triton::nvidia_gpu::ArefCreateOp>(
      arefBuffer.getLoc(), arefTy, arefBuffer);
  if (first_get) {
    aref->setAttr("first_get", builder.getUnitAttr());
  }
  scopedArefs.insert(aref);
  return aref;
}

// Helper function to create Aref Put Enter operations
ttng::ArefPutEnterOp createArefPutEnterOp(OpBuilder &builder, Location loc,
                                          ttng::ArefCreateOp aref,
                                          std::string group) {
  auto arefBufType = cast<MemDescType>(aref.getOperand(0).getType());
  auto dataBufType = getDataMemDescType(arefBufType, true);
  SmallVector<Type> buffers{dataBufType};
  SmallVector<Type> tokens{builder.getType<AsyncTokenType>()};
  auto putEnterOp = builder.create<ttng::ArefPutEnterOp>(
      loc, buffers, tokens, aref, mkConstant(builder, loc, 0, 32, {group}));
  setGroups(putEnterOp, {group});
  return putEnterOp;
}

// Helper function to create Aref Put Exit operations
ttng::ArefPutExitOp
createArefPutExitOp(OpBuilder &builder, Location loc, ttng::ArefCreateOp aref,
                    SmallVector<ttng::ArefProducer> producers,
                    std::string group) {
  SmallVector<Attribute> producerAttrs;
  for (auto producer : producers) {
    producerAttrs.push_back(
        ttng::ArefProducerAttr::get(aref.getContext(), producer));
  }
  auto putExitOp = builder.create<ttng::ArefPutExitOp>(
      loc, aref, mkConstant(builder, loc, 0, 32, {group}),
      builder.getArrayAttr(producerAttrs));
  setGroups(putExitOp, {group});
  return putExitOp;
}

// Helper function to create Aref Get Enter operations
ttng::ArefGetEnterOp createArefGetEnterOp(OpBuilder &builder, Location loc,
                                          ttng::ArefCreateOp aref,
                                          std::string group) {
  auto arefBufType = cast<MemDescType>(aref.getOperand(0).getType());
  auto dataBufType = getDataMemDescType(arefBufType, true);
  SmallVector<Type> buffers{dataBufType};
  SmallVector<Type> tokens{builder.getType<AsyncTokenType>()};
  auto getEnterOp = builder.create<ttng::ArefGetEnterOp>(
      loc, buffers, tokens, aref, mkConstant(builder, loc, 0, 32, {group}));
  setGroups(getEnterOp, {group});
  return getEnterOp;
}

// Helper function to create Aref Get Exit operations
ttng::ArefGetExitOp
createArefGetExitOp(OpBuilder &builder, Location loc, ttng::ArefCreateOp aref,
                    SmallVector<ttng::ArefConsumer> consumers,
                    std::string group) {
  SmallVector<Attribute> consumerAttrs;
  for (auto consumer : consumers) {
    consumerAttrs.push_back(
        ttng::ArefConsumerAttr::get(aref.getContext(), consumer));
  }
  auto getExitOp = builder.create<ttng::ArefGetExitOp>(
      loc, aref, mkConstant(builder, loc, 0, 32, {group}),
      builder.getArrayAttr(consumerAttrs));
  setGroups(getExitOp, {group});
  return getExitOp;
}

// ArefTMEMInsertion: Handles insertion of Aref ops for a TMEM allocation
namespace arefTMEMInsertion {
// TokDag: Represents a DAG of token/buffer dependencies for a TMEM
// allocation.
struct TokDag {
  Operation *op = {};
  std::string group;
  SmallVector<std::unique_ptr<TokDag>> subDags = {};
  std::unique_ptr<TokDag> user = {};
  TokDag *parent = {};
  TokDag(Operation *op, std::string group, TokDag *parent)
      : op(op), group(group), parent(parent) {}
};

// TMEMAref: State for tracking buffer/token and associated ops during aref
// insertion.
struct TMEMAref {
  enum Kind { PUT, GET };
  Value tmemBuffer;
  Value tmemTok;
  Value buffer;
  Value token;
  ttng::ArefPutEnterOp putOp;
  ttng::ArefGetEnterOp getOp;
  ttng::ArefPutEnterOp scopePutOp;
  ttng::ArefGetEnterOp scopeGetOp;
  ttng::ArefProducer producer;
  ttng::ArefConsumer consumer;
  std::string curGroup;
};

// Helper function to insert Put Enter/Exit operations and update maps
void insertPutEnterExit(
    OpBuilder &builder, Location loc, ttng::ArefCreateOp arefOp,
    ttng::ArefProducer producer, std::string group,
    DenseMap<ttng::ArefCreateOp, std::string> &arefPutMap,
    DenseMap<ttng::ArefCreateOp, std::pair<TMEMAref::Kind, int>>
        &arefLastUseMap) {
  assert(arefPutMap.count(arefOp) == 0 || arefPutMap.at(arefOp) == group);
  createArefPutEnterOp(builder, loc, arefOp, group);
  createArefPutExitOp(builder, loc, arefOp, {producer}, group);
  arefPutMap[arefOp] = group;
  arefLastUseMap[arefOp] = {TMEMAref::PUT, (int)producer};
}

// Helper function to insert Get Enter/Exit operations and update maps
void insertGetEnterExit(
    OpBuilder &builder, Location loc, ttng::ArefCreateOp arefOp,
    ttng::ArefConsumer consumer, std::string group,
    DenseMap<ttng::ArefCreateOp, std::string> &arefGetMap,
    DenseMap<ttng::ArefCreateOp, std::pair<TMEMAref::Kind, int>>
        &arefLastUseMap) {
  assert(arefGetMap.count(arefOp) == 0 || arefGetMap.at(arefOp) == group);
  createArefGetEnterOp(builder, loc, arefOp, group);
  createArefGetExitOp(builder, loc, arefOp, {consumer}, group);
  arefGetMap[arefOp] = group;
  arefLastUseMap[arefOp] = {TMEMAref::GET, (int)consumer};
}

// This function ensures proper sequencing between put.enter and get.exit
// operations across different groups to maintain the original schedule and
// prevent data races, at the end of the put scope.
//
// scopeAref: An aref that needs to be matched at scope boundaries. In the
// listing below, %aref1 is the scopeAref, that needs matching get.enter/exit
// at scope boundaries.
//
// Specifically, we need to ensure that get.enter %aref1 waits for get.exit
// %aref2 to complete before yielding. Without this, the last get.enter %aref1
// would only wait for its own previous put.exit %aref1 @1, causing data
// races.
//
// The handshake mechanism below establishes the required dependencies through
// intermediate arefs to enforce the correct ordering.
//
// put.enter %aref1   @1
// put.exit  %aref1   @1
//
// get.enter %aref1   @2
// for  {
//    get.exit %aref1 @2
//
//    put.enter %aref1 @1
//    put.exit  %aref1 @1
//
//    ...
//
//    get.enter %aref2 @3    <- owner changed to @3, need a different aref
//    get.exit  %aref2 @3
//
//     ... scopeArefHandshake generates the code below
//
//    put.enter %aref3 @3
//    put.exit  %aref3 @3   <- signals that get.exit %aref2 in @3 is done
//    get.enter %aref3 @2   <- waits for put.exit %aref3 in @2
//    get.exit  %aref3 @2      guarantees that get.exit %aref2 in @3 is done
//
//      ...
//
//    get.enter %aref1 @2   <- no race, because through %aref3 we wait
//                             for group @3 to finish
//    yield
// }
// get.exit %aref2

// In some communication patterns, we may not have a matching put.exit %aref1
// operation. Without inserting a matching put operation, the get.enter %aref1
// @1 at the end of the for-loop will hang waiting for put.exit %aref1 that
// never arrives. This scenario is demonstrated in the @loop_token2 lit-test.

// put.enter %aref1   @1
// put.exit  %aref    @1
//
// get.enter %aref1   @2
// for  {
//    get.exit %aref1 @2
//
//    put.enter %aref2 @3  <- owner changed to group @3, need different aref
//    put.exit  %aref2 @3  <- no matching put.enter/exit %aref1
//
//     ...
//
//    get.enter %aref3 @3
//    get.exit  %aref3 @3
//
//     ... scopeArefHandshake generates the code below
//
//    put.enter %aref3 @3   <- signals that get.enter %aref2 is done
//    put.exit  %aref3 @3
//    get.enter %aref3 @1   <- waits for put.exit %aref3, but in group @1
//    get.exit  %aref3 @1      in @1, because we need to issue matching put
//
//      .. there is no matching put, inserts aref.put %aref1 @1
//
//    aref.put.enter %aref1 @1  <- insert matching aref.put %aref1
//    aref.put.exit  %aref1 @1
//
//     ..
//
//    get.enter %aref1 @2   <- no-race, no dead-lock
//    yield
// }
// get.exit %aref1 @2
void scopeArefHandshake(
    OpBuilder &builder, Location loc, ttng::ArefCreateOp scopeArefOp,
    bool isPut, int attributeProducerConsumer, std::string curGroup,
    std::string newGroup, DenseMap<ttng::ArefCreateOp, std::string> &arefPutMap,
    DenseMap<ttng::ArefCreateOp, std::string> &arefGetMap,
    DenseMap<ttng::ArefCreateOp, std::pair<TMEMAref::Kind, int>>
        &arefLastUseMap,
    DenseSet<ttng::ArefCreateOp> &scopedArefs) {

  auto newArefOp = createAref(scopeArefOp.getOperand(0), !isPut, scopedArefs);

  if (isPut) {
    // Insert put/get operations for the new aref in current group
    insertPutEnterExit(builder, loc, newArefOp,
                       (ttng::ArefProducer)attributeProducerConsumer, curGroup,
                       arefPutMap, arefLastUseMap);

    if (arefLastUseMap.at(scopeArefOp).first == TMEMAref::PUT) {
      // Matching put exists - handshake with previous put in new group
      insertGetEnterExit(builder, loc, newArefOp, ttng::ArefConsumer::LDTM,
                         newGroup, arefGetMap, arefLastUseMap);
    } else {
      // No matching put - handshake with previous put in its group
      auto putGroup = arefPutMap.at(scopeArefOp);
      insertGetEnterExit(builder, loc, newArefOp, ttng::ArefConsumer::LDTM,
                         putGroup, arefGetMap, arefLastUseMap);
      // Create matching put for scopeAref
      insertPutEnterExit(builder, loc, scopeArefOp, ttng::ArefProducer::NONE,
                         putGroup, arefPutMap, arefLastUseMap);
    }
  } else {
    insertGetEnterExit(builder, loc, newArefOp,
                       (ttng::ArefConsumer)attributeProducerConsumer, curGroup,
                       arefGetMap, arefLastUseMap);

    if (arefLastUseMap.at(scopeArefOp).first == TMEMAref::GET) {
      assert(0);
      insertPutEnterExit(builder, loc, newArefOp, ttng::ArefProducer::STTM,
                         newGroup, arefPutMap, arefLastUseMap);
    } else {
      auto getGroup = arefGetMap.at(scopeArefOp);
      insertPutEnterExit(builder, loc, newArefOp, ttng::ArefProducer::STTM,
                         getGroup, arefPutMap, arefLastUseMap);
      insertGetEnterExit(builder, loc, scopeArefOp, ttng::ArefConsumer::LDTM,
                         getGroup, arefGetMap, arefLastUseMap);
    }
  }
}

// Helper function to create exit operations for PUT state
void createPutExitOperations(
    OpBuilder &builder, Location loc, TMEMAref &state, std::string curGroup,
    DenseMap<ttng::ArefCreateOp, std::string> &arefPutMap,
    DenseMap<ttng::ArefCreateOp, std::string> &arefGetMap,
    DenseMap<ttng::ArefCreateOp, std::pair<TMEMAref::Kind, int>>
        &arefLastUseMap,
    DenseSet<ttng::ArefCreateOp> &scopedAref) {
  assert(!state.getOp);
  auto arefOp = state.putOp.getAref().getDefiningOp<ttng::ArefCreateOp>();
  createArefPutExitOp(builder, loc, arefOp, {state.producer}, curGroup);
  arefLastUseMap[arefOp] = {TMEMAref::PUT, (int)state.producer};
  if (!state.curGroup.empty()) {
    arefOp = createAref(arefOp.getOperand(0), false, scopedAref);
    insertPutEnterExit(builder, loc, arefOp, state.producer, curGroup,
                       arefPutMap, arefLastUseMap);
    insertGetEnterExit(builder, loc, arefOp, ttng::ArefConsumer::NONE,
                       state.curGroup, arefGetMap, arefLastUseMap);
    ttng::ArefCreateOp arefOp;
    for (auto [aref, group] : arefPutMap) {
      if (group == state.curGroup) {
        arefOp = aref;
        break;
      }
    }
    assert(arefOp);
    insertPutEnterExit(builder, loc, arefOp, ttng::ArefProducer::NONE,
                       state.curGroup, arefPutMap, arefLastUseMap);
    state.curGroup = {};
  }
}

// Helper function to create exit operations for GET state
void createGetExitOperations(
    OpBuilder &builder, Location loc, TMEMAref &state, std::string curGroup,
    DenseMap<ttng::ArefCreateOp, std::string> &arefPutMap,
    DenseMap<ttng::ArefCreateOp, std::string> &arefGetMap,
    DenseMap<ttng::ArefCreateOp, std::pair<TMEMAref::Kind, int>>
        &arefLastUseMap,
    DenseSet<ttng::ArefCreateOp> &scopedAref) {
  auto arefOp = state.getOp.getAref().getDefiningOp<ttng::ArefCreateOp>();
  createArefGetExitOp(builder, loc, arefOp, {state.consumer}, curGroup);
  arefLastUseMap[arefOp] = {TMEMAref::GET, (int)state.consumer};
  if (!state.curGroup.empty()) {
    arefOp = createAref(arefOp.getOperand(0), true, scopedAref);
    insertGetEnterExit(builder, loc, arefOp, state.consumer, curGroup,
                       arefGetMap, arefLastUseMap);
    insertPutEnterExit(builder, loc, arefOp, ttng::ArefProducer::NONE,
                       state.curGroup, arefPutMap, arefLastUseMap);
    ttng::ArefCreateOp arefOp;
    for (auto [aref, group] : arefGetMap) {
      if (group == state.curGroup) {
        arefOp = aref;
        break;
      }
    }
    assert(arefOp);
    insertGetEnterExit(builder, loc, arefOp, ttng::ArefConsumer::NONE,
                       state.curGroup, arefGetMap, arefLastUseMap);
    state.curGroup = {};
  }
}

// Recursively insert Aref ops along the DAG, handling group transitions.
std::tuple<int, int, ttng::ArefProducer, ttng::ArefConsumer>
insertTmemArefsImpl(TokDag *dag, std::string curGroup,
                    DenseMap<ttng::ArefCreateOp, std::string> &arefPutMap,
                    DenseMap<ttng::ArefCreateOp, std::string> &arefGetMap,
                    DenseMap<ttng::ArefCreateOp, std::pair<TMEMAref::Kind, int>>
                        &arefLastUseMap,
                    DenseSet<ttng::ArefCreateOp> &scopedArefs, TMEMAref state) {
  auto newGroup = dag->group;

  if (newGroup != curGroup) {
    // tmem ownership is changing
    auto loc = dag->op->getLoc();
    OpBuilder builder(dag->op);
    assert(dag->parent);
    if (dag->parent->op) {
      builder.setInsertionPointAfter(dag->parent->op);
    }

    // Create exit operations for current state
    // if previous was put, we need to create a put exit, otherwise get exit
    if (state.putOp) {
      assert(!state.getOp);
      createPutExitOperations(builder, loc, state, curGroup, arefPutMap,
                              arefGetMap, arefLastUseMap, scopedArefs);
    } else {
      createGetExitOperations(builder, loc, state, curGroup, arefPutMap,
                              arefGetMap, arefLastUseMap, scopedArefs);
    }

    // Set insertion point for epilogue optimization
    if (getGroups(dag->op) != std::set<std::string>({ATTR_WS_EPILOGUE})) {
      builder.setInsertionPoint(dag->op);
    }

    // Check if there's a scf::YieldOp in the same group
    bool isYield = false;
    auto dag1 = dag;
    do {
      isYield = isa<scf::YieldOp>(dag1->op);
      if (isYield)
        break;
      dag1 = dag1->user.get();
    } while (dag1 && dag1->group == dag->group);

    // Handle group transition

    // Determine the current aref operation and whether we need get/put op
    ttng::ArefCreateOp arefOp;
    bool needGetEnterOp = false;

    if (isYield) {
      ttng::ArefCreateOp otherArefOp;
      // Handle yield case -- leaving scope
      if (state.scopeGetOp) {
        arefOp = state.scopeGetOp.getAref().getDefiningOp<ttng::ArefCreateOp>();
        if (state.putOp) {
          otherArefOp =
              state.putOp.getAref().getDefiningOp<ttng::ArefCreateOp>();
        }
        needGetEnterOp = true;
      } else {
        arefOp = state.scopePutOp.getAref().getDefiningOp<ttng::ArefCreateOp>();
        if (state.getOp) {
          otherArefOp =
              state.getOp.getAref().getDefiningOp<ttng::ArefCreateOp>();
        }
        needGetEnterOp = false;
      }
      if (arefOp != otherArefOp) {
        scopeArefHandshake(builder, loc, arefOp, needGetEnterOp,
                           (int)state.producer, curGroup, newGroup, arefPutMap,
                           arefGetMap, arefLastUseMap, scopedArefs);
      }
    } else {
      // Handle non-yield case
      if (state.putOp) {
        arefOp = state.putOp.getAref().getDefiningOp<ttng::ArefCreateOp>();
        needGetEnterOp = true;
      } else {
        assert(state.getOp);
        arefOp = state.getOp.getAref().getDefiningOp<ttng::ArefCreateOp>();
        needGetEnterOp = false;
      }

      // When aref put/get is already used by other group, we crate a new aref
      if (needGetEnterOp && arefGetMap.count(arefOp) != 0 &&
          arefGetMap.at(arefOp) != newGroup) {
        assert(state.curGroup.empty());
        if (arefPutMap.count(arefOp)) {
          state.curGroup = arefGetMap.at(arefOp);
        }
        arefOp = createAref(arefOp.getOperand(0), false, scopedArefs);
        insertPutEnterExit(builder, loc, arefOp, state.producer, curGroup,
                           arefPutMap, arefLastUseMap);
      } else if (!needGetEnterOp && arefPutMap.count(arefOp) != 0 &&
                 arefPutMap.at(arefOp) != newGroup) {
        arefOp = createAref(arefOp.getOperand(0), true, scopedArefs);
        assert(state.curGroup.empty());
        if (arefPutMap.count(arefOp)) {
          state.curGroup = arefPutMap.at(arefOp);
        }
        insertGetEnterExit(builder, loc, arefOp, state.consumer, curGroup,
                           arefGetMap, arefLastUseMap);
      }
    }

    // Create the appropriate operation for the new group
    if (needGetEnterOp) {
      // Create get operation for new group
      assert(arefLastUseMap.count(arefOp) != 0);
      assert(arefLastUseMap.at(arefOp).first == TMEMAref::PUT);
      assert(arefGetMap.count(arefOp) == 0 ||
             arefGetMap.at(arefOp) == newGroup);
      arefGetMap[arefOp] = newGroup;

      state.getOp = createArefGetEnterOp(builder, loc, arefOp, newGroup);
      state.consumer = ttng::ArefConsumer::NONE;
      state.token = state.getOp.getTokens()[0];
      state.buffer = state.getOp.getBuffers()[0];
      state.putOp = {};
    } else {
      // Create put operation for new group
      assert(arefLastUseMap.count(arefOp) != 0);
      assert(arefLastUseMap.at(arefOp).first == TMEMAref::GET);
      assert(arefPutMap.count(arefOp) == 0 ||
             arefPutMap.at(arefOp) == newGroup);
      arefPutMap[arefOp] = newGroup;

      state.putOp = createArefPutEnterOp(builder, loc, arefOp, newGroup);
      state.producer = ttng::ArefProducer::NONE;
      state.token = state.putOp.getTokens()[0];
      state.buffer = state.putOp.getBuffers()[0];
      state.getOp = {};
    }
  }

  // Handle sub-DAGs (for loops and if statements)
  int yieldTokPos = -1;
  int yieldBufPos = -1;
  if (!dag->subDags.empty()) {
    if (isa<scf::ForOp>(dag->op)) {
      auto forOp = cast<scf::ForOp>(dag->op);
      int tokPos = findValuePos(forOp.getInitArgs(), state.tmemTok);
      int bufPos = findValuePos(forOp.getInitArgs(), state.tmemBuffer);
      assert(tokPos != -1 && bufPos != -1);

      auto subdagState = state;
      subdagState.tmemBuffer = forOp.getRegionIterArg(bufPos);
      subdagState.tmemTok = forOp.getRegionIterArg(tokPos);
      subdagState.buffer = subdagState.tmemBuffer;
      subdagState.token = subdagState.tmemTok;
      subdagState.scopePutOp = state.putOp;
      subdagState.scopeGetOp = state.getOp;

      assert(dag->subDags.size() == 1);
      DenseSet<ttng::ArefCreateOp> scopedArefs;
      std::tie(yieldTokPos, yieldBufPos, state.producer, state.consumer) =
          insertTmemArefsImpl(dag->subDags.front().get(), newGroup, arefPutMap,
                              arefGetMap, arefLastUseMap, scopedArefs,
                              subdagState);
    } else {
      assert(isa<scf::IfOp>(dag->op));
      assert(dag->subDags.size() == 2);
      auto subdagState = state;
      subdagState.scopePutOp = state.putOp;
      subdagState.scopeGetOp = state.getOp;
      for (auto &subDag : dag->subDags) {
        DenseSet<ttng::ArefCreateOp> scopedArefs;
        std::tie(yieldTokPos, yieldBufPos, state.producer, state.consumer) =
            insertTmemArefsImpl(subDag.get(), newGroup, arefPutMap, arefGetMap,
                                arefLastUseMap, scopedArefs, state);
      }
    }
  }

  // Update state based on current operation
  if (dag->op) {
    if (isa<ttng::MMAv5OpInterface>(dag->op)) {
      state.consumer = ttng::ArefConsumer::UMMA;
      state.producer = ttng::ArefProducer::UMMA;
    } else if (isa<ttng::TMEMStoreOp, ttng::TMEMLoadOp>(dag->op)) {
      state.producer = ttng::ArefProducer::STTM;
      state.consumer = ttng::ArefConsumer::LDTM;
    } else {
      assert((isa<scf::YieldOp, scf::ForOp, scf::IfOp>(dag->op)));
    }

    // Update operation operands
    if (auto tmemLoadOp = dyn_cast<ttng::TMEMLoadOp>(dag->op)) {
      if (tmemLoadOp.getSrc() == state.tmemBuffer) {
        assert(state.buffer);
        tmemLoadOp.getSrcMutable().assign(state.buffer);
        if (state.token) {
          assert(state.tmemTok == tmemLoadOp.getDep());
          tmemLoadOp.getDepMutable().assign(state.token);
          state.token = state.tmemTok = tmemLoadOp.getToken();
        }
      }
    } else if (auto tmemStoreOp = dyn_cast<ttng::TMEMStoreOp>(dag->op)) {
      if (tmemStoreOp.getDst() == state.tmemBuffer) {
        assert(state.buffer);
        tmemStoreOp.getDstMutable().assign(state.buffer);
        assert(state.token);
        assert(state.tmemTok == tmemStoreOp.getDep());
        tmemStoreOp.getDepMutable().assign(state.token);
        state.token = state.tmemTok = tmemStoreOp.getToken();
      }
    } else if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(dag->op)) {
      if (mmaOp.getAccumulator() == state.tmemBuffer) {
        assert(state.buffer);
        mmaOp.setAccumulator(state.buffer);
        assert(state.token);
        assert(state.tmemTok == mmaOp.getAccDep());
        mmaOp.getAccDepMutable().assign(state.token);
        state.token = state.tmemTok = mmaOp.getToken();
      } else if (mmaOp.getA() == state.tmemBuffer) {
        assert(state.buffer);
        mmaOp.getAMutable().assign(state.buffer);
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(dag->op)) {
      assert(yieldTokPos != -1 && yieldBufPos != -1);
      forOp.getInitArgsMutable()[yieldTokPos].assign(state.token);
      forOp.getInitArgsMutable()[yieldBufPos].assign(state.buffer);
      state.token = state.tmemTok = forOp.getResult(yieldTokPos);
      state.buffer = state.tmemBuffer = forOp.getResult(yieldBufPos);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(dag->op)) {
      assert(yieldTokPos != -1 && yieldBufPos != -1);
      state.token = state.tmemTok = ifOp.getResult(yieldTokPos);
      state.buffer = state.tmemBuffer = ifOp.getResult(yieldBufPos);
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(dag->op)) {
      yieldTokPos = findValuePos(yieldOp.getOperands(), state.tmemTok);
      yieldBufPos = findValuePos(yieldOp.getOperands(), state.tmemBuffer);
      yieldOp.setOperand(yieldTokPos, state.token);
      yieldOp.setOperand(yieldBufPos, state.buffer);
    } else {
      llvm_unreachable("unsupported op");
    }
  }

  if (!dag->user) {
    // Handle final operations and cleanup
    OpBuilder builder(dag->op);
    builder.setInsertionPointAfter(dag->op);
    if (!isa<scf::YieldOp>(dag->op)) {
      // Create exit operations for pending arefs
      if (state.putOp) {
        assert(!state.getOp);
        auto loc = state.putOp.getLoc();
        createPutExitOperations(builder, loc, state, newGroup, arefPutMap,
                                arefGetMap, arefLastUseMap, scopedArefs);
      } else {
        auto loc = state.getOp.getLoc();
        createGetExitOperations(builder, loc, state, newGroup, arefPutMap,
                                arefGetMap, arefLastUseMap, scopedArefs);
      }
    }

    // Ensure each put has a matching get and vice versa for scoped arefs
    for (auto arefOp : scopedArefs) {
      assert(arefLastUseMap.count(arefOp) != 0);
      auto loc = arefOp.getLoc();

      if (arefLastUseMap.at(arefOp).first == TMEMAref::GET) {
        if (arefOp->hasAttr("first_get")) {
          auto putGroup = arefPutMap.at(arefOp);
          insertPutEnterExit(builder, loc, arefOp, ttng::ArefProducer::NONE,
                             putGroup, arefPutMap, arefLastUseMap);
        }
      } else if (!arefOp->hasAttr("first_get")) {
        auto getGroup = arefGetMap.at(arefOp);
        insertGetEnterExit(builder, loc, arefOp, ttng::ArefConsumer::NONE,
                           getGroup, arefGetMap, arefLastUseMap);
      }
    }
  } else {
    return insertTmemArefsImpl(dag->user.get(), newGroup, arefPutMap,
                               arefGetMap, arefLastUseMap, scopedArefs, state);
  }

  return {yieldTokPos, yieldBufPos, state.producer, state.consumer};
}

// Entry point: Insert Aref ops for a given TMEM allocation DAG.
void insertTmemArefs(TokDag *dag) {
  assert(dag->op);
  auto curGroup = dag->user->group;
  auto allocOp = cast<ttng::TMEMAllocOp>(dag->op);

  // Create Aref buffer allocation at top level
  OpBuilder builder(allocOp);
  builder.setInsertionPointToStart(
      &allocOp->getParentOfType<tt::FuncOp>().getBody().front());
  TMEMAref state;

  state.tmemBuffer = allocOp.getResult();
  state.tmemTok = allocOp.getToken();
  auto arefBufType = getArefbufMemDescType(allocOp.getResult().getType(), 1);
  auto arefBufAlloc =
      createAlloc(builder, allocOp.getLoc(), arefBufType, Value());
  arefBufAlloc->setAttr("aref_tmem_buffer", builder.getUnitAttr());

  builder.setInsertionPoint(allocOp);
  std::string group;
  if (allocOp.getSrc()) {
    group = dag->group;
  } else {
    group = dag->user->group;
  }

  DenseMap<ttng::ArefCreateOp, std::string> arefPutMap;
  DenseMap<ttng::ArefCreateOp, std::string> arefGetMap;
  DenseMap<ttng::ArefCreateOp, std::pair<TMEMAref::Kind, int>> arefLastUseMap;

  DenseSet<ttng::ArefCreateOp> scopedArefs;
  auto aref = createAref(arefBufAlloc->getResult(0), false, scopedArefs);
  state.putOp = createArefPutEnterOp(builder, allocOp.getLoc(), aref, group);
  arefPutMap[aref] = group;

  state.getOp = {};
  state.scopePutOp = {};
  state.scopeGetOp = {};
  state.buffer = state.putOp.getBuffers()[0];
  state.token = state.putOp.getTokens()[0];
  state.producer = ttng::ArefProducer::NONE;
  state.consumer = ttng::ArefConsumer::NONE;

  // Handle source initialization if present
  if (auto src = allocOp.getSrc()) {
    auto vTrue = builder.create<arith::ConstantIntOp>(allocOp.getLoc(), 1, 1);
    setGroups(vTrue, {group});
    auto store = builder.create<ttng::TMEMStoreOp>(
        allocOp.getLoc(), builder.getType<AsyncTokenType>(), state.buffer,
        state.token, src, vTrue);
    setGroups(store, {group});
    state.producer = ttng::ArefProducer::STTM;
    state.consumer = ttng::ArefConsumer::LDTM;
  }

  insertTmemArefsImpl(dag->user.get(), group, arefPutMap, arefGetMap,
                      arefLastUseMap, scopedArefs, state);
}

Value buildTokDag(Value tok, Operation *user, TokDag *dag,
                  std::string groupScope);
// Build the token DAG for a for-loop, following token dependencies.
Value buildTokDagForOp(scf::ForOp forOp, Value tok, TokDag *dag,
                       std::string groupScope) {
  auto tokPos = findValuePos(forOp.getInitArgs(), tok);
  assert(tokPos != -1);

  auto tokArg = forOp.getRegionIterArg(tokPos);
  assert(tokArg.hasOneUse());
  auto group = getOne(getGroupsIdx(forOp, tokPos));

  // Create a subdag with a null-op as the first op to track tmem ownership
  // by for-op in its body.
  auto subDag = std::make_unique<TokDag>(nullptr, group, dag->parent);
  buildTokDag(tokArg, *tokArg.getUsers().begin(), subDag.get(), group);

  dag->group = group;
  dag->subDags.push_back(std::move(subDag));

  tok = forOp.getResult(tokPos);
  return tok;
}

// Build the token DAG for an if-op, following token dependencies.
Value buildTokDagIfOp(Value tok, TokDag *dag, std::string groupScope) {
  SmallVector<Operation *> users;
  for (auto user : tok.getUsers()) {
    users.push_back(user);
  }
  assert(users.size() == 2 && "expecting two users of a token");
  auto userThen = users[0];
  auto userElse = users[1];
  auto ifOp = dyn_cast<scf::IfOp>(userThen->getParentOp());
  assert(ifOp && "assume if-op, one token use per then/else branch");
  if (ifOp.thenBlock() != userThen->getBlock()) {
    std::swap(userThen, userElse);
  }
  assert(ifOp.thenBlock() == userThen->getBlock());
  assert(ifOp.elseBlock() == userElse->getBlock());

  std::string group;
  int tokPos = -1;
  {
    // Build then-subdag to find token operand position in yield
    auto dagTmp = std::make_unique<TokDag>(nullptr, group, nullptr);
    auto tokThen = buildTokDag(tok, userThen, dagTmp.get(), groupScope);
    tokPos =
        findValuePos(ifOp.thenBlock()->getTerminator()->getOperands(), tokThen);
    // Build else-subdag to verify tokPos is the same as in then-subdag
    auto tokElse = buildTokDag(tok, userThen, dagTmp.get(), groupScope);
    for (auto [pos, arg] :
         llvm::enumerate(ifOp.elseBlock()->getTerminator()->getOperands())) {
      if (arg == tokElse) {
        assert(tokPos == pos);
      }
      // Get groups associated with the tokPos result
      group = getOne(getGroupsIdx(ifOp, tokPos));
    }
  }

  dag->user.reset(new TokDag(ifOp, group, dag));
  auto userDag = dag->user.get();

  auto thenDag = std::make_unique<TokDag>(nullptr, group, dag);
  auto elseDag = std::make_unique<TokDag>(nullptr, group, dag);
  buildTokDag(tok, userThen, thenDag.get(), group);
  buildTokDag(tok, userElse, elseDag.get(), group);
  userDag->group = group;
  userDag->subDags.push_back(std::move(thenDag));
  userDag->subDags.push_back(std::move(elseDag));

  tok = ifOp->getResult(tokPos);
  assert(tok.hasOneUse());
  return buildTokDag(tok, *tok.getUsers().begin(), userDag, groupScope);
}

// Build the token DAG for a generic user operation.
Value buildTokDag(Value tok, Operation *user, TokDag *dag,
                  std::string groupScope) {
  dag->user.reset(new TokDag(user, groupScope, dag));
  auto userDag = dag->user.get();

  if (isa<scf::YieldOp>(user)) {
    return tok; // propagate the token back all the way to the caller
  } else if (auto tmemLoad = dyn_cast<ttng::TMEMLoadOp>(user)) {
    tok = tmemLoad.getToken();
    userDag->group = getOne(getGroups(user));
  } else if (auto tmemStore = dyn_cast<ttng::TMEMStoreOp>(user)) {
    tok = tmemStore.getToken();
    userDag->group = getOne(getGroups(user));
  } else if (auto mmav5 = dyn_cast<ttng::MMAv5OpInterface>(user)) {
    tok = mmav5.getToken();
    userDag->group = getOne(getGroups(user));
  } else if (auto forOp = dyn_cast<scf::ForOp>(user)) {
    tok = buildTokDagForOp(forOp, tok, userDag, groupScope);
  } else {
    llvm_unreachable("unsupported user");
  }

  if (tok.use_empty()) {
    return tok;
  }

  if (tok.hasOneUse()) {
    return buildTokDag(tok, *tok.getUsers().begin(), userDag, groupScope);
  }

  // Currently, we only handle the case where a token has multiple uses in
  // an if-op, one in then & one in else branches. If we encounter
  // tokens being used in other ways, this code will need to be updated to
  // handle those cases.
  return buildTokDagIfOp(tok, userDag, groupScope);
}

static void printTokDag(TokDag *dag, int indent, llvm::raw_ostream &os) {
  for (int i = 0; i < indent; i++) {
    os << " ";
  }
  std::set<std::string> groups{dag->group};
  if (dag->op) {
    os << "|- " << dag->op->getName().getStringRef() << " ";
    if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(dag->op)) {
      groups = getGroups(dag->op);
      if (tmemAlloc.getSrc()) {
        os << " %src ";
      }
    }
  }
  os << "  [";
  for (auto group : groups) {
    os << " @" << group << " ";
  }
  os << "]\n";
  for (auto &subDag : dag->subDags) {
    printTokDag(subDag.get(), indent + 4, os);
  }
  if (dag->user) {
    printTokDag(dag->user.get(), indent, os);
  }
};
static std::set<std::string> collectGroups(TokDag *dag) {
  std::set<std::string> groups{dag->group};
  while (dag->user) {
    dag = dag->user.get();
    groups.insert(dag->group);
    for (auto &subDag : dag->subDags) {
      auto gs = collectGroups(subDag.get());
      groups.insert(gs.begin(), gs.end());
    }
  }
  return groups;
};

// Run Aref insertion on a Triton function.
static void runOnFunc(triton::FuncOp funcOp) {

  // Build DAG for all allocations
  SmallVector<TokDag> tmemDags;
  funcOp.walk([&](ttng::TMEMAllocOp allocOp) {
    auto group = *getGroups(allocOp).begin();
    if (auto src = allocOp.getSrc()) {
      tmemDags.push_back(TokDag{allocOp, group, nullptr});
      auto dag = &tmemDags.back();
      dag->op = allocOp;
      assert(allocOp->hasOneUse());
      auto &use = *allocOp->getUses().begin();
      dag->user.reset(new TokDag{
          use.getOwner(), getOne(getGroups(use.getOwner())), &tmemDags.back()});
    } else {
      tmemDags.push_back(TokDag{allocOp, group, nullptr});
      auto dag = &tmemDags.back();
      dag->op = allocOp;
      auto tok = allocOp.getToken();
      assert(tok.hasOneUse());
      buildTokDag(tok, *tok.getUsers().begin(), dag, {});
    }
  });

  // Process DAGs, one per tmem alloc
  for (auto [idx, dag] : llvm::enumerate(tmemDags)) {
    LLVM_DEBUG({
      auto &os = DBGS();
      os << "TMEMDAG[" << idx << "]:\n";
      printTokDag(&dag, 2, os);
    });
    if (collectGroups(&dag).size() == 1) {
      continue;
    }
    insertTmemArefs(&dag);
  }
}
} // namespace arefTMEMInsertion

// ThreadBuffer: Handles threading of buffer operations through control flow.
namespace threadBuffer {
struct BufTok {
  Value tok;
  Value buf;
};
BufTok threadBuffer(OpBuilder &builder, Operation *user, BufTok buftok);

// Thread buffer through a for-loop.
BufTok threadForOp(OpBuilder &builder, scf::ForOp forOp, BufTok buftok) {
  assert(buftok.tok.hasOneUse());
  auto tokPos = findValuePos(forOp.getInitArgs(), buftok.tok);
  assert(tokPos != -1);
  builder.setInsertionPoint(forOp);
  buftok.tok = forOp.getRegionIterArg(tokPos);
  buftok.buf = addIterArgsToLoop(builder, forOp, buftok.buf).front();
  setGroupsIdx(forOp, forOp.getResults().size() - 1,
               getGroupsIdx(forOp, tokPos));
  assert(buftok.tok.hasOneUse());
  threadBuffer(builder, *buftok.tok.getUsers().begin(), buftok);
  buftok.tok = forOp.getResult(tokPos);
  buftok.buf = forOp.getResults().back();
  return buftok;
}

// Thread buffer through an if-op.
BufTok threadIfOp(OpBuilder &builder, BufTok buftok) {
  SmallVector<Operation *> users;
  for (auto user : buftok.tok.getUsers()) {
    users.push_back(user);
  }
  assert(users.size() == 2 && "expecting two users of a token");
  auto userThen = users[0];
  auto userElse = users[1];
  auto ifOp = dyn_cast<scf::IfOp>(userThen->getParentOp());
  assert(ifOp && "assume if-op, one token use per then/else branch");
  if (ifOp.thenBlock() != userThen->getBlock()) {
    std::swap(userThen, userElse);
  }
  assert(ifOp.thenBlock() == userThen->getBlock());
  assert(ifOp.elseBlock() == userElse->getBlock());

  auto newIfOp =
      replaceIfOpWithNewSignature(builder, ifOp, {buftok.buf.getType()});
  ifOp.erase();

  auto buftokThen = threadBuffer(builder, userThen, buftok);
  auto buftokElse = threadBuffer(builder, userElse, buftok);
  int tokPos = -1, bufPos = -1;
  for (auto [pos, arg] :
       llvm::enumerate(newIfOp.thenBlock()->getTerminator()->getOperands())) {
    if (arg == buftokThen.tok) {
      assert(tokPos == -1);
      tokPos = pos;
    }
    if (arg == buftokThen.buf) {
      assert(bufPos == -1);
      bufPos = pos;
    }
  }
  assert(tokPos != -1 && bufPos != -1);
  setGroupsIdx(newIfOp, bufPos, getGroupsIdx(newIfOp, tokPos));
  buftok.tok = newIfOp->getResult(tokPos);
  buftok.buf = newIfOp->getResult(bufPos);
  assert(buftok.tok.hasOneUse());
  return threadBuffer(builder, *buftok.tok.getUsers().begin(), buftok);
}

// Recursively thread buffer through users.
BufTok threadBuffer(OpBuilder &builder, Operation *user, BufTok buftok) {
  if (auto tmemStore = dyn_cast<ttng::TMEMStoreOp>(user)) {
    buftok.tok = tmemStore.getToken();
    tmemStore.getDstMutable().assign(buftok.buf);
  } else if (auto tmemLoad = dyn_cast<ttng::TMEMLoadOp>(user)) {
    buftok.tok = tmemLoad.getToken();
    tmemLoad.getSrcMutable().assign(buftok.buf);
  } else if (auto mmav5 = dyn_cast<ttng::MMAv5OpInterface>(user)) {
    buftok.tok = mmav5.getToken();
    mmav5.setAccumulator(buftok.buf);
  } else if (auto forOp = dyn_cast<scf::ForOp>(user)) {
    buftok = threadForOp(builder, forOp, buftok);
  } else if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
    SmallVector<Value> operands(yieldOp->getOperands());
    operands.push_back(buftok.buf);
    builder.setInsertionPoint(yieldOp);
    builder.create<scf::YieldOp>(yieldOp->getLoc(), operands);
    yieldOp->erase();
    return buftok;
  } else {
    llvm_unreachable("unsupported user");
  }

  if (buftok.tok.use_empty()) {
    return buftok;
  }

  if (buftok.tok.hasOneUse()) {
    return threadBuffer(builder, *buftok.tok.getUsers().begin(), buftok);
  }

  // Currently, we only handle the case where a token has multiple uses in
  // an if-op, one in then & one in else branches. If we encounter
  // tokens being used in other ways, this code will need to be updated to
  // handle those cases.
  return threadIfOp(builder, buftok);
}

// Run buffer threading on a Triton function.
void runOnFunc(tt::FuncOp funcOp) {
  SmallVector<ttng::TMEMAllocOp> allocOps;
  funcOp.walk([&](ttng::TMEMAllocOp allocOp) {
    if (!allocOp.getSrc()) {
      allocOps.push_back(allocOp);
    }
  });
  OpBuilder builder(funcOp);
  for (auto allocOp : allocOps) {
    BufTok buftok{allocOp.getToken(), allocOp.getResult()};
    buftok = threadBuffer(builder, *buftok.tok.getUsers().begin(), buftok);
  }
}
}; // namespace threadBuffer

// Remove TMEMAlloc ops that are no longer used.
void removeStaleAllocs(tt::FuncOp funcOp) {
  SmallVector<ttng::TMEMAllocOp> allocs;
  funcOp.walk([&](ttng::TMEMAllocOp allocOp) { allocs.push_back(allocOp); });
  for (auto allocOp : allocs) {
    if (allocOp.use_empty()) {
      allocOp->erase();
    }
  }
}

// Update ArefExitOp attributes to ensure correct producer/consumer info
void updateExitOpAttributes(tt::FuncOp funcOp) {
  SmallVector<ttng::ArefCreateOp> arefOps;
  funcOp.walk([&](ttng::ArefCreateOp arefOp) { arefOps.push_back(arefOp); });
  for (auto arefOp : arefOps) {
    DenseSet<ttng::ArefProducer> producers;
    DenseSet<ttng::ArefConsumer> consumers;
    for (auto user : arefOp->getUsers()) {
      if (auto exitOp = dyn_cast<ttng::ArefExitOpInterface>(user)) {
        auto attrs = exitOp.getAssociatedOpAttrs();
        assert(attrs.size() == 1);
        auto attr = attrs[0];
        if (auto producer = dyn_cast<ttng::ArefProducerAttr>(attr)) {
          if (producer.getValue() != ttng::ArefProducer::NONE) {
            producers.insert(producer.getValue());
          }
        } else if (auto consumer = dyn_cast<ttng::ArefConsumerAttr>(attr)) {
          if (consumer.getValue() != ttng::ArefConsumer::NONE) {
            consumers.insert(consumer.getValue());
          }
        }
      }
    }
    if (producers.empty()) {
      producers.insert(ttng::ArefProducer::STTM);
    }
    if (consumers.empty()) {
      consumers.insert(ttng::ArefConsumer::LDTM);
    }
    SmallVector<Attribute> producersAttr, consumersAttr;
    for (auto producer : producers) {
      producersAttr.push_back(
          ttng::ArefProducerAttr::get(arefOp->getContext(), producer));
    }
    for (auto consumer : consumers) {
      consumersAttr.push_back(
          ttng::ArefConsumerAttr::get(arefOp->getContext(), consumer));
    }
    auto producersAttrArray =
        ArrayAttr::get(arefOp->getContext(), producersAttr);
    auto consumersAttrArray =
        ArrayAttr::get(arefOp->getContext(), consumersAttr);
    for (auto user : arefOp->getUsers()) {
      if (auto exitOp = dyn_cast<ttng::ArefExitOpInterface>(user)) {
        auto attrArray = isa<ttng::ArefPutExitOp>(exitOp) ? producersAttrArray
                                                          : consumersAttrArray;
        exitOp.setAssociatedOpAttrs(attrArray);
      }
    }
  }
}
// By default, we thread buffers through loops in case aref ops modify it:
//    %buf = put.enter  %aref
//    for .. iter_args(%buf0 = vbuf) {
//       put.exit %aref
//       ..
//       %buf1 = put.enter %aef
//       yield %buf1
//    }
// If a buffer iter_arg is yielded unchanged in each iteration, we can remove it
// from the loop and use the original buffer directly since it's not modified.

void unthreadbuffer(tt::FuncOp funcOp) {
  SmallVector<Operation *> op2erase;
  funcOp.walk([&](ttng::ArefEnterOpInterface enterOp) {
    auto buffer = enterOp.getBuffers()[0];
    for (auto user : buffer.getUsers()) {
      if (!isa<scf::ForOp>(user)) {
        continue;
      }
      auto forOp = cast<scf::ForOp>(user);
      int bufPos = findValuePos(forOp.getInitArgs(), buffer);
      assert(bufPos != -1);
      auto oldBufArg = forOp.getRegionIterArg(bufPos);
      bool canUnthread =
          oldBufArg == cast<scf::YieldOp>(forOp.getBody()->getTerminator())
                           .getOperand(bufPos);

      if (!canUnthread) {
        continue;
      }

      OpBuilder builder(forOp);

      // Create a new loop before the existing one, with the extra operands.
      SmallVector<Value> operands;
      for (auto [pos, arg] : llvm::enumerate(forOp.getInitArgs())) {
        if (pos != bufPos) {
          operands.push_back(arg);
        }
      }
      scf::ForOp newLoop = builder.create<scf::ForOp>(
          forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
          forOp.getStep(), operands);
      newLoop->setAttrs(forOp->getAttrs());
      newLoop.getBody()->erase();
      newLoop.getRegion().getBlocks().splice(
          newLoop.getRegion().getBlocks().begin(),
          forOp.getRegion().getBlocks());

      for (auto [result, value] :
           llvm::zip(forOp.getResults(), newLoop.getResults())) {
        result.replaceAllUsesWith(value);
      }
      for (int oldIdx = 0, newIdx = 0; oldIdx < forOp.getResults().size();
           ++oldIdx) {
        if (oldIdx == bufPos)
          continue;
        forOp.getResult(oldIdx).replaceAllUsesWith(newLoop.getResult(newIdx++));
      }
      auto body = newLoop.getBody();
      auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
      SmallVector<Value> yieldOperands;
      for (auto [pos, operand] : llvm::enumerate(yieldOp.getOperands())) {
        if (pos != bufPos) {
          yieldOperands.push_back(operand);
        }
      }
      builder.setInsertionPoint(yieldOp);
      builder.create<scf::YieldOp>(yieldOp.getLoc(), yieldOperands);
      yieldOp.erase();
      for (int pos = bufPos; pos < newLoop.getNumResults(); pos++) {
        setGroupsIdx(newLoop, pos, getGroupsIdx(newLoop, bufPos + 1));
      }
      auto newBufArg = newLoop.getBody()->getArgument(bufPos + 1);
      newBufArg.replaceAllUsesWith(buffer);
      body->eraseArgument(bufPos + 1);
      op2erase.push_back(forOp);
    }
  });
  for (auto op : op2erase) {
    op->erase();
  }
}

// Main pass: Applies all Aref/ThreadBuffer transformations to each Triton
// function in the module.
class NVWSArefTMEMInsertion
    : public NVWSArefTMEMInsertionBase<NVWSArefTMEMInsertion> {
public:
  void runOnOperation() override {
    auto mod = getOperation();
    mod.walk([&](triton::FuncOp funcOp) {
      threadBuffer::runOnFunc(funcOp);
      arefTMEMInsertion::runOnFunc(funcOp);
      removeStaleAllocs(funcOp);
      updateExitOpAttributes(funcOp);
      unthreadbuffer(funcOp);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createNVWSArefTMEMInsertionPass() {
  return std::make_unique<NVWSArefTMEMInsertion>();
}
