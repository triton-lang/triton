#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stack>

#include "WSUtility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace triton {
namespace gpu {

namespace ttng = triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONGPUSPLITWARPGROUPLOOPS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritongpu-split-warp-group-loops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
MemDescType getDataMemDescType(MemDescType memDescType) {
  auto shape = memDescType.getShape();
  SmallVector<int64_t> dataShape(shape.begin() + 1, shape.end());
  return MemDescType::get(dataShape, memDescType.getElementType(),
                          memDescType.getEncoding(),
                          memDescType.getMemorySpace(), true);
};
MemDescType getArefbufMemDescType(MemDescType memDescType, int32_t AREF_SIZE) {
  auto shape = memDescType.getShape();
  SmallVector<int64_t> bufferShape(shape.begin(), shape.end());
  bufferShape.insert(bufferShape.begin(), AREF_SIZE);
  return MemDescType::get(bufferShape, memDescType.getElementType(),
                          memDescType.getEncoding(),
                          memDescType.getMemorySpace(), true);
};

// test if targetValue is a predecessor of startValue (not necessarily
// immediate)
bool isValueUsedIn(mlir::Value targetValue, mlir::Value startValue,
                   llvm::SetVector<mlir::Value> &visited) {
  if (startValue == targetValue)
    return true;

  // Check if we've already visited this value
  if (!visited.insert(startValue))
    return false;

  mlir::Operation *defOp = startValue.getDefiningOp();
  if (!defOp)
    return false; // Block arguments have no defining operation

  if (isMMAOp(defOp))
    return false;

  // recursively check all operands of the defining operation
  for (mlir::Value operand : defOp->getOperands()) {
    if (isValueUsedIn(targetValue, operand, visited))
      return true;
  }

  return false;
}

// the criteria for reusing aref is that the number of loads in the loop and
// tensor types of the load ops are exactly the same. in causal case, the two
// loops are back to back so we know we can reuse the aref in the second loop
bool canReuseAref(MLIRContext *ctx, SmallVector<MemDescType> &memDescTypes,
                  SmallVector<triton::nvidia_gpu::ArefCreateOp> &arefOps,
                  bool isAllUsedInDot, int AREF_SIZE) {
  if (isAllUsedInDot) {
    SmallVector<Type> arefBuffersMemDescType;
    for (auto memDescType : memDescTypes) {
      auto bufferMemDescType = getArefbufMemDescType(memDescType, AREF_SIZE);
      arefBuffersMemDescType.push_back(bufferMemDescType);
    }
    auto arefTy = triton::nvidia_gpu::ArefType::get(
        ctx, TypeArrayAttr::get(ctx, arefBuffersMemDescType));
    if (arefOps.size() != 1) {
      std::cout << "arefOps size is not 1 in tuple case\n";
      return false;
    }
    if (arefOps[0].getType() != arefTy) {
      std::cout << "arefOps type mismatch in tuple case\n";
      return false;
    }
    return true;
  } else {
    if (arefOps.size() != memDescTypes.size()) {
      std::cout << "arefOps size is not equal to loadOps size\n";
      return false;
    }
    // create separate aref for each load
    for (auto [arefOp, memDescType] : llvm::zip(arefOps, memDescTypes)) {
      auto bufferMemDescType = getArefbufMemDescType(memDescType, AREF_SIZE);
      auto arefTy = triton::nvidia_gpu::ArefType::get(
          ctx, TypeArrayAttr::get(ctx, {bufferMemDescType}));
      if (arefOp.getType() != arefTy) {
        std::cout << "arefOps type mismatch in separate case\n";
        return false;
      }
    }
    return true;
  }
}

// check if output of all ops are eventually only used in dotOp
bool opsUsedinDot(Operation *dotOp, SmallVector<Value> &ops) {
  return std::all_of(ops.begin(), ops.end(), [&](Value val) {
    for (Value operand : dotOp->getOperands()) {
      llvm::SetVector<mlir::Value> visited;
      if (isValueUsedIn(val, operand, visited)) {
        return true;
      }
    }
    return false;
  });
}

// check if all the loads in origLoadOps are only used in the same dot op in the
// forOp
bool loadOpsAllUsedInDot(scf::ForOp &forOp, SmallVector<Value> &origLoadOps) {
  bool result = false;
  forOp.walk([&](Operation *op) {
    if (isMMAOp(op)) {
      if (opsUsedinDot(op, origLoadOps)) {
        result = true;
        return WalkResult::interrupt();
      }
    }
    // if all used in this op, we are done
    // if not in this dot op, check if the ops are used in another dot op
    return result ? WalkResult::skip() : WalkResult::advance();
  });
  return result;
}

void createArefCreate(OpBuilderWithGroup &builder, Location loc,
                      SmallVector<MemDescType> &memDescTypes,
                      SmallVector<triton::nvidia_gpu::ArefCreateOp> &arefOps,
                      int AREF_SIZE, bool isAllUsedInDot) {
  auto doAlloc = [&](MemDescType memDesc) -> Value {
    if (isa<triton::gpu::SharedMemorySpaceAttr>(memDesc.getMemorySpace())) {
      return builder.create<LocalAllocOp>(loc, memDesc, Value());
    } else if (isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(
                   memDesc.getMemorySpace())) {
      return builder.create<triton::nvidia_gpu::TMEMAllocOp>(loc, memDesc,
                                                             Value());
    }
    llvm_unreachable("unknown memory space");
    return Value();
  };

  if (isAllUsedInDot) {
    auto ctx = builder.getContext();
    SmallVector<Value> arefBuffers;
    SmallVector<Type> arefBuffersMemDescType;
    for (auto memDescType : memDescTypes) {
      auto bufferMemDescType = getArefbufMemDescType(memDescType, AREF_SIZE);
      auto arefBuffer = doAlloc(bufferMemDescType);
      arefBuffer.getDefiningOp()->setAttr("aref_buffer", builder.getUnitAttr());
      arefBuffers.push_back(arefBuffer);
      arefBuffersMemDescType.push_back(bufferMemDescType);
    }

    auto arefTy = triton::nvidia_gpu::ArefType::get(
        ctx, TypeArrayAttr::get(ctx, arefBuffersMemDescType));
    auto aref = builder.create<triton::nvidia_gpu::ArefCreateOp>(loc, arefTy,
                                                                 arefBuffers);
    arefOps.push_back(aref);
  } else {
    for (auto memDescType : memDescTypes) {
      auto ctx = builder.getContext();
      auto bufferMemDescType = getArefbufMemDescType(memDescType, AREF_SIZE);
      auto arefBuffer = doAlloc(bufferMemDescType);
      arefBuffer.getDefiningOp()->setAttr("aref_buffer", builder.getUnitAttr());
      auto arefTy = triton::nvidia_gpu::ArefType::get(
          ctx, TypeArrayAttr::get(ctx, bufferMemDescType));
      auto aref = builder.create<triton::nvidia_gpu::ArefCreateOp>(loc, arefTy,
                                                                   arefBuffer);
      arefOps.push_back(aref);
    }
  }
}

void createArefPut(OpBuilderWithGroup &builder, scf::ForOp &forOp,
                   scf::ForOp &newForOp, SmallVector<Value> &origLoadOps,
                   SmallVector<Value> &loadOps,
                   SmallVector<triton::nvidia_gpu::ArefCreateOp> &arefOps,
                   int AREF_SIZE, Value normLoadIdx, bool isAllUsedInDot) {
  // create aref_put[normLoadIdx].
  SmallVector<Value> indices;
  if (isAllUsedInDot) {
    builder.setInsertionPointAfter(loadOps[loadOps.size() - 1].getDefiningOp());
    // create single ArefPut
    auto arefOp = arefOps[0];
    auto aref_put = builder.create<triton::nvidia_gpu::ArefPutOp>(
        forOp.getLoc(), arefOp, normLoadIdx);
    auto &region = aref_put.getRegion();
    auto block = &region.emplaceBlock();

    SmallVector<Value> memDescs, buffers;
    for (auto operand : arefOp.getOperands()) {
      auto memDescType = mlir::cast<MemDescType>(operand.getType());
      assert(memDescType);
      auto dataMemDescType = getDataMemDescType(memDescType);
      auto buffer = block->addArgument(dataMemDescType, operand.getLoc());
      buffers.push_back(buffer);
    }
    builder.setInsertionPointToEnd(block);
    auto ret = builder.create<triton::nvidia_gpu::ArefReturnOp>(
        forOp.getLoc(), ArrayRef<Type>(), ArrayRef<Value>());
    builder.setInsertionPoint(ret);

    for (auto [loadOp, buffer] : llvm::zip(loadOps, buffers)) {
      auto loadOpDef = loadOp.getDefiningOp();
      loadOpDef->moveBefore(ret);
      builder.create<LocalStoreOp>(loadOpDef->getLoc(), loadOpDef->getResult(0),
                                   buffer);
    }
  } else {
    // create ArefPut per load
    for (int i = 0; i < loadOps.size(); i++) {
      builder.setInsertionPointAfter(loadOps[i].getDefiningOp());
      auto arefOp = arefOps[i];
      auto aref_put = builder.create<triton::nvidia_gpu::ArefPutOp>(
          forOp.getLoc(), arefOp, normLoadIdx);
      auto &region = aref_put.getRegion();
      auto block = &region.emplaceBlock();

      assert(arefOp.getOperands().size() == 1);
      auto operand = arefOp.getOperand(0);
      auto memDescType = mlir::cast<MemDescType>(operand.getType());
      assert(memDescType);
      auto dataMemDescType = getDataMemDescType(memDescType);
      auto buffer = block->addArgument(dataMemDescType, operand.getLoc());
      builder.setInsertionPointToEnd(block);
      auto ret = builder.create<triton::nvidia_gpu::ArefReturnOp>(
          forOp.getLoc(), ArrayRef<Type>(), ArrayRef<Value>());
      builder.setInsertionPoint(ret);

      auto loadOpDef = loadOps[i].getDefiningOp();
      loadOpDef->moveBefore(ret);
      builder.create<LocalStoreOp>(loadOpDef->getLoc(), loadOpDef->getResult(0),
                                   buffer);
    }
  }
}

void createArefGet(OpBuilderWithGroup &builder, Location loc,
                   scf::ForOp &newForOp, SmallVector<Value> &loadOps,
                   SmallVector<triton::nvidia_gpu::ArefCreateOp> &arefOps,
                   IRMapping &mapping, Value normMathIdx, int AREF_SIZE) {
  // create aref_get[normMathIdx % D].
  SmallVector<Value> indices;
  auto depth = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerType(32),
      builder.getIntegerAttr(builder.getIntegerType(32), AREF_SIZE));
  if (arefOps.size() == 1) {
    auto arefOp = arefOps[0];
    SmallVector<Type> memDescTypeBufs;
    for (auto operand : arefOp.getOperands()) {
      auto memDescType = mlir::cast<MemDescType>(operand.getType());
      assert(memDescType);
      Type type = getDataMemDescType(memDescType);
      memDescTypeBufs.push_back(type);
    }

    auto aref_get = builder.create<triton::nvidia_gpu::ArefGetEnterOp>(
        loc, memDescTypeBufs, arefOp, normMathIdx);
    for (int i = 0; i < loadOps.size(); i++) {
      mapping.map(loadOps[i], aref_get.getResult(i));
    }
  } else {
    // separate aref
    for (int i = 0; i < loadOps.size(); i++) {
      auto arefOp = arefOps[i];
      assert(arefOp.getOperands().size() == 1);
      auto operand = arefOp.getOperand(0);
      auto memDescType = mlir::cast<MemDescType>(operand.getType());
      assert(memDescType);
      auto dataMemDescType = getDataMemDescType(memDescType);
      auto aref_get = builder.create<triton::nvidia_gpu::ArefGetEnterOp>(
          loc, dataMemDescType, arefOp, normMathIdx);
      mapping.map(loadOps[i], aref_get.getResult(0));
    }
  }
}

Operation *updateWarpGroupDotOp(scf::ForOp forOp, OpBuilderWithGroup &builder,
                                int pendings, int numWarps) {
  SmallVector<triton::nvidia_gpu::WarpGroupDotWaitOp> waitOps;
  SmallVector<triton::nvidia_gpu::WarpGroupDotOp> dotOps;
  forOp.walk([&](triton::nvidia_gpu::WarpGroupDotOp dotOp) {
    dotOp.setIsAsync(true);
    SetVector<Value> operands;
    operands.insert(dotOp.getResult());
    builder.setInsertionPointAfter(dotOp);

    // pendings = N means there are *at most* N dots on the fly
    // only when the # of exisiting dots is larger than N, it will block
    auto waitOp = builder.create<triton::nvidia_gpu::WarpGroupDotWaitOp>(
        dotOp.getLoc(), llvm::to_vector(operands), /*pendings*/ pendings);
    dotOp.getResult().replaceAllUsesExcept(waitOp.getResult(0), waitOp);
    waitOps.push_back(waitOp);
    dotOps.push_back(dotOp);
  });
  if (dotOps.size() > 1 || waitOps.size() == 0 || pendings == 0 ||
      forOp.getNumResults() == 0) {
    return nullptr;
  }
  /* Wait for all the remaining dot operations
   *
   * A bit tricky here, we need to wait the results before the results
   * are converted and written back to the global memory (tt.store)
   * Two cases we have seen currently:
   * 1. The final results are stored back outside the loop (non-persistent)
   *    %res = for (...) {
   *      %dot = wgmma
   *      %dot_wait = wgmma_wait %dot {pendings=1}
   *      yield %dot_wait
   *    }
   *    ... <- need to ensure %res is ready
   *           by inserting a wait {pendings=0} here
   *    tt.store %res
   *
   * 2. The final results are stored back inside the loop (persistent)
   *    for (...) {
   *      %dot = wgmma
   *      %dot_wait = wgmma_wait %dot {pendings=1}
   *      ... some other operations used %dot_wait
   *      if (last k_tile) {
   *        ... <- need to insert a wait {pendings=0} here
   *        %dot_fp16 = convert %dot_wait
   *        tt.store %dot_fp16
   *      }
   *    }
   *
   * So we find the users of the dot_wait operation, if the parent
   * block of the users changes, we insert the wait operation there.
   */
  Operation *lastWarpGroupDotWait = nullptr;

  // XXX: Hack to insert barrier
  const auto NWARP_THRESHOLD = 4;
  for (auto waitOp : waitOps) {
    for (auto user : waitOp.getResult(0).getUsers()) {
      if (user->getBlock() != forOp.getBody()) {
        builder.setInsertionPoint(user);
        lastWarpGroupDotWait =
            builder.create<triton::nvidia_gpu::WarpGroupDotWaitOp>(
                forOp.getLoc(), ArrayRef<Value>{waitOp.getResult(0)},
                /*pendings*/ 0);
        user->replaceUsesOfWith(waitOp.getResult(0),
                                lastWarpGroupDotWait->getResult(0));
        // XXX: Hack to insert
        if (numWarps > NWARP_THRESHOLD) {
          // use id 2 for named barrier
          auto barId =
              builder.create<arith::ConstantIntOp>(forOp.getLoc(), 2, 32);
          auto numThreads = builder.create<arith::ConstantIntOp>(
              forOp.getLoc(), numWarps * 32, 32);
          builder.create<NVVM::BarrierOp>(forOp.getLoc(), barId, numThreads);
        }
        break;
      }
    }
  }
  if (!lastWarpGroupDotWait) {
    builder.setInsertionPointAfter(forOp);
    // FIXME: the wgmma result may not be the 0-th result of the forOp
    lastWarpGroupDotWait =
        builder.create<triton::nvidia_gpu::WarpGroupDotWaitOp>(
            forOp.getLoc(), ArrayRef<Value>{forOp.getResult(0)},
            /*pendings*/ 0);
    forOp.getResult(0).replaceAllUsesExcept(lastWarpGroupDotWait->getResult(0),
                                            lastWarpGroupDotWait);
    if (numWarps > NWARP_THRESHOLD) {
      // use id 2 for named barrier
      auto barId = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 2, 32);
      auto numThreads = builder.create<arith::ConstantIntOp>(forOp.getLoc(),
                                                             numWarps * 32, 32);
      builder.create<NVVM::BarrierOp>(forOp.getLoc(), barId, numThreads);
    }
  }
  return lastWarpGroupDotWait;
}

scf::ForOp createNewForOp(scf::ForOp forOp, OpBuilderWithGroup &builder,
                          const IRMapping &mapping,
                          const SmallVector<Value> &loopArgs) {
  auto newLowerBound = mapping.lookupOrDefault(forOp.getLowerBound());
  auto newUpperBound = mapping.lookupOrDefault(forOp.getUpperBound());
  auto newStep = mapping.lookupOrDefault(forOp.getStep());

  return builder.create<scf::ForOp>(forOp.getLoc(), newLowerBound,
                                    newUpperBound, newStep, loopArgs);
}

scf::ForOp createLoadGroup(
    scf::ForOp forOp, OpBuilderWithGroup &builder, int forOpIdx, int numStages,
    SmallVector<triton::nvidia_gpu::ArefCreateOp> &arefOps, IRMapping &mapping,
    Operation *arefCreatePoint, Value normIdxInit) {
  if (!isOpInGroup(forOp, ATTR_WS_TMALOAD)) {
    return nullptr;
  }

  auto innerMostLoop = true;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (isa<scf::ForOp>(op) && isOpInGroup(&op, ATTR_WS_TMALOAD)) {
      innerMostLoop = false;
    }
  }

  if (!innerMostLoop) {
    SmallVector<Value> loopArgs{forOp.getInitArgs()};
    loopArgs.push_back(normIdxInit);

    auto newForOp = createNewForOp(forOp, builder, mapping, loopArgs);
    auto normLoadIdxVar = newForOp.getRegionIterArgs().back();

    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
    for (unsigned i = 0; i < forOp.getRegionIterArgs().size(); ++i) {
      auto oldArg = forOp.getRegionIterArgs()[i];
      auto newArg = newForOp.getRegionIterArgs()[i];
      mapping.map(oldArg, newArg);
    }

    scf::ForOp newInnerFor;
    builder.setInsertionPointToStart(newForOp.getBody());

    for (auto &op : forOp.getBody()->without_terminator()) {
      if (!isOpInGroup(&op, ATTR_WS_TMALOAD)) {
        continue;
      }
      if (auto innerFor = dyn_cast<scf::ForOp>(op)) {
        assert(!newInnerFor);
        newInnerFor =
            createLoadGroup(innerFor, builder, forOpIdx, numStages, arefOps,
                            mapping, arefCreatePoint, normLoadIdxVar);
      } else {
        builder.clone(op, mapping);
      }
    }
    assert(newInnerFor);
    auto nextNormLoadIdx = newInnerFor.getResults().back();

    SmallVector<Value> yieldValues;
    for (const auto &v : forOp.getBody()->getTerminator()->getOperands()) {
      yieldValues.push_back(mapping.lookupOrDefault(v));
    }
    yieldValues.push_back(nextNormLoadIdx);
    builder.setInsertionPointToEnd(newForOp.getBody());
    builder.create<scf::YieldOp>(newForOp.getLoc(), yieldValues);
    return newForOp;
  } else {
    // gather memDescTypes and erase localAllocOps
    // FIXME: This code creates temporarily invalid IR where the result of
    // experiemntal_descriptor_load is fed to MMA.
    SmallVector<MemDescType> memDescTypes;
    SmallVector<Operation *> ops2erase;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (isa<triton::DescriptorLoadOp, triton::LoadOp>(op)) {
        SmallVector<Operation *> users(op.user_begin(), op.user_end());
        assert(users.size() == 1);
        auto localAllocOp = users[0];
        assert(isa<LocalAllocOp>(localAllocOp));
        assert(localAllocOp->getNumOperands() == 1);
        auto input = localAllocOp->getOperand(0);
        localAllocOp->getResult(0).replaceAllUsesWith(input);
        auto type = localAllocOp->getResult(0).getType();
        auto memDescType = mlir::cast<MemDescType>(type);
        assert(memDescType);
        memDescTypes.push_back(memDescType);
        ops2erase.push_back(localAllocOp);
      }
    }
    for (auto op : ops2erase) {
      op->erase();
    }
    ops2erase.clear();

    int AREF_SIZE = numStages;

    std::map<int, int> oldLoopId2New;
    SmallVector<Value> loopArgs;
    for (const auto &v :
         llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
      auto attrs =
          getGroupsAttr(forOp.getOperation(), std::string(ATTR_WSGROUPS) + "." +
                                                  std::to_string(v.index()));

      bool in_partition = false;
      for (auto attr : attrs) {
        if (attr.getRootReference().str() == ATTR_WS_TMALOAD) {
          mlir::ModuleOp mod = forOp->getParentOfType<mlir::ModuleOp>();
          auto group = getGroupFromSymbolRefAttr(mod, attr);
          in_partition = true;
          break;
        }
      }
      if (in_partition) {
        oldLoopId2New[v.index()] = loopArgs.size();
        loopArgs.push_back(forOp.getInitArgs()[v.index()]);
      } else {
        oldLoopId2New[v.index()] = -1;
      }
    }
    // append normalized loop index to the loop args
    loopArgs.push_back(normIdxInit);

    auto newForOp = createNewForOp(forOp, builder, mapping, loopArgs);
    // within the loop body, the normalized loop index is the corresponding
    // loop variable
    auto normLoadIdxVar = newForOp.getRegionIterArgs().back();

    builder.setInsertionPointToStart(newForOp.getBody());
    for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs())) {
      int id = oldLoopId2New[arg.index()];
      if (id != -1) {
        mapping.map(arg.value(), newForOp.getRegionIterArgs()[id]);
      }
    }
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

    // get all the load op
    SmallVector<Value> loadOps, origLoadOps;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (isOpInGroup(&op, ATTR_WS_TMALOAD)) {
        Operation *newOp = builder.clone(op, mapping);
        if (isa<triton::DescriptorLoadOp, triton::LoadOp>(newOp)) {
          loadOps.push_back(newOp->getResult(0));
          origLoadOps.push_back(op.getResult(0));
        }
      }
    }
    bool isAllUsedInDot = loadOpsAllUsedInDot(forOp, origLoadOps);
    if (forOpIdx == 0) {
      // create aref objects for the first loop
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(arefCreatePoint);
      // check if the loaded tensors are used in the same dot op
      createArefCreate(builder, forOp.getLoc(), memDescTypes, arefOps,
                       AREF_SIZE, isAllUsedInDot);
    } else {
      // reuse the aref objects for the rest of the loops, but we need to
      // verify it meets the criteria for reuse
      assert(canReuseAref(builder.getContext(), memDescTypes, arefOps,
                          isAllUsedInDot, AREF_SIZE) &&
             "Aref objects can't be reused");
    }

    builder.setInsertionPointToStart(newForOp.getBody());
    createArefPut(builder, forOp, newForOp, origLoadOps, loadOps, arefOps,
                  AREF_SIZE, normLoadIdxVar, isAllUsedInDot);

    builder.setInsertionPointToEnd(newForOp.getBody());
    Value one = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 1, 32);
    auto nextNormLoadIdx =
        builder.create<arith::AddIOp>(forOp.getLoc(), normLoadIdxVar, one);
    SmallVector<Value> yieldValues;
    for (const auto &v :
         llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
      int id = oldLoopId2New[v.index()];
      if (id != -1) {
        yieldValues.push_back(mapping.lookupOrDefault(v.value()));
      }
    }
    yieldValues.push_back(nextNormLoadIdx);
    builder.create<scf::YieldOp>(forOp.getLoc(), yieldValues);
    return newForOp;
  }
}

Value getLoopNumIter(scf::ForOp forOp, OpBuilder &builder, IRMapping mapping) {
  auto lb = mapping.lookupOrDefault(forOp.getLowerBound());
  auto ub = mapping.lookupOrDefault(forOp.getUpperBound());
  auto step = mapping.lookupOrDefault(forOp.getStep());
  auto loc = forOp.getLoc();
  return gpu::getLoopNumIter(lb, ub, step, loc, builder);
}

scf::ForOp
createMMAGroup(scf::ForOp forOp, OpBuilderWithGroup builder, int forOpIdx,
               int numStages, int mmaDepth, int numWarps,
               SmallVector<triton::nvidia_gpu::ArefCreateOp> &arefSMEMOps,
               SmallVector<triton::nvidia_gpu::ArefCreateOp> &arefTMEMOps,
               IRMapping &mapping, Value normIdxInit, Value outerNormIdx,
               Operation *tmemAllocHoistPoint) {
  if (!isOpInGroup(forOp, ATTR_WS_MMA)) {
    return nullptr;
  }

  auto subI = [&](Value a, Value b) -> Value {
    return builder.create<arith::SubIOp>(forOp.getLoc(), a, b);
  };
  auto mulI = [&](Value a, Value b) -> Value {
    return builder.create<arith::MulIOp>(forOp.getLoc(), a, b);
  };
  auto divI = [&](Value a, Value b) -> Value {
    return builder.create<arith::DivSIOp>(forOp.getLoc(), a, b);
  };
  auto makeCstI32 = [&](int c) -> Value {
    return builder.create<arith::ConstantIntOp>(forOp.getLoc(), c, 32);
  };
  auto zero = makeCstI32(0);
  auto one = makeCstI32(1);

  auto innerMostLoop = true;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (isa<scf::ForOp>(op) && isOpInGroup(&op, ATTR_WS_MMA)) {
      innerMostLoop = false;
    }
  }

  if (!innerMostLoop) {
    SmallVector<Value> loopArgs{forOp.getInitArgs()};
    loopArgs.push_back(normIdxInit);

    auto newForOp = createNewForOp(forOp, builder, mapping, loopArgs);
    auto normMathIdxVar = newForOp.getRegionIterArgs().back();

    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
    for (unsigned i = 0; i < forOp.getRegionIterArgs().size(); ++i) {
      auto oldArg = forOp.getRegionIterArgs()[i];
      auto newArg = newForOp.getRegionIterArgs()[i];
      mapping.map(oldArg, newArg);
    }

    scf::ForOp newInnerFor;
    builder.setInsertionPointToStart(newForOp.getBody());

    for (auto &op : forOp.getBody()->without_terminator()) {
      if (!isOpInGroup(&op, ATTR_WS_MMA)) {
        continue;
      }
      if (auto innerFor = dyn_cast<scf::ForOp>(op)) {
        assert(!newInnerFor);
        auto newNormIdxInit =
            mulI(normMathIdxVar, getLoopNumIter(innerFor, builder, mapping));
        newInnerFor =
            createMMAGroup(innerFor, builder, forOpIdx, numStages, mmaDepth,
                           numWarps, arefSMEMOps, arefTMEMOps, mapping,
                           newNormIdxInit, normMathIdxVar, tmemAllocHoistPoint);
      } else if (isa<nvidia_gpu::TMEMAllocOp>(op)) {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPoint(tmemAllocHoistPoint);
        builder.clone(op, mapping);
      } else {
        builder.clone(op, mapping);
      }
    }
    assert(newInnerFor);
    auto nextNormMathIdx =
        builder.create<arith::AddIOp>(forOp.getLoc(), normMathIdxVar, one);
    SmallVector<Value> yieldValues;
    for (const auto &v : forOp.getBody()->getTerminator()->getOperands()) {
      yieldValues.push_back(mapping.lookupOrDefault(v));
    }
    yieldValues.push_back(nextNormMathIdx);
    builder.setInsertionPointToEnd(newForOp.getBody());
    builder.create<scf::YieldOp>(newForOp.getLoc(), yieldValues);
    return newForOp;
  } else {
    int AREF_SIZE = numStages;
    int MMA_STAGES = mmaDepth;
    int pendings = MMA_STAGES - 1;

    std::map<int, int> oldLoopId2New;
    SmallVector<Value> loopArgs;
    for (const auto &v :
         llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
      auto attrs =
          getGroupsAttr(forOp.getOperation(), std::string(ATTR_WSGROUPS) + "." +
                                                  std::to_string(v.index()));
      bool in_partition = false;
      for (auto attr : attrs) {
        if (attr.getRootReference().str() == ATTR_WS_MMA) {
          mlir::ModuleOp mod = forOp->getParentOfType<mlir::ModuleOp>();
          auto group = getGroupFromSymbolRefAttr(mod, attr);
          in_partition = true;
          break;
        }
      }
      if (in_partition) {
        oldLoopId2New[v.index()] = loopArgs.size();
        loopArgs.push_back(forOp.getInitArgs()[v.index()]);
      } else {
        oldLoopId2New[v.index()] = -1;
      }
    }
    loopArgs.push_back(normIdxInit);

    auto newForOp = createNewForOp(forOp, builder, mapping, loopArgs);
    auto normMathIdxVar = newForOp.getRegionIterArgs().back();
    // create tma_get operations at the beginning of the loop
    builder.setInsertionPointToStart(newForOp.getBody());

    // traverse through all original load op in the forop body
    SmallVector<Value> loadOps;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (isa<triton::DescriptorLoadOp, triton::LoadOp>(op))
        loadOps.push_back(op.getResult(0));
    }
    createArefGet(builder, forOp.getLoc(), newForOp, loadOps, arefSMEMOps,
                  mapping, normMathIdxVar, AREF_SIZE);

    int newIterArgId = 0;
    for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs())) {
      if (oldLoopId2New[arg.index()] != -1) {
        mapping.map(arg.value(), newForOp.getRegionIterArgs()[newIterArgId++]);
      }
    }
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isOpInGroup(&op, ATTR_WS_MMA)) {
        continue;
      }
      Operation *newOp = builder.clone(op, mapping);
    }

    builder.setInsertionPointToEnd(newForOp.getBody());
    auto newNormMathIdx =
        builder.create<arith::AddIOp>(forOp.getLoc(), normMathIdxVar, one);
    SmallVector<Value> yieldValues;
    for (const auto &v :
         llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
      if (oldLoopId2New[v.index()] != -1) {
        yieldValues.push_back(mapping.lookupOrDefault(v.value()));
      }
    }
    yieldValues.push_back(newNormMathIdx);
    builder.create<scf::YieldOp>(forOp.getLoc(), yieldValues);

    // update forOp results
    int idx = 0;
    for (const auto &v :
         llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
      if (oldLoopId2New[v.index()] != -1) {
        forOp.getResult(v.index()).replaceAllUsesWith(
            newForOp.getResult(idx++));
      }
    }

    {
      // XXX: we have the following can code in input
      //   %buf = local_alloc ..  : memdesc<.., /* mutable */ false>
      //   %tbuf = memdesc_trans %buf : memdesc<.., false> -> memdesc<.., false>
      // when we generaet aref_get.enter, we have
      //   %buf1 = aref_get.enter : memdesc<..., /* mutable */ true>
      // and when we update all uses of %buf with %buf1 we get
      //  %tbuf1 = memdesc_trans %buf1 : memdesc<.., true> -> memdesc<.., false>
      // and verifier complains on type mismatch
      // when it happens, we need to update the output type of memdesc_trans to
      //   %tbuf1 = memdesc_trans %buf1 : memdesc<.., true> -> memdesc<.., true>
      // there could be a better way to handle this, for now we just do this war
      SmallVector<Operation *> ops2erase;
      for (auto &op : newForOp.getBody()->without_terminator()) {
        if (!isa<triton::gpu::MemDescTransOp>(op))
          continue;
        if (!isa<triton::nvidia_gpu::ArefGetEnterOp>(
                op.getOperand(0).getDefiningOp()))
          continue;
        auto memDescTransOp = cast<MemDescTransOp>(op);
        auto memDescType =
            cast<MemDescType>(memDescTransOp->getResult(0).getType());
        assert(memDescType);
        auto memDescTypeNew = MemDescType::get(
            memDescType.getShape(), memDescType.getElementType(),
            memDescType.getEncoding(), memDescType.getMemorySpace(), true);
        builder.setInsertionPoint(&op);
        auto memDescTransOpNew = builder.create<triton::gpu::MemDescTransOp>(
            memDescTransOp->getLoc(), memDescTypeNew,
            memDescTransOp.getOperand(), memDescTransOp.getOrderAttr());
        memDescTransOp->getResult(0).replaceAllUsesWith(
            memDescTransOpNew->getResult(0));
        ops2erase.push_back(memDescTransOp);
      }
      for (auto op : ops2erase) {
        op->erase();
      }
    }

    ModuleOp mod = forOp->getParentOfType<ModuleOp>();
    // insert MMA pipeline
    Operation *lastWarpGroupDotWaitOp =
        updateWarpGroupDotOp(newForOp, builder, pendings, numWarps);

    // attach aref.consumed after WarpGroupDotWaitOp
    int cntDotOp = 0;
    for (Operation &op : llvm::reverse(newForOp.getBody()->getOperations())) {
      if (isa<triton::nvidia_gpu::WarpGroupDotWaitOp>(op)) {
        builder.setInsertionPointAfter(&op);
        /* Depending on the MMA pipeline depth, we need to skip the first few
           as they are not yet finished. Essentially we are doing

           for (int tile_idx = 0; tile_idx < num_tiles_per_sm; ++tile_idx) {
               for (int i = 0; i < num_k_tile; +i) {
                  if (i >= pending) {
                    int norm_idx = tile_idx * num_k_tile + i
                    int old_stage_idx = (norm_idx - pending) % num_stage
                    arrive_barrier mbars_tma_empty[old_stage_idx]
                  }
               }
             }
           }

           If the loops are fused, we compare the index tile_idx * num_k_tile
           against pending. The code is still correct.
        */
        auto pendingCst = makeCstI32(pendings);
        auto kIterId = builder.create<arith::DivSIOp>(
            forOp.getLoc(), newForOp.getInductionVar(), newForOp.getStep());
        auto cond = builder.create<arith::CmpIOp>(
            forOp.getLoc(), arith::CmpIPredicate::sge, kIterId, pendingCst);
        auto ifOp = builder.create<scf::IfOp>(forOp.getLoc(), cond,
                                              /*withElseRegion*/ false);
        builder.setInsertionPointToStart(ifOp.thenBlock());
        auto arefIdx = subI(normMathIdxVar, pendingCst);
        builder.create<triton::nvidia_gpu::ArefGetExitOp>(
            forOp.getLoc(), arefSMEMOps[0], arefIdx);
        if (cntDotOp > arefSMEMOps.size()) {
          break;
        }
        cntDotOp++;
      } else if (auto mmaOp =
                     dyn_cast<triton::nvidia_gpu::MMAv5OpInterface>(op)) {
        // For MMAv5, we extract the "current" stage aref and assign it to
        // the "barrier" parameter of the MMA op. Later, the "empty" barrier
        // associated with the aref is given to tcgen05.commit, which will
        // do an arrive on the empty barrier after MMA completes.
        // Thus, there is no need for a seperate ConsmedArefOp or an
        // ArriveBarrier op.
        builder.setInsertionPoint(&op);
        mmaOp.addCompletionBarrier(
            arefSMEMOps[arefSMEMOps.size() - 1 - cntDotOp], normMathIdxVar);
        if (cntDotOp > arefSMEMOps.size()) {
          break;
        }
        cntDotOp++;

        auto mod = mmaOp->getParentOfType<mlir::ModuleOp>();
        if (mod->hasAttr(ATTR_WS_EPILOGUE)) {
          OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPoint(tmemAllocHoistPoint);

          auto canDoubleBufferTMEM = [&](MemDescType tmemDesc) {
            if (!outerNormIdx) {
              // Non persistent or fused persistent case
              return false;
            }
            auto blockM = tmemDesc.getShape()[0];
            auto blockN = tmemDesc.getShape()[1];
            constexpr int numTMEMColumns = 512;
            constexpr int numTMEMRows = 128;
            if (blockM * blockN > numTMEMRows * numTMEMColumns / 2)
              return false;
            if (isa<nvidia_gpu::TCGen5MMAScaledOp>(mmaOp) && blockN == 256)
              return false;
            return true;
          };

          auto memDescType =
              mlir::cast<MemDescType>(mmaOp.getAccumulator().getType());
          SmallVector<MemDescType> memDescTypes{memDescType};
          const int numTMEMBuffer = canDoubleBufferTMEM(memDescType) ? 2 : 1;

          createArefCreate(builder, forOp.getLoc(), memDescTypes, arefTMEMOps,
                           numTMEMBuffer, true);

          auto arefOp = arefTMEMOps[0];
          auto tmemOperand = arefOp.getOperands()[0];
          auto arefTMEMIdx = numTMEMBuffer == 1 ? zero : outerNormIdx;

          // create ArefPut
          builder.setInsertionPoint(newForOp);
          auto arefPut = builder.create<triton::nvidia_gpu::ArefPutEnterOp>(
              forOp.getLoc(), memDescType, arefOp, arefTMEMIdx);
          mmaOp.setAccumulator(arefPut.getResult(0));
          builder.setInsertionPointAfter(newForOp);
          builder.create<triton::nvidia_gpu::ArefPutExitOp>(
              forOp.getLoc(), arefOp, arefTMEMIdx);
        }
      }
    }

    // after the math WG loop, postLoopNormIdx is the last loop result, which
    // can be used by the next math WG loop in the causal FMHA case and
    // math_wg_pipe is False.
    // Its typical value for persistent matmul is
    // * tile_idx * num_k_tile + num_k_tile for nested loops
    // * num_tiles_per_sm * num_k_tiles for a fused loop
    auto postLoopNormIdx = newForOp.getResults().back();

    // sync all mbarrier after the last WarpGroupDotWaitOp
    // if it is outside the loop
    /*
     * Note: This may introduce delay for the persistent kernel
     * e.g., K_TILE=4, AREF=3, MMA=2
     *
     * Expected: (^ means the start of a new K_TILE)
     * AREF     0 1 2 0 ^ 1 2 0 1 ^ ...
     * arrive   x 0 1 2   x 1 2 0
     *                0         1
     *
     * Actually implemented:
     * AREF     0 1 2 0 ^ 1 2 0 1 ^ 2 ...
     * arrive   x 0 1 2   0 1 2 0   1
     */
    if (lastWarpGroupDotWaitOp &&
        lastWarpGroupDotWaitOp->getPrevNode() == newForOp.getOperation()) {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointAfter(lastWarpGroupDotWaitOp);
      /* Generate a loop to release all previous-stage TMA empty barriers
         for (int i = min(num_k_tile, pendings); i > 0; --i) {
            aref_get.exit arefs[post_loop_norm_idx - i)]
         }

         Since MLIR does not support such loop, we instead generate
         for (int i = 0; i < min(num_k_tile, pendings); ++i) {
            old_iter_idx = min(num_k_tile, pendings) - i
            aref_get.exit arefs[post_loop_norm_idx - old_iter_idx]
         }
      */
      auto numGetExit = builder.create<arith::MinSIOp>(
          forOp.getLoc(), getLoopNumIter(newForOp, builder),
          makeCstI32(pendings));
      auto getExitLoop = builder.create<scf::ForOp>(
          forOp.getLoc(), zero, numGetExit, one, SmallVector<Value>{});
      builder.setInsertionPointToStart(getExitLoop.getBody());
      auto oldArefIdx = subI(postLoopNormIdx,
                             subI(numGetExit, getExitLoop.getInductionVar()));
      // lastWarpGroupDotWaitOp is set only for single GEMM case and when we
      // apply wgmma pipelining
      assert(arefSMEMOps.size() == 1);
      builder.create<triton::nvidia_gpu::ArefGetExitOp>(
          forOp.getLoc(), arefSMEMOps[0], oldArefIdx);
    }

    return newForOp;
  }
}

void cloneEpilogueOps(SmallVector<Operation *> ops,
                      triton::nvidia_gpu::ArefCreateOp arefOp, Value arefIdx,
                      Location loc, OpBuilderWithGroup &builder,
                      IRMapping &mapping) {
  for (auto op : ops) {
    if (auto tmemLoad = dyn_cast<nvidia_gpu::TMEMLoadOp>(op)) {
      auto memDescType =
          mlir::cast<MemDescType>(arefOp.getOperands()[0].getType());
      assert(memDescType);
      Type type = getDataMemDescType(memDescType);
      auto arefGet = builder.create<triton::nvidia_gpu::ArefGetEnterOp>(
          loc, SmallVector<Type>{type}, arefOp, arefIdx);
      tmemLoad.getSrc().replaceAllUsesWith(arefGet.getResult(0));
    }

    builder.clone(*op, mapping);

    if (auto tmemLoad = dyn_cast<nvidia_gpu::TMEMLoadOp>(op)) {
      // Release TMEM empty immediately after tmem load
      builder.create<triton::nvidia_gpu::ArefGetExitOp>(loc, arefOp, arefIdx);
    }
  }
}

scf::ForOp
createEpilogueGroup(scf::ForOp forOp, OpBuilderWithGroup builder,
                    SmallVector<triton::nvidia_gpu::ArefCreateOp> &arefOps,
                    IRMapping &mapping, Value normIdxInit) {
  if (!isOpInGroup(forOp, ATTR_WS_EPILOGUE)) {
    return nullptr;
  }

  auto innerMostLoop = true;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (isa<scf::ForOp>(op) && isOpInGroup(&op, ATTR_WS_EPILOGUE)) {
      innerMostLoop = false;
    }
  }

  if (!innerMostLoop) {
    SmallVector<Value> loopArgs{forOp.getInitArgs()};
    loopArgs.push_back(normIdxInit);

    auto newForOp = createNewForOp(forOp, builder, mapping, loopArgs);
    auto normLoadIdxVar = newForOp.getRegionIterArgs().back();

    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
    for (unsigned i = 0; i < forOp.getRegionIterArgs().size(); ++i) {
      auto oldArg = forOp.getRegionIterArgs()[i];
      auto newArg = newForOp.getRegionIterArgs()[i];
      mapping.map(oldArg, newArg);
    }

    scf::ForOp newInnerFor;
    builder.setInsertionPointToStart(newForOp.getBody());

    for (auto &op : forOp.getBody()->without_terminator()) {
      if (!isOpInGroup(&op, ATTR_WS_EPILOGUE)) {
        continue;
      }
      if (auto innerFor = dyn_cast<scf::ForOp>(op)) {
        assert(!newInnerFor);
        newInnerFor = createEpilogueGroup(innerFor, builder, arefOps, mapping,
                                          normLoadIdxVar);
      } else {
        builder.clone(op, mapping);
      }
    }
    assert(newInnerFor);
    auto nextNormLoadIdx = newInnerFor.getResults().back();

    SmallVector<Value> yieldValues;
    for (const auto &v : forOp.getBody()->getTerminator()->getOperands()) {
      yieldValues.push_back(mapping.lookupOrDefault(v));
    }
    yieldValues.push_back(nextNormLoadIdx);
    builder.setInsertionPointToEnd(newForOp.getBody());
    builder.create<scf::YieldOp>(newForOp.getLoc(), yieldValues);
    return newForOp;
  } else {
    std::map<int, int> oldLoopId2New;
    SmallVector<Value> loopArgs;
    for (const auto &v :
         llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
      auto attrs =
          getGroupsAttr(forOp.getOperation(), std::string(ATTR_WSGROUPS) + "." +
                                                  std::to_string(v.index()));

      bool in_partition = false;
      for (auto attr : attrs) {
        if (attr.getRootReference().str() == ATTR_WS_TMALOAD) {
          mlir::ModuleOp mod = forOp->getParentOfType<mlir::ModuleOp>();
          auto group = getGroupFromSymbolRefAttr(mod, attr);
          in_partition = true;
          break;
        }
      }
      if (in_partition) {
        oldLoopId2New[v.index()] = loopArgs.size();
        loopArgs.push_back(forOp.getInitArgs()[v.index()]);
      } else {
        oldLoopId2New[v.index()] = -1;
      }
    }
    // append normalized loop index to the loop args
    loopArgs.push_back(normIdxInit);

    auto newForOp = createNewForOp(forOp, builder, mapping, loopArgs);
    // within the loop body, the normalized loop index is the corresponding
    // loop variable
    auto normLoadIdxVar = newForOp.getRegionIterArgs().back();

    builder.setInsertionPointToStart(newForOp.getBody());
    for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs())) {
      int id = oldLoopId2New[arg.index()];
      if (id != -1) {
        mapping.map(arg.value(), newForOp.getRegionIterArgs()[id]);
      }
    }
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

    builder.setInsertionPointToEnd(newForOp.getBody());

    SmallVector<Operation *> epiOps;
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (!isOpInGroup(&op, ATTR_WS_EPILOGUE))
        continue;
      epiOps.push_back(&op);
    }

    cloneEpilogueOps(epiOps, arefOps[0], normLoadIdxVar, forOp.getLoc(),
                     builder, mapping);

    Value one = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 1, 32);
    auto nextNormLoadIdx =
        builder.create<arith::AddIOp>(forOp.getLoc(), normLoadIdxVar, one);
    SmallVector<Value> yieldValues;
    for (const auto &v :
         llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
      int id = oldLoopId2New[v.index()];
      if (id != -1) {
        yieldValues.push_back(mapping.lookupOrDefault(v.value()));
      }
    }
    yieldValues.push_back(nextNormLoadIdx);
    builder.create<scf::YieldOp>(forOp.getLoc(), yieldValues);
    return newForOp;
  }
}

bool eligibleCausalLoops(scf::ForOp &forOp1, scf::ForOp &forOp2) {
  // check if the two loops are causal
  // we currently only handle the following case:
  // 1. the lower bound of the first loop is 0
  // 2. the lower bound of the second loop is the upper bound of the first
  // loop
  Value lower1 = forOp1.getLowerBound();
  Value upper1 = forOp1.getUpperBound();
  Value lower2 = forOp2.getLowerBound();
  // in IR, upper1 and lower2 are different Values, so we can't simply check
  // upper1 == lower2. We need to do a recursive check if they have the same
  // expression.
  Operation *defUpper1 = upper1.getDefiningOp();
  Operation *defLower2 = lower2.getDefiningOp();
  if ((defUpper1->getName().getStringRef().str() !=
       defLower2->getName().getStringRef().str()) ||
      (defUpper1->getNumOperands() != defLower2->getNumOperands())) {
    std::cout << "not equivalent when comparing operands\n";
    return false;
  }
  for (auto opnds :
       llvm::zip(defUpper1->getOperands(), defLower2->getOperands())) {
    if (std::get<0>(opnds) != std::get<1>(opnds)) {
      std::cout << "not equivalent when comparing operands\n";
      return false;
    }
  }
  if (!isa<arith::ConstantIntOp>(lower1.getDefiningOp())) {
    return false;
  }
  if (auto constantOp = lower1.getDefiningOp<mlir::arith::ConstantIntOp>()) {
    if (constantOp.value() != 0) {
      std::cout << "not constant 0\n";
      return false;
    }
  }
  return true;
}

void moveOpBetweenLoops(scf::ForOp &forOp1, scf::ForOp &forOp2) {
  // iterate all the operations between forOp1 and forOp2
  // and move them before forOp1
  Block::iterator start = ++Block::iterator(forOp1);
  Block::iterator end = Block::iterator(forOp2);
  // If there are no operations between the two loops, return
  if (start == end)
    return;
  for (auto it = start; it != end; /* no increment here */) {
    Operation *op = &(*it++);
    op->moveBefore(forOp1);
  }
}

ttng::WarpGroupReturnOp createWgOp(ModuleOp mod, int barId, std::string group,
                                   Location loc, OpBuilderWithGroup &builder) {
  auto wsGroup = getGroupFromSymbolRefAttr(
      mod, mlir::SymbolRefAttr::get(builder.getContext(), group));
  auto wgOp = builder.create<ttng::WarpGroupOp>(loc, wsGroup.startWarp,
                                                wsGroup.numWarps, 1);

  // set wgOp barId attribute
  wgOp->setAttr(ATTR_WS_BARID, builder.getI32IntegerAttr(barId));
  auto &wgRegion = wgOp.getPartitionRegions()[0];
  Block *wgBlock = &wgRegion.emplaceBlock();
  OpBuilder wgBuilder = OpBuilder::atBlockEnd(wgBlock);
  auto wgRetOp = wgBuilder.create<mlir::triton::nvidia_gpu::WarpGroupReturnOp>(
      wgOp.getLoc());
  return wgRetOp;
}

bool shouldReallocRegisters(ModuleOp mod) {
  auto target = mod->getAttrOfType<StringAttr>(AttrTargetName);
  // TODO: Use setmaxregisterop on Blackwell?
  // https://jirasw.nvidia.com/browse/OT-116
  return target == "cuda:90";
}

} // namespace

class TritonGPUSplitWarpGroupLoopsPass
    : public impl::TritonGPUSplitWarpGroupLoopsBase<
          TritonGPUSplitWarpGroupLoopsPass> {

  using impl::TritonGPUSplitWarpGroupLoopsBase<
      TritonGPUSplitWarpGroupLoopsPass>::TritonGPUSplitWarpGroupLoopsBase;

public:
  void runOnOperation() override {
    ModuleOp m = getOperation();
    int numWarps = maybeLookupNumWarps(m).value();

    LLVM_DEBUG({
      DBGS() << "numStages=" << numStages << ", mmaDepth=" << mmaDepth
             << ", warps=" << numWarps << "\n";
      DBGS() << "Module before splitting warp group loops:\n";
      m.dump();
    });

    for (auto func : m.getOps<triton::FuncOp>()) {
      {
        // after rebase:
        //   %b = tmem_alloc %a
        //   ...
        // was replaced by
        //   %b = tmem_alloc
        //   tmem_store %a, %b
        //   ..
        // For now, as a work-around we replace first tmem_store as we handled
        // it differently in this function
        SmallVector<nvidia_gpu::TMEMAllocOp> tmemAllocOps;
        func.walk(
            [&](nvidia_gpu::TMEMAllocOp op) { tmemAllocOps.push_back(op); });
        SmallVector<ttng::TMEMStoreOp> staleTMEMStoreOps;
        for (auto allocOp : tmemAllocOps) {
          if (auto storeOp =
                  dyn_cast<ttng::TMEMStoreOp>(*allocOp->getNextNode())) {
            if (storeOp.getDst() == allocOp.getResult()) {
              staleTMEMStoreOps.push_back(storeOp);
            }
          }
        }
        for (auto storeOp : staleTMEMStoreOps) {
          storeOp->erase();
        }
      }
      SmallVector<scf::ForOp> forOps;
      auto &body = func.getBody().front();
      for (const auto &op : body.getOperations()) {
        if (auto forOp = dyn_cast<scf::ForOp>(op)) {
          forOps.push_back(forOp);
        }
      }

      // [FMHA]: Which group is assigned to the code in between the loops?
      if (forOps.size() == 2) {
        // NOTE: only works for two consecutive loops
        if (eligibleCausalLoops(forOps[0], forOps[1])) {
          moveOpBetweenLoops(forOps[0], forOps[1]);
        } else {
          assert(false &&
                 "can only handle causal FMHA for multiple loops case");
        }
      } else {
        assert(forOps.size() == 1);
      }

      OpBuilder builder(func);
      auto loc = func.getLoc();
      // insert the ifOp before the first forOp
      builder.setInsertionPoint(forOps[0]);
      auto barrier = builder.create<NVVM::Barrier0Op>(loc);
      barrier->setAttr(ATTR_WS_INIT_BARRIER_SYNC, builder.getUnitAttr());
      Operation *nextInsertPointAfter = barrier;
      Operation *arefCreatePoint = barrier;
      Operation *tmemAllocHoistPoint = barrier;

      SmallVector<triton::nvidia_gpu::ArefCreateOp> arefSMEMOps;
      SmallVector<triton::nvidia_gpu::ArefCreateOp> arefTMEMOps;
      // TODO: Do not hard code currently supported group IDs
      // However, ATTR_WS_TMALOAD must be processed first since the arefOps
      // is populated during createLoadGroup (by createArefPut).
      IRMapping mapping;
      int barId = 1; // warp-group-wide barriers start with barId = 1
                     // barId = 0 is reserved for the CTA-wide barrier
      for (auto group : {ATTR_WS_TMALOAD, ATTR_WS_MMA, ATTR_WS_EPILOGUE}) {
        if (!m->hasAttr(group)) {
          continue;
        }
        OpBuilderWithGroup builder(forOps[0], group);
        builder.setInsertionPointAfter(nextInsertPointAfter);
        auto wgRetOp = createWgOp(m, barId++, group, loc, builder);
        builder.setInsertionPoint(wgRetOp);

        Value normIdxInit = builder.create<arith::ConstantIntOp>(loc, 0, 32);

        if (group == ATTR_WS_TMALOAD) {
          if (shouldReallocRegisters(m)) {
            builder.create<NVVM::SetMaxRegisterOp>(
                loc, 40, NVVM::SetMaxRegisterAction::decrease);
          }

          if (forOps.size() == 2) {
            // causal attention
            auto firstLoop =
                createLoadGroup(forOps[0], builder, 0, numStages, arefSMEMOps,
                                mapping, arefCreatePoint, normIdxInit);
            auto nextNormIdxInit = firstLoop.getResults().back();
            builder.setInsertionPointAfter(firstLoop);
            createLoadGroup(forOps[1], builder, 1, numStages, arefSMEMOps,
                            mapping, arefCreatePoint, nextNormIdxInit);
          } else {
            createLoadGroup(forOps[0], builder, 0, numStages, arefSMEMOps,
                            mapping, arefCreatePoint, normIdxInit);
          }
        } else if (group == ATTR_WS_MMA) {
          if (shouldReallocRegisters(m)) {
            builder.create<NVVM::SetMaxRegisterOp>(
                loc, 232, NVVM::SetMaxRegisterAction::increase);
          }
          if (forOps.size() == 2) {
            // causal attention
            auto firstLoop =
                createMMAGroup(forOps[0], builder, 0, numStages, mmaDepth,
                               numWarps, arefSMEMOps, arefTMEMOps, mapping,
                               normIdxInit, Value(), tmemAllocHoistPoint);
            auto nextNormIdxInit = firstLoop.getResults().back();
            builder.setInsertionPointAfter(firstLoop);
            createMMAGroup(forOps[1], builder, 1, numStages, mmaDepth, numWarps,
                           arefSMEMOps, arefTMEMOps, mapping, nextNormIdxInit,
                           Value(), tmemAllocHoistPoint);
          } else {
            createMMAGroup(forOps[0], builder, 0, numStages, mmaDepth, numWarps,
                           arefSMEMOps, arefTMEMOps, mapping, normIdxInit,
                           Value(), tmemAllocHoistPoint);
          }
          // For attention and non-persistent matmul cases, put operations after
          // the MMA loop into the same aref-if block that the MMA loop is in
          for (mlir::Operation &op : llvm::make_early_inc_range(
                   llvm::make_range(std::next(forOps.back()->getIterator()),
                                    body.getTerminator()->getIterator()))) {
            if (!isa<scf::ForOp>(op) && isOpInGroup(&op, ATTR_WS_MMA)) {
              op.moveBefore(wgRetOp);
            }
          }
        } else if (group == ATTR_WS_EPILOGUE) {
          if (isOpInGroup(forOps[0], ATTR_WS_EPILOGUE)) {
            createEpilogueGroup(forOps[0], builder, arefTMEMOps, mapping,
                                normIdxInit);
          } else {
            SmallVector<Operation *> epiOps;
            for (mlir::Operation &op : llvm::make_early_inc_range(
                     llvm::make_range(std::next(forOps.back()->getIterator()),
                                      body.getTerminator()->getIterator()))) {
              if (isOpInGroup(&op, ATTR_WS_EPILOGUE)) {
                epiOps.push_back(&op);
              }
            }
            assert(epiOps.size() > 0);
            auto loc = forOps[0].getLoc();
            auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
            builder.setInsertionPoint(wgRetOp);
            cloneEpilogueOps(epiOps, arefTMEMOps[0], zero, loc, builder,
                             mapping);

            for (int i = epiOps.size() - 1; i >= 0; --i) {
              epiOps[i]->erase();
            }
          }
        }

        nextInsertPointAfter = wgRetOp.getParentOp();
      }

      // The second loop in causal attention depends on the first, so erase in
      // the reverse order
      for (int i = forOps.size() - 1; i >= 0; --i) {
        forOps[i].erase();
      }

      // If the epilogue group is created, the original TMEM alloc for the
      // accumulator is no longer used. Manually remove it since DCE does not
      // remove mutable TMEM alloc
      llvm::SmallSetVector<nvidia_gpu::TMEMAllocOp, 5> staleTMEMAlloc;
      m.walk([&](nvidia_gpu::TMEMAllocOp tmemAlloc) {
        if (tmemAlloc.getResult().user_end() ==
            tmemAlloc.getResult().user_begin())
          staleTMEMAlloc.insert(tmemAlloc);
      });
      for (auto alloc : staleTMEMAlloc) {
        alloc.erase();
      }
    }

    LLVM_DEBUG({
      DBGS() << "Module after splitting warp group loops:\n";
      m.dump();
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
