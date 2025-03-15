#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonamdgpu-in-thread-transpose"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttag = mlir::triton::amdgpu;

namespace {

static Type getNewType(Type type, Attribute encoding) {
  RankedTensorType tensorType = dyn_cast<RankedTensorType>(type);
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}

void convertLayout(Attribute encoding, Operation *op) {
  OpBuilder builder(op);
  // Convert operands
  // For load/store with tensor pointers, we don't have to change the
  // operands' type, we do this by changing the outputs' type of
  // `make_tensor_ptr`
  SmallVector<Value, 4> newArgs;
  for (auto operand : op->getOperands()) {
    auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
    if (tensorType && !isa<triton::gpu::SwizzledSharedEncodingAttr>(
                          tensorType.getEncoding())) {
      Type newType = getNewType(tensorType, encoding);
      newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), newType, operand));
    } else {
      newArgs.push_back(operand);
    }
  }

  // Convert output types
  SmallVector<Type, 4> newTypes;
  for (auto t : op->getResultTypes()) {
    bool isAsync = isa<triton::gpu::AsyncCopyGlobalToLocalOp>(op);
    newTypes.push_back(isAsync ? t : getNewType(t, encoding));
  }

  // Construct new op with the new encoding
  Operation *newOp = builder.create(op->getLoc(), op->getName().getIdentifier(),
                                    newArgs, newTypes, op->getAttrs());

  // Cast the results back to the original layout
  for (size_t i = 0; i < op->getNumResults(); i++) {
    Value newResult = newOp->getResult(i);
    if (newTypes[i] != op->getResultTypes()[i]) {
      newResult = builder.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), op->getResult(i).getType(), newResult);
    }
    op->getResult(i).replaceAllUsesWith(newResult);
  }
  op->erase();
}

void transposeInRegsitersBeforeStoreInLocalMemory(
    Operation *memStoreOp, ArrayRef<int64_t> loadShape,
    ttg::BlockedEncodingAttr newLoadEncoding) {
  if (memStoreOp->getNumOperands() == 0)
    return;
  auto data = memStoreOp->getOperand(0);
  OpBuilder builder(memStoreOp);

  // local alloc has optional src
  // if it is not provided, nothing to do
  if (!data)
    return;

  auto transposedLayout =
      ttag::InThreadTransposeOp::deduceOutputLayout(loadShape, newLoadEncoding);
  auto transposedEncoding = triton::gpu::LinearEncodingAttr::get(
      memStoreOp->getContext(), transposedLayout);

  auto loc = memStoreOp->getLoc();
  auto preEncodedType = getNewType(data.getType(), newLoadEncoding);
  auto preEncoded =
      builder.create<ttg::ConvertLayoutOp>(loc, preEncodedType, data);

  auto transposedType = getNewType(data.getType(), transposedEncoding);
  auto inThreadTransposed = builder.create<ttag::InThreadTransposeOp>(
      loc, transposedType, preEncoded);
  memStoreOp->setOperand(0, inThreadTransposed);
}

Attribute createNewSharedEncoding(RankedTensorType operandType) {
  auto ctx = operandType.getContext();
  auto dotOperandEnc =
      cast<ttg::DotOperandEncodingAttr>(operandType.getEncoding());
  auto ctaLayout = ttg::getCTALayout(dotOperandEnc);
  auto bitWidth = operandType.getElementTypeBitWidth();
  SmallVector<unsigned> order{1, 0};
  if (dotOperandEnc.getOpIdx() == 1)
    std::swap(order[0], order[1]);

  auto tempAttr = ttg::SwizzledSharedEncodingAttr::get(
      ctx, dotOperandEnc, operandType.getShape(), order, ctaLayout, bitWidth,
      /*needTrans=*/false);

  auto sharedVec = tempAttr.getVec();
  auto perPhase = tempAttr.getPerPhase();
  auto maxPhase = tempAttr.getMaxPhase();

  auto newSharedEnc = ttg::AMDRotatingSharedEncodingAttr::get(
      ctx, sharedVec, perPhase, maxPhase, order, ctaLayout);

  return newSharedEnc;
}

void changeSharedEncoding(Value memVal, Attribute newEncoding) {
  auto originalType = cast<ttg::MemDescType>(memVal.getType());
  auto sharedEnc =
      dyn_cast<ttg::SwizzledSharedEncodingAttr>(originalType.getEncoding());
  // Already transformed this value
  if (!sharedEnc)
    return;

  auto newType = ttg::MemDescType::get(
      originalType.getShape(), originalType.getElementType(), newEncoding,
      originalType.getMemorySpace(), originalType.getMutableMemory());

  memVal.setType(newType);
}

/// Structure describes operations involved in local_alloc->local_load pattern
struct loadStoreLoadPatternComponents {
  SmallVector<tt::LoadOp> globalLoads;
  // list of localAllocOp and localStoreOp operations
  SmallVector<Operation *> localMemStores;
  // list of MemDescSubviewOp, control flow results and block operands
  SmallVector<Value> sharedMemVals;
};

/// For a given value return all operations that define it.
///
/// If val is a result of operation, return definingOp.
/// If val is a result of some control flow operation or block argument,
/// traverse control flow instructions.
FailureOr<SmallVector<Value>> traverseCFForValueDefs(Value val) {
  SmallVector<Value> totalDefs{val};
  LDBG("    traverseCFForValueDefs processing " << val);
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    Block *block = blockArg.getOwner();

    // Get parent operation (e.g., scf.for, scf.if, scf.while)
    Operation *parentOp = block->getParentOp();
    if (!parentOp) {
      LDBG("    block without parent op, can not analyze further");
      return failure();
    }

    // If block belongs to a function, stop tracking (function arguments)
    if (isa<triton::FuncOp>(parentOp)) {
      LDBG("    can not traverse def-use chains, found function argument");
      return failure();
    }

    int argIdx = blockArg.getArgNumber();

    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      // Handle `scf.for`
      // Continue traversing this op, even if it is visited
      // It could be visited with different arg idx
      int iterArgIdx = argIdx - 1; // Skip induction variable
      if (iterArgIdx >= 0) {
        Value yieldVal =
            forOp.getBody()->getTerminator()->getOperand(iterArgIdx);
        // look inside of a loop
        auto inLoop = traverseCFForValueDefs(yieldVal);
        // look outside of a loop
        int forOpArgIdx = iterArgIdx + forOp.getNumControlOperands();
        auto outLoop = traverseCFForValueDefs(forOp.getOperand(forOpArgIdx));
        if (failed(inLoop) || failed(outLoop))
          return failure();

        totalDefs.append(inLoop.value());
        totalDefs.append(outLoop.value());
      } else {
        // Induction variable
        auto search = traverseCFForValueDefs(forOp.getOperand(0));
        if (failed(search))
          return failure();
        totalDefs.append(search.value());
      }
    } else if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
      // Handle `scf.if`
      auto thenYield = ifOp.thenYield();
      auto elseYield = ifOp.elseYield();

      // Track all possible yielded values from then/else blocks
      if (thenYield) {
        auto ops = traverseCFForValueDefs(thenYield->getOperand(argIdx));
        if (failed(ops))
          return failure();
        totalDefs.append(ops.value());
      }
      if (elseYield) {
        auto ops = traverseCFForValueDefs(elseYield->getOperand(argIdx));
        if (failed(ops))
          return failure();
        totalDefs.append(ops.value());
      }
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
      // Handle `scf.while`
      auto terminator = whileOp.getBefore().front().getTerminator();
      auto search = traverseCFForValueDefs(terminator->getOperand(argIdx));
      if (failed(search))
        return failure();
      totalDefs.append(search.value());
    } else if (isa<RegionBranchOpInterface>(parentOp)) {
      // Deal with the case that convert_layout intakes from scf.if, etc.
      llvm::SmallVector<scf::YieldOp> yieldOps;
      parentOp->walk([&](Operation *op) {
        if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
          yieldOps.push_back(yieldOp);
        }
      });

      for (auto yieldOp : yieldOps) {
        auto ops = traverseCFForValueDefs(yieldOp->getOperand(argIdx));
        if (failed(ops))
          return failure();
        totalDefs.append(ops.value());
      }
    } else {
      assert(false && "unexpected control flow operation");
    }
  }
  return totalDefs;
}

struct ForwardSearchAnalysis {
  SmallVector<Operation *> ops;
  SmallVector<Value> transitiveCF;
};

/// For a given value return all operations that uses it.
///
/// Traverses control flow instructions forward.
FailureOr<ForwardSearchAnalysis> traverseCFForValueUses(Value val) {
  LDBG("    traverseCFForValueUses processing " << val);
  ForwardSearchAnalysis result;
  for (auto &use : val.getUses()) {
    auto user = use.getOwner();
    LDBG("      processing user " << *user);
    if (isa<triton::ReturnOp>(user)) {
      LDBG("    Reached return from function");
      return failure();
    } else if (isa<scf::YieldOp>(user)) {
      auto opIdx = use.getOperandNumber();
      auto parent = user->getParentOp();
      // traverse outside data flow
      auto parentResult = parent->getResult(opIdx);
      auto cfSearch = traverseCFForValueUses(parentResult);
      if (failed(cfSearch))
        return failure();
      result.ops.append(cfSearch.value().ops);
      result.transitiveCF.push_back(parentResult);
      result.transitiveCF.append(cfSearch.value().transitiveCF);

      // traverse loop internal data flow
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        int forBodyOperandIdx = opIdx + forOp.getNumInductionVars();
        auto blockArg = forOp.getBody()->getArgument(forBodyOperandIdx);
        auto cfSearch = traverseCFForValueUses(blockArg);
        if (failed(cfSearch))
          return failure();
        result.ops.append(cfSearch.value().ops);
        result.transitiveCF.push_back(blockArg);
        result.transitiveCF.append(cfSearch.value().transitiveCF);
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      LDBG("      for op num operands: " << forOp.getNumOperands());
      LDBG("      for op body num operands: "
           << forOp.getBody()->getNumArguments());
      assert(use.getOperandNumber() >= forOp.getNumControlOperands());
      int blockArgIdx = use.getOperandNumber() - forOp.getNumControlOperands() +
                        forOp.getNumInductionVars();
      auto blockArg = forOp.getBody()->getArgument(blockArgIdx);
      auto cfSearch = traverseCFForValueUses(blockArg);
      if (failed(cfSearch))
        return failure();
      result.ops.append(cfSearch.value().ops);
      result.transitiveCF.push_back(blockArg);
      result.transitiveCF.append(cfSearch.value().transitiveCF);
    } else {
      result.ops.push_back(user);
    }
  }
  return result;
}

/// Look for defining operation, hopping over control flow.
///
/// Gather all operations of type T within one def-use hop from val,
/// control flow constructions are not considered as an operations.
/// \returns true on success, false if analysis failed
template <typename Op>
FailureOr<SmallVector<Op>> findAllDefiningOps(Value val) {
  auto candidates = traverseCFForValueDefs(val);
  if (failed(candidates))
    return failure();
  SmallVector<Op> result;
  for (auto candidateValue : candidates.value()) {
    auto op = candidateValue.getDefiningOp();
    if (!op)
      continue;
    if (auto typedOp = dyn_cast<Op>(op))
      result.push_back(typedOp);
  }
  return result;
}

/// Look for all operations with one of OpTy types in def-use chains in both
/// forward and backward directions.
///
/// Traversal goes through control flow operations and and stops at non OpTy
/// operation. For example: findAllDefUseOps<local_load, mem_subview,
/// local_store>(dot_operand)
///
///                                                    ----------------->
///                                                    local_store | traversed
/// global_load -> local_store -> mem_subview -> local_load -> dot
///                 traversed      traversed      traversed
///
/// \returns true on success, false if analysis failed
FailureOr<loadStoreLoadPatternComponents>
findReachableSMemOps(ttg::LocalLoadOp root,
                     SetVector<mlir::Operation *> &visited) {
  // breadth-first search for reachable opeations
  loadStoreLoadPatternComponents foundNetwork;
  SmallVector<Operation *> traversalStep{root};
  while (!traversalStep.empty()) {
    LDBG("begin new step in smem op analysis");
    SmallVector<Operation *> nextTraversalStep;
    for (auto candidate : traversalStep) {
      if (visited.contains(candidate))
        continue;
      visited.insert(candidate);
      LDBG("  processing in smem op analysis: " << *candidate);

      int forwardIdx = -1;
      int backwardIdx = -1;
      if (isa<ttg::LocalAllocOp>(candidate)) {
        foundNetwork.localMemStores.push_back(candidate);
        forwardIdx = 0;
      } else if (isa<ttg::LocalStoreOp>(candidate)) {
        foundNetwork.localMemStores.push_back(candidate);
        backwardIdx = 1;
      } else if (isa<ttg::MemDescSubviewOp>(candidate)) {
        forwardIdx = 0;
        backwardIdx = 0;
      } else if (isa<ttg::LocalLoadOp, ttg::LocalDeallocOp>(candidate)) {
        backwardIdx = 0;
      } else {
        // this operation is not part of shared memory def-use network,
        // algorithm should not reach this point
        assert(false);
        continue;
      }

      // Look backward
      if (backwardIdx != -1) {
        auto backwardSearch =
            traverseCFForValueDefs(candidate->getOperand(backwardIdx));
        if (failed(backwardSearch))
          return failure();
        for (auto def : backwardSearch.value()) {
          foundNetwork.sharedMemVals.push_back(def);
          if (Operation *op = def.getDefiningOp()) {
            // additional check, to ignore control flow operations
            if (isa<ttg::MemDescSubviewOp, ttg::LocalAllocOp>(op))
              nextTraversalStep.push_back(op);
          }
        }
      }

      // Look forward
      if (forwardIdx != -1) {
        auto forwardSearch =
            traverseCFForValueUses(candidate->getResult(forwardIdx));
        if (failed(forwardSearch))
          return failure();
        foundNetwork.sharedMemVals.append(forwardSearch.value().transitiveCF);
        nextTraversalStep.append(forwardSearch.value().ops);
      }
    }
    traversalStep = std::move(nextTraversalStep);
  }
  return foundNetwork;
}

unsigned getDimRepeats(RankedTensorType type, int dimIdx) {
  auto loadEnc = type.getEncoding();
  auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(loadEnc);
  if (!blockedEnc)
    return 0;
  auto sizePerThread = blockedEnc.getSizePerThread();
  auto lanes = blockedEnc.getThreadsPerWarp();
  auto warps = blockedEnc.getWarpsPerCTA();
  auto shape = type.getShape();
  int repeats =
      shape[dimIdx] / (sizePerThread[dimIdx] * lanes[dimIdx] * warps[dimIdx]);
  return std::max(1, repeats);
}

llvm::FailureOr<loadStoreLoadPatternComponents>
matchThreadRakePattern(Value operand) {
  auto opTensorTy = cast<RankedTensorType>(operand.getType());
  auto opEnc = opTensorTy.getEncoding();
  auto opDotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(opEnc);
  if (!opDotOpEnc)
    return failure();

  int kDimNum = opDotOpEnc.getOpIdx() == 0 ? 1 : 0;
  // TODO: support wmma
  if (!isa<ttg::AMDMfmaEncodingAttr, ttg::AMDWmmaEncodingAttr>(
          opDotOpEnc.getParent())) {
    LDBG("Operand's parent encoding is not MFMA");
    return failure();
  }

  // Find nearest local_load
  auto localLoadSearch = findAllDefiningOps<ttg::LocalLoadOp>(operand);
  if (failed(localLoadSearch)) {
    LDBG("Failed to traverse local loads");
    return failure();
  }

  if (localLoadSearch.value().size() == 0) {
    LDBG("Did not find local load operation");
    return failure();
  }

  loadStoreLoadPatternComponents pattern;

  SetVector<Operation *> visited;
  for (auto lLoad : localLoadSearch.value()) {
    // find local_alloc, local_store, local_load and ttg.memdesc_subview
    // operations
    auto sharedMemSearch = findReachableSMemOps(lLoad, visited);
    if (failed(sharedMemSearch)) {
      LDBG("Failed to traverse shared memmory operation network");
      return failure();
    }
    pattern = sharedMemSearch.value();
  }

  if (pattern.localMemStores.empty()) {
    LDBG("Did not find local alloc or store operations");
    return failure();
  }

  for (auto localMemStore : pattern.localMemStores) {
    LDBG("processing local mem store operation: " << *localMemStore);
    // check if it is a local alloc with no predecessor
    if (localMemStore->getNumOperands() == 0)
      continue;
    Value loadCandidate = localMemStore->getOperand(0);
    auto loadedEnc =
        cast<RankedTensorType>(loadCandidate.getType()).getEncoding();
    auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(loadedEnc);
    if (!blockedEnc)
      return failure();
    auto order = blockedEnc.getOrder();
    if (order[0] == kDimNum) {
      return failure();
    }
    auto globalLoadSearch = findAllDefiningOps<triton::LoadOp>(loadCandidate);
    if (failed(globalLoadSearch)) {
      LDBG("Failed to traverse path to global loads");
      return failure();
    }
    pattern.globalLoads.append(globalLoadSearch.value());
  }
  if (pattern.globalLoads.empty()) {
    LDBG("Did not find global load operation");
    return failure();
  }
  // check that all global loads have same type(i.e. shape and layout),
  // otherwise can not guarantee transformation overhead is cheap
  auto expectedLoadType = pattern.globalLoads.front()->getResult(0).getType();

  auto kDimRepeats =
      getDimRepeats(cast<RankedTensorType>(expectedLoadType), kDimNum);
  // kDimRepeats == 0 means loadType has unexpected layout
  // kDimRepeats == 1 means there are no room in k dimension in layout to
  // transpose in registers
  if (kDimRepeats < 2) {
    LDBG("Can not extend load layout");
    return failure();
  }

  for (auto load : pattern.globalLoads) {
    if (load->getResult(0).getType() != expectedLoadType) {
      LDBG("Mismatch between global loads result types");
      return failure();
    }
  }

  // TODO implement general heuristic,
  // analyzing local load/store vectorization and estimating bank conflicts

  return pattern;
}

ttg::BlockedEncodingAttr getThreadRakedBlockedEnc(Value dotOperand,
                                                  RankedTensorType loadType,
                                                  ModuleOp &mod) {
  // get the K dim according to dotOp operand's index
  auto tensorTy = cast<RankedTensorType>(dotOperand.getType());
  auto shape = tensorTy.getShape();
  auto opEnc = tensorTy.getEncoding();
  auto opDotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(opEnc);
  int kDimNum = opDotOpEnc.getOpIdx() == 0 ? 1 : 0;
  // get the current blocked encoding
  auto loadEnc = cast<RankedTensorType>(loadType).getEncoding();
  auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(loadEnc);

  auto numMaxIters = getDimRepeats(loadType, kDimNum);
  auto elemBitwidth = tensorTy.getElementType().getIntOrFloatBitWidth();
  // Current the widest is set to ds_write_b64
  // In some cases b64 works best, in others 128
  // TODO introduce a heuristic
  const unsigned dsBitWidth = 64;
  auto newKOuterDim = std::min(numMaxIters, dsBitWidth / elemBitwidth);
  LDBG("Choose the minimum of numIters: " << numMaxIters << " and numDtype: "
                                          << dsBitWidth / elemBitwidth);
  SmallVector<unsigned> newSizePerThread{blockedEnc.getSizePerThread()};
  newSizePerThread[kDimNum] = newKOuterDim;

  // return the new blocked encoding
  auto order = blockedEnc.getOrder();
  int numWarps = ttg::lookupNumWarps(mod.getOperation());
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
  return ttg::BlockedEncodingAttr::get(mod.getContext(), shape,
                                       newSizePerThread, order, numWarps,
                                       threadsPerWarp, numCTAs);
}

} // namespace

class TritonAMDGPUInThreadTransposePass
    : public TritonAMDGPUInThreadTransposeBase<
          TritonAMDGPUInThreadTransposePass> {

public:
  TritonAMDGPUInThreadTransposePass() = default;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    m.walk([&](tt::DotOp dotOp) {
      LDBG("DotOp under inspection: " << dotOp);
      auto mod = dotOp->getParentOfType<ModuleOp>();

      auto tryToConvertToThreadRaked = [&](Value operand) {
        LDBG("Consider " << operand);
        // Dot operand
        auto matchResult = matchThreadRakePattern(operand);
        if (!llvm::succeeded(matchResult)) {
          LDBG("Failed to match threadRake pattern and nothing to be done");
          return;
        }
        auto pattern = matchResult.value();

        LDBG("Adjusting global loads");
        RankedTensorType loadResultType = cast<RankedTensorType>(
            pattern.globalLoads[0].getResult().getType());
        auto newBlockedEnc =
            getThreadRakedBlockedEnc(operand, loadResultType, mod);
        auto loadShape = loadResultType.getShape();

        for (auto gLoad : pattern.globalLoads) {
          LDBG("operand newBlockedEnc = " << newBlockedEnc);
          convertLayout(newBlockedEnc, (Operation *)gLoad);
        }

        LDBG("Inserting transpose in registers before store in LDS");
        for (auto memOp : pattern.localMemStores)
          transposeInRegsitersBeforeStoreInLocalMemory(memOp, loadShape,
                                                       newBlockedEnc);

        LDBG("Adjust shared encoding");
        auto newSharedEncoding =
            createNewSharedEncoding(cast<RankedTensorType>(operand.getType()));
        for (auto memVal : pattern.sharedMemVals)
          changeSharedEncoding(memVal, newSharedEncoding);
      };
      // Check opA
      tryToConvertToThreadRaked(dotOp.getA());

      // Check opB
      tryToConvertToThreadRaked(dotOp.getB());
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUInThreadTransposePass() {
  return std::make_unique<TritonAMDGPUInThreadTransposePass>();
}
