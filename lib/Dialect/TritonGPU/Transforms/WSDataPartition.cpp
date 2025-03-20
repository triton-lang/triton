#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {
namespace triton {
namespace gpu {

#define DEBUG_TYPE "tritongpu-warp-spec-data-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static bool oneVecCoversTheOther(SmallVector<AsyncTaskId> &one,
                                 SmallVector<AsyncTaskId> &other) {
  // Every element of other appears in one.
  for (AsyncTaskId t : other) {
    // If t doesn't appear in one, return false.
    bool found = false;
    for (AsyncTaskId t2 : one) {
      if (t2 == t) {
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

// Make sure the def chain contains the right taskId.
void fixTaskId(triton::FuncOp &funcOp) {
  funcOp.walk([&](Operation *op) {
    auto asyncTaskIds = getAsyncTaskIds(op);
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      // Do not update loads.
      if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(defOp))
        continue;
      auto defTaskIds = getAsyncTaskIds(defOp);
      // Make sure defTaskIds cover asyncTaskIds. Call addAsyncTaskIds if
      // necessary.
      if (!oneVecCoversTheOther(defTaskIds, asyncTaskIds)) {
        // Skip control flow ops.
        if (isa<scf::YieldOp, scf::ForOp, scf::IfOp>(op))
          continue;
        // Const ops with same value but different task ids can be folded.
        if (defOp->getDialect()->getNamespace() == "arith") {
          LLVM_DEBUG({
            LDBG("backward fixing taskId for");
            defOp->dump();
          });
          addAsyncTaskIds(defOp, asyncTaskIds);
          LLVM_DEBUG({
            LDBG("resulting");
            defOp->dump();
          });
        }
      }
      if (operand.hasOneUse() &&
          !oneVecCoversTheOther(asyncTaskIds, defTaskIds)) {
        // YieldOp may lose task attribute during MLIR canonicalization.
        if (isa<scf::YieldOp, scf::IfOp>(op)) {
          LLVM_DEBUG({
            LDBG("forward fixing taskId for");
            defOp->dump();
          });
          addAsyncTaskIds(op, defTaskIds);
          LLVM_DEBUG({
            LDBG("resulting");
            defOp->dump();
          });
        }
      }
    }
  });
}

static SmallVector<int64_t> getShape(Value v) {
  auto type = v.getType();
  if (auto type = dyn_cast<MemDescType>(v.getType())) {
    return {type.getShape().begin(), type.getShape().end()};
  } else if (auto type = dyn_cast<RankedTensorType>(v.getType())) {
    return {type.getShape().begin(), type.getShape().end()};
  } else if (auto type = dyn_cast<TensorDescType>(v.getType())) {
    return {type.getBlockType().getShape().begin(),
            type.getBlockType().getShape().end()};
  }
  return {};
}

bool needToSlice(Value v, int dim, int size) {
  auto shape = getShape(v);
  return shape.size() > dim && shape[dim] > size;
}

void getBackwardSliceToPartition(Value root, unsigned dim, int sliceSize,
                                 SetVector<Operation *> &backwardSlice) {
  SmallVector<Value> queue = {root};
  while (!queue.empty()) {
    auto v = queue.back();
    queue.pop_back();
    if (!needToSlice(v, dim, sliceSize))
      continue;
    if (auto op = v.getDefiningOp()) {
      if (backwardSlice.insert(op)) {
        if (op->hasTrait<OpTrait::Elementwise>() ||
            isa<arith::ConstantOp, arith::ExtSIOp, arith::ExtUIOp,
                arith::ExtFOp, BroadcastOp, ExpandDimsOp, MakeRangeOp, SplatOp,
                ConvertLayoutOp, triton::gpu::LocalAllocOp, LoadOp,
                ExperimentalDescriptorLoadOp, nvidia_gpu::TMEMAllocOp,
                nvidia_gpu::TMEMLoadOp>(op)) {
          for (Value operand : op->getOperands())
            queue.push_back(operand);
        } else if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(op)) {
          queue.push_back(dim == 0 ? dotOp.getA() : dotOp.getB());
          queue.push_back(dotOp.getC());
        } else if (auto dotOp = dyn_cast<nvidia_gpu::TCGen5MMAOp>(op)) {
          queue.push_back(dim == 0 ? dotOp.getA() : dotOp.getB());
          queue.push_back(dotOp.getD());
        } else if (auto tensorDescOp = dyn_cast<ReinterpretTensorDescOp>(op)) {
          continue;
        } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          // track yield value
          // find result index of v
          unsigned resultIndex = 0;
          for (int i = 0; i < op->getNumResults(); ++i) {
            if (op->getResult(i) == v) {
              resultIndex = i;
              break;
            }
          }

          auto thenYieldArg = ifOp.thenYield().getOperand(resultIndex);
          backwardSlice.insert(ifOp.thenYield());
          queue.push_back(thenYieldArg);
          auto elseYieldArg = ifOp.elseYield().getOperand(resultIndex);
          backwardSlice.insert(ifOp.elseYield());
          queue.push_back(elseYieldArg);
        } else {
          llvm_unreachable("Unexpected op");
        }
      }
    } else {
      assert(isa<BlockArgument>(v) && "value is not an operation or block ");
      auto bbArg = cast<BlockArgument>(v);
      Operation *bbAargOwner = bbArg.getOwner()->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(bbAargOwner)) {
        // track initial value
        auto initArg = forOp.getInitArgs()[bbArg.getArgNumber() - 1];
        queue.push_back(initArg);
        // track yield value
        auto yieldArg = forOp.getYieldedValues()[bbArg.getArgNumber() - 1];
        queue.push_back(yieldArg);
      }
    }
  }
};

void getForwardSliceToPartition(Value root, unsigned dim, int sliceSize,
                                SetVector<Operation *> &forwardSlice) {
  SmallVector<Value> queue = {root};
  llvm::SmallDenseSet<Value> seen;
  while (!queue.empty()) {
    auto v = queue.back();
    queue.pop_back();
    if (!seen.insert(v).second)
      continue;
    if (!needToSlice(v, dim, sliceSize))
      continue;
    getForwardSlice(v, &forwardSlice);
    for (Operation *op : forwardSlice) {
      if (op->getNumResults() > 0)
        seen.insert(op->getResult(0));
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        auto parentOp = yieldOp->getParentOp();
        for (OpOperand &operand : yieldOp->getOpOperands()) {
          if (seen.count(operand.get())) {
            queue.push_back(parentOp->getResult(operand.getOperandNumber()));
            forwardSlice.insert(parentOp);
          }
        }
      }
    }
  }
};

// Compute a closure of all ops originated from or being dependent on by the
// root op.
void getSliceToPartition(Value root, unsigned dim, int sliceSize,
                         SetVector<Operation *> &slice) {
  getBackwardSliceToPartition(root, dim, sliceSize, slice);
  SetVector<Operation *> forwardSlice;
  getForwardSliceToPartition(root, dim, sliceSize, forwardSlice);
  slice.insert(forwardSlice.begin(), forwardSlice.end());
  for (auto op : forwardSlice) {
    if (op->hasTrait<OpTrait::Elementwise>() ||
        isa<tt::StoreOp, ExperimentalDescriptorStoreOp>(op)) {
      for (OpOperand &operand : op->getOpOperands()) {
        getBackwardSliceToPartition(operand.get(), dim, sliceSize, slice);
      }
    } else if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(op)) {
      getBackwardSliceToPartition(dim == 0 ? dotOp.getA() : dotOp.getB(), dim,
                                  sliceSize, slice);
      getBackwardSliceToPartition(dotOp.getC(), dim, sliceSize, slice);
    } else if (auto dotOp = dyn_cast<nvidia_gpu::TCGen5MMAOp>(op)) {
      getBackwardSliceToPartition(dim == 0 ? dotOp.getA() : dotOp.getB(), dim,
                                  sliceSize, slice);
      getBackwardSliceToPartition(dotOp.getD(), dim, sliceSize, slice);
    }
  }
}

struct DataPartitionScheme {
  // Which dimension to partition. For dot, dim 0 means along M dimension, 1
  // means along N dimensiont.
  unsigned partitionDim = 0;
  unsigned numPartitions = 0;
  SetVector<Operation *> ops;
};

bool computePartitionScheme(triton::FuncOp &funcOp,
                            DataPartitionScheme &partitionScheme) {
  // Do not partition producer tasks

  // Use dot to drive the partition
  SetVector<Operation *> dots;

  // check all dot ops that have more than one async task id
  funcOp.walk([&](Operation *op) {
    auto asyncTaskIds = getAsyncTaskIds(op);
    if (asyncTaskIds.size() > 1) {
      if (isa<nvidia_gpu::WarpGroupDotOp, nvidia_gpu::TCGen5MMAOp>(op)) {
        dots.insert(op);
      }
    }
  });

  // Checking if all dots can be partitioned in the same way
  int numWarps = mlir::triton::gpu::lookupNumWarps(funcOp);
  for (auto op : dots) {
    // partition along M first, otherwise along N
    Value opndA, opndB, accumulator;

    if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(op)) {
      opndA = dotOp.getA();
      opndB = dotOp.getB();
      accumulator = dotOp.getD();
    } else if (auto dotOp = dyn_cast<nvidia_gpu::TCGen5MMAOp>(op)) {
      opndA = dotOp.getA();
      opndB = dotOp.getB();
      accumulator = dotOp.getD();
    }

    auto dotType = accumulator.getType();
    LLVM_DEBUG({
      LDBG("Computing partition scheme for");
      op->dump();
      LDBG("\n");
    });
    auto shapePerCTA = getShapePerCTA(dotType);
    if (shapePerCTA.size() != 2) {
      LDBG("partition not possible: shapePerCTA " << shapePerCTA.size());
      return false;
    }
    auto asyncTaskIds = getAsyncTaskIds(op);
    int sliceSizeM = shapePerCTA[0] / asyncTaskIds.size();
    int sliceSizeN = shapePerCTA[1] / asyncTaskIds.size();
    int partitionDim, partitionSize;
    Value partitionOperand;

    if (sliceSizeM >= 64) {
      LLVM_DEBUG({ LDBG("partition along M\n"); });
      partitionDim = 0;
      partitionSize = sliceSizeM;
      partitionOperand = opndA;
    } else if (sliceSizeN >= 256) {
      LLVM_DEBUG({ LDBG("partition along N\n"); });
      partitionDim = 1;
      partitionSize = sliceSizeN;
      partitionOperand = opndB;
    } else {
      LDBG("partition not possible: " << sliceSizeM << " " << sliceSizeN);
      return false;
    }

    if (partitionScheme.numPartitions == 0) {
      partitionScheme.partitionDim = partitionDim;
      partitionScheme.numPartitions = asyncTaskIds.size();
    } else {
      if (partitionScheme.partitionDim != partitionDim ||
          partitionScheme.numPartitions != asyncTaskIds.size()) {
        LDBG("partition not possible, in conflict with previous partition\n");
        return false;
      }
    }

    // Partition the slice closure
    SetVector<Operation *> &slice = partitionScheme.ops;
    getSliceToPartition(accumulator, partitionDim, partitionSize, slice);

    LLVM_DEBUG({
      partitionOperand.dump();
      LDBG("\n");
      LDBG(" slice:");
      for (auto &op : slice) {
        op->dump();
      }
      LDBG("\n");
    });

    for (auto op : partitionScheme.ops) {
      auto opTaskIds = getAsyncTaskIds(op);
      // skip check for control flow ops
      if (isa<scf::IfOp, scf::ForOp, scf::YieldOp>(op))
        continue;
#if 0
      if (opTaskIds.size() > partitionScheme.numPartitions) {
        LLVM_DEBUG({
          LDBG("partition not possible: numPartitions" << opTaskIds.size() << " " << partitionScheme.numPartitions);
          op->dump();
        });
        return false;
      }
#endif
    }
  }

  return !partitionScheme.ops.empty();
}

Operation *sliceOp(Value v, int offset, IRMapping &mappings,
                   IRMapping &reverseMappings,
                   DataPartitionScheme &partitionScheme);

Operation *sliceOp(Operation *op, int offset, IRMapping &mappings,
                   IRMapping &reverseMappings,
                   DataPartitionScheme &partitionScheme) {
  if (!partitionScheme.ops.contains(op))
    return op;
  if (mappings.contains(op))
    return mappings.lookupOrNull(op);
  if (reverseMappings.contains(op))
    return op;

  LLVM_DEBUG({
    LDBG("slicing:");
    op->dump();
    LDBG("\n");
  });

  int dim = partitionScheme.partitionDim;
  int numOfPartitions = partitionScheme.numPartitions;

  auto asyncTaskIds = getAsyncTaskIds(op);
  SmallVector<mlir::AsyncTaskId, 3> sliceTaskIds;
  if (asyncTaskIds.size() == numOfPartitions) {
    // We are slicing the op for consumer only
    sliceTaskIds.push_back(asyncTaskIds[offset]);
  } else if (asyncTaskIds.size() == 1) {
    // We are slicing the op for producer only
    sliceTaskIds.push_back(asyncTaskIds.front());
  } else if (asyncTaskIds.size() > numOfPartitions) {
    // We are slicing the op for both producer and consumer
    sliceTaskIds.push_back(asyncTaskIds.front());
    sliceTaskIds.push_back(asyncTaskIds[offset + 1]);
  } else {
    llvm_unreachable("Unexpected asyncTaskIds.size()");
  }

  OpBuilderWithAsyncTaskIds builder(op->getContext());
  builder.setAsynTaskIdsFromArray(sliceTaskIds);
  auto cloneAndSetResultType = [&](Operation *op) {
    builder.setInsertionPoint(op);
    auto newOp = builder.clone(*op, mappings);
    setAsyncTaskIds(newOp, sliceTaskIds);
    mappings.map(op, newOp);
    reverseMappings.map(newOp, op);
    // set result shape
    if (!op->getResults().empty()) {
      auto v = op->getResult(0);
      auto newV = newOp->getResult(0);
      if (auto type = dyn_cast<MemDescType>(v.getType())) {
        SmallVector<int64_t> shape{type.getShape().begin(),
                                   type.getShape().end()};
        int sliceSize = shape[dim] / numOfPartitions;
        shape[dim] = sliceSize;
        // change encoding for ttng.tensor_memory_encoding to match gen5.
        if (auto tmem = dyn_cast<nvidia_gpu::TensorMemoryEncodingAttr>(
                type.getEncoding())) {
          Attribute accEncoding =
              triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
                  builder.getContext(),
                  dim == 0 ? tmem.getBlockM() / 2 : tmem.getBlockM(),
                  dim == 1 ? tmem.getBlockN() / 2 : tmem.getBlockN(),
                  tmem.getUnpacked(), tmem.getCTASplitM(), tmem.getCTASplitN());
          auto newType =
              MemDescType::get(shape, type.getElementType(), accEncoding,
                               type.getMemorySpace(), type.getMutableMemory());
          newV.setType(newType);
        } else {
          auto newType =
              MemDescType::get(shape, type.getElementType(), type.getEncoding(),
                               type.getMemorySpace(), type.getMutableMemory());
          newV.setType(newType);
        }
      } else if (auto type = dyn_cast<RankedTensorType>(v.getType())) {
        SmallVector<int64_t> shape{type.getShape().begin(),
                                   type.getShape().end()};
        int sliceSize = shape[dim] / numOfPartitions;
        shape[dim] = sliceSize;
        auto newType = RankedTensorType::get(shape, type.getElementType(),
                                             type.getEncoding());
        newV.setType(newType);
      } else if (auto type = dyn_cast<TensorDescType>(v.getType())) {
        auto blockType = type.getBlockType();
        SmallVector<int64_t> shape{blockType.getShape().begin(),
                                   blockType.getShape().end()};
        int sliceSize = shape[dim] / numOfPartitions;
        shape[dim] = sliceSize;
        auto newBlockType =
            RankedTensorType::get(shape, blockType.getElementType());
        auto newType = TensorDescType::get(builder.getContext(), newBlockType);
        newV.setType(newType);
      }

      mappings.map(v, newV);
      reverseMappings.map(newV, v);
    }
    return newOp;
  };

  // slice operands first
  Operation *newOp;
  if (op->hasTrait<OpTrait::Elementwise>() ||
      isa<ConvertLayoutOp, BroadcastOp, SplatOp, ExpandDimsOp, LocalAllocOp,
          FpToFpOp>(op)) {
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    newOp = cloneAndSetResultType(op);
  } else if (auto tmemLdOp = dyn_cast<nvidia_gpu::TMEMLoadOp>(op)) {
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    auto srcTy = mappings.lookupOrNull(tmemLdOp.getSrc()).getType();
    auto type = cast<MemDescType>(srcTy);
    auto tmem = cast<nvidia_gpu::TensorMemoryEncodingAttr>(type.getEncoding());

    RankedTensorType oldRetType = tmemLdOp.getType();
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    int numWarps = mlir::triton::gpu::lookupNumWarps(op);
    auto CTALayout = getCTALayout(oldRetType.getEncoding());
    builder.setInsertionPoint(op);
    // The source op is already sliced at this point, so srcTy, type, tmem is
    // sliced. We use getTmemCompatibleLayout to get a block layout that is for
    // the sliced tmem here.
    Attribute newDistributedEncoding = nvidia_gpu::getTmemCompatibleLayout(
        tmem.getBlockM(), tmem.getBlockN(), retShapePerCTA, numWarps,
        CTALayout);

    // oldRetType is the desired output, we slice it and convert from the
    // compatible layout to the sliced desired output.
    SmallVector<int64_t> shape{oldRetType.getShape().begin(),
                               oldRetType.getShape().end()};
    int sliceSize = shape[dim] / numOfPartitions;
    shape[dim] = sliceSize;
    auto newAccType = RankedTensorType::get(shape, oldRetType.getElementType(),
                                            newDistributedEncoding);
    auto ld = builder.createWithAsyncTaskIds<triton::nvidia_gpu::TMEMLoadOp>(
        op->getLoc(), newAccType, mappings.lookupOrNull(tmemLdOp.getSrc()));

    auto newType = RankedTensorType::get(shape, oldRetType.getElementType(),
                                         oldRetType.getEncoding());
    auto cvtOp = builder.createWithAsyncTaskIds<ConvertLayoutOp>(op->getLoc(),
                                                                 newType, ld);
    auto v = tmemLdOp->getResult(0);
    auto newV = cvtOp->getResult(0);
    mappings.map(v, newV);
    reverseMappings.map(newV, v);
    newOp = cvtOp;
  } else if (auto tmemAllocOp = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    // Check for src.
    if (tmemAllocOp.getSrc()) {
      // src is blocked layout. apply convert layout on src
      auto srcTy = cast<RankedTensorType>(
          mappings.lookupOrNull(tmemAllocOp.getSrc()).getType());

      // convert from srcTy to a compatible blocked layout.
      auto retShapePerCTA = getShapePerCTA(srcTy);
      int numWarps = mlir::triton::gpu::lookupNumWarps(op);
      auto CTALayout = getCTALayout(srcTy.getEncoding());
      builder.setInsertionPoint(op);

      // calculate new tmem type.
      auto retType = cast<MemDescType>(tmemAllocOp.getType());
      SmallVector<int64_t> shape{retType.getShape().begin(),
                                 retType.getShape().end()};
      int sliceSize = shape[dim] / numOfPartitions;
      shape[dim] = sliceSize;
      auto tmem =
          cast<nvidia_gpu::TensorMemoryEncodingAttr>(retType.getEncoding());
      auto accEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
          builder.getContext(),
          dim == 0 ? tmem.getBlockM() / 2 : tmem.getBlockM(),
          dim == 1 ? tmem.getBlockN() / 2 : tmem.getBlockN(),
          tmem.getUnpacked(), tmem.getCTASplitM(), tmem.getCTASplitN());
      auto newType = MemDescType::get(shape, retType.getElementType(),
                                      accEncoding, retType.getMemorySpace(),
                                      retType.getMutableMemory());

      Attribute newDistributedEncoding = nvidia_gpu::getTmemCompatibleLayout(
          accEncoding.getBlockM(), accEncoding.getBlockN(), retShapePerCTA,
          numWarps, CTALayout);
      auto newAccType = RankedTensorType::get(
          srcTy.getShape(), srcTy.getElementType(), newDistributedEncoding);
      auto cvtOp = builder.createWithAsyncTaskIds<ConvertLayoutOp>(
          op->getLoc(), newAccType,
          mappings.lookupOrNull(tmemAllocOp.getSrc()));

      // replace tmemAllocOp with alloc, where the src is cvtOp.
      auto alloc =
          builder.createWithAsyncTaskIds<triton::nvidia_gpu::TMEMAllocOp>(
              op->getLoc(), newType, cvtOp);

      auto v = tmemAllocOp->getResult(0);
      auto newV = alloc->getResult(0);
      mappings.map(v, newV);
      reverseMappings.map(newV, v);
      newOp = alloc;
    } else
      newOp = cloneAndSetResultType(op);
  } else if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    builder.setInsertionPoint(op);
    auto valAttr = cast<DenseElementsAttr>(constOp.getValueAttr());
    auto valType = cast<ShapedType>(valAttr.getType());
    SmallVector<int64_t> shape{valType.getShape().begin(),
                               valType.getShape().end()};
    int sliceSize = shape[dim] / numOfPartitions;
    shape[dim] = sliceSize;
    auto newValType = valType.clone(shape);
    auto newValAttr = valAttr.resizeSplat(newValType);
    newOp = builder.createWithAsyncTaskIds<arith::ConstantOp>(op->getLoc(),
                                                              newValAttr);
    // Do not drop original task id as constant folding may lose one constant.
    setAsyncTaskIds(newOp, getAsyncTaskIds(op));
    auto v = op->getResult(0);
    auto newV = newOp->getResult(0);
    mappings.map(v, newV);
    reverseMappings.map(newV, v);
  } else if (auto makeRangeOp = dyn_cast<MakeRangeOp>(op)) {
    builder.setInsertionPoint(op);
    int newRangeStart = makeRangeOp.getStart();
    int newRangeEnd = makeRangeOp.getEnd();
    int sliceSize = (newRangeEnd - newRangeStart) / numOfPartitions;
    newRangeStart += offset * sliceSize;
    newRangeEnd = newRangeStart + sliceSize;
    auto v = op->getResult(0);
    auto type = cast<RankedTensorType>(v.getType());
    auto newType = RankedTensorType::get({sliceSize}, builder.getI32Type(),
                                         type.getEncoding());
    newOp = builder.createWithAsyncTaskIds<MakeRangeOp>(
        op->getLoc(), newType, newRangeStart, newRangeEnd);
    auto newV = newOp->getResult(0);
    mappings.map(v, newV);
    reverseMappings.map(newV, v);
  } else if (isa<StoreOp, LoadOp>(op)) {
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    // TODO: slice store base ptr
    newOp = cloneAndSetResultType(op);
  } else if (isa<ExperimentalDescriptorLoadOp, ExperimentalDescriptorStoreOp>(
                 op)) {
    SmallVector<int64_t> shape;
    Value coordVal;
    if (auto loadOp = dyn_cast<ExperimentalDescriptorLoadOp>(op)) {
      sliceOp(loadOp.getDesc(), offset, mappings, reverseMappings,
              partitionScheme);
      coordVal = loadOp.getIndices()[dim];
      shape = getShape(loadOp.getResult());
    } else if (auto storeOp = dyn_cast<ExperimentalDescriptorStoreOp>(op)) {
      sliceOp(storeOp.getDesc(), offset, mappings, reverseMappings,
              partitionScheme);
      coordVal = storeOp.getIndices()[dim];
      shape = getShape(storeOp.getSrc());
    }
    auto newCoordVal = coordVal;
    if (offset) {
      builder.setInsertionPointAfter(coordVal.getDefiningOp());
      Value offsetVal = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          op->getLoc(), offset * shape[dim] / numOfPartitions, 32);
      newCoordVal = builder.createWithAsyncTaskIds<arith::AddIOp>(
          op->getLoc(), coordVal, offsetVal);
      mappings.map(coordVal, newCoordVal);
      reverseMappings.map(newCoordVal, coordVal);
    }

    newOp = cloneAndSetResultType(op);
    if (isa<ExperimentalDescriptorLoadOp>(op)) {
      // map load result
      auto v = op->getResult(0);
      auto newV = newOp->getResult(0);
      mappings.map(v, newV);
      reverseMappings.map(newV, v);
    }
  } else if (auto tensorDescOp = dyn_cast<ReinterpretTensorDescOp>(op)) {
    newOp = cloneAndSetResultType(op);
  } else if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(op)) {
    // Only hanlde A and accumulator
    sliceOp(dim == 0 ? dotOp.getA() : dotOp.getB(), offset, mappings,
            reverseMappings, partitionScheme);
    sliceOp(dotOp.getC(), offset, mappings, reverseMappings, partitionScheme);
    newOp = cloneAndSetResultType(op);
  } else if (auto dotOp = dyn_cast<nvidia_gpu::TCGen5MMAOp>(op)) {
    // Only hanlde A and accumulator
    sliceOp(dim == 0 ? dotOp.getA() : dotOp.getB(), offset, mappings,
            reverseMappings, partitionScheme);
    sliceOp(dotOp.getD(), offset, mappings, reverseMappings, partitionScheme);
    newOp = cloneAndSetResultType(op);
  } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    // Add new loop arguments
    SmallVector<Value> newLoopArgs;
    for (auto initArg : forOp.getInitArgs())
      newLoopArgs.push_back(initArg);
    DenseMap<int, int> newArgIdices;
    for (unsigned i = 0; i < forOp.getInitArgs().size(); i++) {
      auto initArg = forOp.getInitArgs()[i];
      Value newInitArg;
      auto newInitArgOp =
          sliceOp(initArg, offset, mappings, reverseMappings, partitionScheme);
      if (auto bbArg = dyn_cast<BlockArgument>(initArg)) {
        // find the corresponding new block argument
        Block *parentBlock = bbArg.getOwner();
        unsigned argIndex = parentBlock->getNumArguments();
        for (unsigned i = 0; i < parentBlock->getNumArguments(); ++i) {
          if (parentBlock->getArgument(i) == bbArg) {
            argIndex = i;
            break;
          }
        }
        assert(argIndex < parentBlock->getNumArguments() &&
               "new init argment not found");
        Region *parentRegion = parentBlock->getParent();
        Region &newParentRegion =
            newInitArgOp->getRegion(parentRegion->getRegionNumber());
        newInitArg = parentRegion->getArgument(argIndex);
      } else {
        auto initArgOp = initArg.getDefiningOp();
        unsigned resultIndex = cast<mlir::OpResult>(initArg).getResultNumber();
        newInitArg = newInitArgOp->getResult(resultIndex);
      }

      if (newInitArg != initArg) {
        newLoopArgs.append({newInitArg});
        forOp.getBody()->insertArgument(forOp.getBody()->getNumArguments(),
                                        newInitArg.getType(), forOp.getLoc());
        newArgIdices[i] = newLoopArgs.size() - 1;
      }
    }

    // Create newForOp and take the region of forOp
    builder.setInsertionPoint(op);
    auto newForOp = builder.createWithAsyncTaskIds<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newLoopArgs);
    assert(newForOp.getRegionIterArgs().size() ==
           newForOp.getInitArgs().size());
    newForOp->setAttrs(forOp->getAttrs());
    partitionScheme.ops.insert(newForOp);
    newOp = newForOp;

    // Replace forOp with newForOp
    newForOp.getRegion().takeBody(forOp.getRegion());
    for (unsigned i = 0; i < forOp.getNumResults(); ++i)
      forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
    op->setAttr("to_be_removed", builder.getUnitAttr());

    // Map new loop arguments
    for (auto argIndex : newArgIdices) {
      Value v = newForOp.getResult(argIndex.first);
      Value newV = newForOp.getResult(argIndex.second);
      mappings.map(v, newV);
      reverseMappings.map(newV, v);

      auto regionArg = newForOp.getRegionIterArg(argIndex.first);
      auto newRegionArg = newForOp.getRegionIterArg(argIndex.second);
      mappings.map(regionArg, newRegionArg);
      reverseMappings.map(newRegionArg, regionArg);
    }
  } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    // Slice the yield op and update if results
    auto thenYieldOp = ifOp.thenYield();
    auto elseYieldOp = ifOp.elseYield();
    auto newThenYieldOp = sliceOp(thenYieldOp, offset, mappings,
                                  reverseMappings, partitionScheme);
    sliceOp(elseYieldOp, offset, mappings, reverseMappings, partitionScheme);
    assert(newThenYieldOp->getNumOperands() > ifOp->getNumResults() &&
           "no need to slice if op");
    // Clone ifOp with updated results but re-use the original regions.
    builder.setInsertionPoint(op);
    SmallVector<Type, 4> newResultTypes;
    for (auto thenResult : thenYieldOp.getResults()) {
      newResultTypes.push_back(thenResult.getType());
    }
    auto newIfOp = builder.create<scf::IfOp>(ifOp.getLoc(), newResultTypes,
                                             ifOp.getCondition());
    // Move the original regions to the cloned operation.
    newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
    newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
    newOp = newIfOp;
    newIfOp->setAttrs(ifOp->getAttrs());
    partitionScheme.ops.insert(newIfOp);
    ifOp->setAttr("to_be_removed", builder.getUnitAttr());

    // Replace ifOp with newIfOp
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i)
      ifOp.getResult(i).replaceAllUsesWith(newIfOp.getResult(i));

    // Map if results based on the mapping for yield
    for (auto &v : thenYieldOp->getOpOperands()) {
      auto newV = mappings.lookupOrNull(v.get());
      if (newV) {
        int operandIndex = v.getOperandNumber();
        // find the corresponding operand index of newV in newYieldOp
        int newOperandIndex = -1;
        for (int i = 0; i < newThenYieldOp->getNumOperands(); ++i) {
          if (newThenYieldOp->getOperand(i) == newV) {
            newOperandIndex = i;
            break;
          }
        }
        assert(newOperandIndex >= 0 && "newV not found in newYieldOp");
        auto newResult = newIfOp.getResult(operandIndex);
        auto newSlicedResult = newIfOp.getResult(newOperandIndex);
        mappings.map(newResult, newSlicedResult);
        reverseMappings.map(newSlicedResult, newResult);
      }
    }
  } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
    int num = yieldOp.getNumOperands();
    for (int i = 0; i < num; i++) {
      auto operand = yieldOp.getOperand(i);
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
      if (auto newV = mappings.lookupOrNull(operand))
        yieldOp->insertOperands(op->getNumOperands(), newV);
    }
    newOp = op;
  } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
    assert(reduceOp.getAxis() != partitionScheme.partitionDim &&
           "reduce should not happen on the partitioned dimension");
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    newOp = cloneAndSetResultType(op);
    // recursively set async task ids for child ops
    newOp->walk(
        [&](Operation *childOp) { setAsyncTaskIds(childOp, sliceTaskIds); });
  } else {
    llvm_unreachable("unsupported op type");
  }

  LLVM_DEBUG({
    LDBG("resulting");
    newOp->dump();
    LDBG("\n");
  });
  mappings.map(op, newOp);
  reverseMappings.map(newOp, op);
  return newOp;
}

Operation *sliceOp(Value v, int offset, IRMapping &mappings,
                   IRMapping &reverseMappings,
                   DataPartitionScheme &partitionScheme) {
  if (auto op = v.getDefiningOp()) {
    return sliceOp(op, offset, mappings, reverseMappings, partitionScheme);
  } else {
    assert(isa<BlockArgument>(v) && "value is not an operation or block ");
    auto bbArg = cast<BlockArgument>(v);
    Operation *bbAargOwner = bbArg.getOwner()->getParentOp();
    return sliceOp(bbAargOwner, offset, mappings, reverseMappings,
                   partitionScheme);
  }
}

void doDeepCleanup(triton::FuncOp &funcOp,
                   DataPartitionScheme &partitionScheme) {
  SmallVector<Operation *> opsToDelete;
  DenseSet<Operation *> opsCanBeTriviallyDead;

  do {
    opsToDelete.clear();
    opsCanBeTriviallyDead.clear();

    // Identify root ops that are not used so to be deleted.
    funcOp.walk([&](Operation *op) {
      if (isa<scf::YieldOp>(op))
        return;
      if (!partitionScheme.ops.contains(op))
        return;

      // Ignore the side effect of ops that are already sliced. The
      // resulting ops preserve the side effect.
      if (!isMemoryEffectFree(op))
        opsCanBeTriviallyDead.insert(op);

      bool notUsed = true;
      for (auto result : op->getResults()) {
        if (!result.getUsers().empty()) {
          notUsed = false;
          break;
        }
      }
      if (notUsed)
        opsToDelete.push_back(op);
    });

    LLVM_DEBUG({
      LDBG("opsToDelete:\n");
      for (auto op : opsToDelete) {
        LDBG("op: ");
        op->dump();
      }
      LDBG("\n");
    });

    if (opsToDelete.empty())
      return;

    // Delete root ops.
    for (auto op : opsToDelete) {
      partitionScheme.ops.remove(op);
      op->erase();
    }

    LLVM_DEBUG({
      LDBG("prior to loop arg deletion:");
      funcOp.dump();
    });

    // delete block arguments
    RewritePatternSet cleanUpPatterns(funcOp.getContext());
    populateForOpDeadArgumentElimination(cleanUpPatterns,
                                         opsCanBeTriviallyDead);
    scf::ForOp::getCanonicalizationPatterns(cleanUpPatterns,
                                            funcOp.getContext());
    scf::IfOp::getCanonicalizationPatterns(cleanUpPatterns,
                                           funcOp.getContext());
    if (applyPatternsGreedily(funcOp, std::move(cleanUpPatterns)).failed()) {
      llvm_unreachable("failed to clean up");
      // signalPassFailure();
    }
  } while (!opsToDelete.empty());
}

void partitionTasks(triton::FuncOp &funcOp, int numConsumerGroups) {

  // op -> (partition dim, num of partitions)
  DataPartitionScheme partitionScheme;
  if (!computePartitionScheme(funcOp, partitionScheme)) {
    if (numConsumerGroups > 1)
      llvm::errs() << "computePartitionScheme failed when requested\n";
    return;
  }

  for (int i = 0; i < partitionScheme.numPartitions; i++) {
    IRMapping mappings, reverseMappings;

    LLVM_DEBUG({ LDBG("partitioning op for task " << i << ":\n"); });

    // TODO: compute a topological order for partitionScheme.ops and
    // slice in that order.
    int numOps = partitionScheme.ops.size();
    for (int j = 0; j < numOps; j++) {
      auto op = partitionScheme.ops[j];
      sliceOp(op, i, mappings, reverseMappings, partitionScheme);
    }

    // clean up
    LLVM_DEBUG({
      LDBG("prior to clean up:");
      funcOp.dump();
    });
    SmallVector<Operation *> opsToDelete;
    for (auto op : partitionScheme.ops) {
      if (op->hasAttr("to_be_removed"))
        opsToDelete.push_back(op);
    }
    for (auto op : opsToDelete) {
      partitionScheme.ops.remove(op);
      op->erase();
    }
  }

  LLVM_DEBUG({
    LDBG("prior to final cleanup:");
    funcOp.dump();
  });

  // Make sure original ops are not used
  doDeepCleanup(funcOp, partitionScheme);

  // Make sure original ops are not used
  LLVM_DEBUG({
    LDBG("after partition");
    funcOp.dump();
    LDBG("\n");
  });
  fixTaskId(funcOp);
}

#define GEN_PASS_DEF_TRITONGPUWSDATAPARTITION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUWSDataPartitionPass
    : public impl::TritonGPUWSDataPartitionBase<TritonGPUWSDataPartitionPass> {
public:
  using impl::TritonGPUWSDataPartitionBase<
      TritonGPUWSDataPartitionPass>::TritonGPUWSDataPartitionBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    if (numConsumerGroups == 0)
      return;
    partitionTasks(funcOp, numConsumerGroups);
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
