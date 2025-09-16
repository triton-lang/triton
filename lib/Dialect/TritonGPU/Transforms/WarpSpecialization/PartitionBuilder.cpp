#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

Value PartitionBuilder::intCst(int value, unsigned width) {
  return create<arith::ConstantIntOp>(value, width);
}

Value PartitionBuilder::boolCst(bool value) {
  return intCst(value, /*width=*/1);
}

void PartitionBuilder::assignPartition(Operation *op, Partition &partition) {
  setPartition(op, &partition);
}

StageCluster triton::gpu::getStageCluster(Operation *op) {
  auto stageAttr = op->getAttrOfType<IntegerAttr>(kLoopStageAttrName);
  auto clusterAttr = op->getAttrOfType<IntegerAttr>(kLoopClusterAttrName);
  if (!stageAttr || !clusterAttr)
    return std::nullopt;
  return std::make_pair(stageAttr.getInt(), clusterAttr.getInt());
}

void triton::gpu::setStageCluster(OpBuilder &b, Operation *op,
                                  StageCluster stageCluster) {
  if (stageCluster) {
    op->setAttr(kLoopStageAttrName, b.getI32IntegerAttr(stageCluster->first));
    op->setAttr(kLoopClusterAttrName,
                b.getI32IntegerAttr(stageCluster->second));
  }
}
