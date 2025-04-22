
#pragma once
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton {
LogicalResult verifyDescriptorLoadStoreType(Operation *op, TensorDescType desc,
                                            RankedTensorType tensor);
}
