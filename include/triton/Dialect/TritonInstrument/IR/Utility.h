#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

namespace mlir::triton::instrument {

Operation *createStoreScratchMemory(OpBuilder &b, Location loc, Value alloc,
                                    Value tensor, RankedTensorType tensorType);
Operation *createLoadScratchMemory(OpBuilder &b, Location loc, Value alloc,
                                   RankedTensorType tensorType);
Value expandOuterSlicedDim(OpBuilder &b, Location loc, Value tensor);
TypedValue<RankedTensorType> createConstIntTensor(OpBuilder &builder,
                                                  Location loc, int val,
                                                  RankedTensorType tensorType);

} // namespace mlir::triton::instrument
