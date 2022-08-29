#ifndef TRITON_ANALYSIS_UTILITY_H
#define TRITON_ANALYSIS_UTILITY_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <string>
namespace mlir {

bool isSharedEncoding(Value value);

bool maybeSharedAllocationOp(Operation *op);

std::string getValueOperandName(Value value, AsmState &state);

} // namespace mlir

#endif // TRITON_ANALYSIS_UTILITY_H
