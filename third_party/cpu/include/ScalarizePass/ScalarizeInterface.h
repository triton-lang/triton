#ifndef MLIR_INTERFACES_SCALARIZE_INTERFACE_H_
#define MLIR_INTERFACES_SCALARIZE_INTERFACE_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include "mlir/IR/OpDefinition.h"

/// Include the ODS generated interface header files.
#include "cpu/include/ScalarizePass/ScalarizeInterface.h.inc"

namespace mlir {
namespace triton {
namespace cpu {

mlir::Value computeScalarValue(mlir::Operation *scalarizationOp,
                               mlir::Value vals,
                               mlir::ArrayRef<int64_t> indices,
                               mlir::PatternRewriter &rewriter);

mlir::Value computeScalarValue(mlir::Operation *scalarizationOp,
                               mlir::Value vals, mlir::ValueRange indices,
                               mlir::PatternRewriter &rewriter);

bool canComputeScalarValue(mlir::Value vals);
} // namespace cpu
} // namespace triton
} // namespace mlir

#endif // MLIR_INTERFACES_SCALARIZE_INTERFACE_H_
