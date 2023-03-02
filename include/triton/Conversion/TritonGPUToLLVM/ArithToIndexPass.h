#ifndef TRITON_CONVERSION_ARITH_TO_INDEX_H
#define TRITON_CONVERSION_ARITH_TO_INDEX_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonConvertArithToIndexPass();

}
} // namespace mlir

#endif