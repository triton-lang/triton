#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace LLVMDIUtils {
LLVM::DITypeAttr convertType(MLIRContext *context, mlir::Type type);
LLVM::DITypeAttr convertPtrType(MLIRContext *context, mlir::Type pointerType,
                                mlir::Type pointeeType, unsigned sizeInBits);

FileLineColLoc extractFileLoc(Location loc, bool getCaller = true);
std::optional<unsigned> calcBitWidth(mlir::Type type);
} // namespace LLVMDIUtils
} // namespace mlir
