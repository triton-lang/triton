#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace LLVMDIUtils {
LLVM::DITypeAttr convertType(MLIRContext *context, mlir::Type type);
LLVM::DITypeAttr convertPtrType(MLIRContext *context,
                                LLVM::LLVMPointerType pointerType,
                                mlir::Type pointeeType, DataLayout datalayout);
LLVM::DITypeAttr convertStructType(MLIRContext *context,
                                   LLVM::LLVMStructType structType,
                                   LLVM::DIFileAttr fileAttr,
                                   DataLayout datalayout, int64_t line);
LLVM::DITypeAttr convertArrayType(MLIRContext *context,
                                  LLVM::LLVMArrayType arrayType,
                                  LLVM::DIFileAttr fileAttr,
                                  DataLayout datalayout, int64_t line);
FileLineColLoc extractFileLoc(Location loc, bool getCaller = true);
std::optional<unsigned> calcBitWidth(mlir::Type type);
} // namespace LLVMDIUtils
} // namespace mlir
