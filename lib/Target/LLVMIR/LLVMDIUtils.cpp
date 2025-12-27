#include "lib/Target/LLVMIR/LLVMDIUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "llvm/BinaryFormat/Dwarf.h"

namespace mlir {

// Note: mlir does not provided any built-in conversion from mlir::Type to
// mlir::LLVM::DITypeAttr
LLVM::DITypeAttr LLVMDIUtils::convertType(MLIRContext *context,
                                          mlir::Type type) {
  if (type.isInteger(1)) {
    return LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type,
                                      mlir::StringAttr::get(context, "bool"),
                                      type.getIntOrFloatBitWidth(),
                                      llvm::dwarf::DW_ATE_boolean);
  }
  if (type.isInteger()) {
    return LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type,
                                      mlir::StringAttr::get(context, "int"),
                                      type.getIntOrFloatBitWidth(),
                                      llvm::dwarf::DW_ATE_signed);
  } else if (type.isF16()) {
    return LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type,
                                      mlir::StringAttr::get(context, "half"),
                                      type.getIntOrFloatBitWidth(),
                                      llvm::dwarf::DW_ATE_float);
  } else if (type.isF32()) {
    return LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type,
                                      mlir::StringAttr::get(context, "float"),
                                      type.getIntOrFloatBitWidth(),
                                      llvm::dwarf::DW_ATE_float);
  } else if (type.isF64()) {
    return LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type,
                                      mlir::StringAttr::get(context, "double"),
                                      type.getIntOrFloatBitWidth(),
                                      llvm::dwarf::DW_ATE_float);
  } else if (mlir::isa<mlir::VectorType>(type)) {
    if (auto vectorTypeSize = calcBitWidth(type); vectorTypeSize.has_value()) {
      return LLVM::DIBasicTypeAttr::get(
          context, llvm::dwarf::DW_TAG_base_type,
          mlir::StringAttr::get(context, "vector"), vectorTypeSize.value(),
          llvm::dwarf::DW_ATE_float);
    } else {
      // TODO: falling back to unknown_type, perhaps theres a better way to
      // handle when element type size is not determined
    }
  }
  return LLVM::DIBasicTypeAttr::get(
      context, llvm::dwarf::DW_TAG_base_type,
      mlir::StringAttr::get(context, "unknown_type"), 0,
      llvm::dwarf::DW_ATE_signed);
}

LLVM::DITypeAttr LLVMDIUtils::convertPtrType(MLIRContext *context,
                                             mlir::Type pointerType,
                                             mlir::Type pointeeType,
                                             unsigned sizeInBits) {
  // LLVMPointerType does not include pointee info, need to pass from external
  // source
  if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(pointerType)) {
    unsigned addrSpace = ptrType.getAddressSpace();

    LLVM::DITypeAttr diElTypeAttr = convertType(context, pointeeType);
    LLVM::DITypeAttr diTypeAttr = mlir::LLVM::DIDerivedTypeAttr::get(
        context, llvm::dwarf::DW_TAG_pointer_type,
        mlir::StringAttr::get(context, "pointer"), diElTypeAttr, sizeInBits,
        /*alignInBits=*/0, /*offset=*/0, addrSpace, /*extra data=*/nullptr);
    return diTypeAttr;
  }
  // Return unknown_type if fail to construct DIDerivedTypeAttr with
  // WD_TAG_pointer_type.
  return LLVM::DIBasicTypeAttr::get(
      context, llvm::dwarf::DW_TAG_base_type,
      mlir::StringAttr::get(context, "unknown_type"), 0,
      llvm::dwarf::DW_ATE_signed);
}

std::optional<unsigned> LLVMDIUtils::calcBitWidth(mlir::Type type) {
  if (type.isIntOrFloat()) {
    return type.getIntOrFloatBitWidth();
  } else if (mlir::isa<mlir::VectorType>(type)) {
    auto vectorType = dyn_cast<mlir::VectorType>(type);
    llvm::ArrayRef<int64_t> shape = vectorType.getShape();
    mlir::Type elementType = vectorType.getElementType();
    llvm::ArrayRef<bool> scalableDims = vectorType.getScalableDims();
    unsigned size = 1;
    for (auto i : shape) {
      size *= i;
    }

    if (auto elementTypeSize = calcBitWidth(elementType);
        elementTypeSize.has_value()) {
      return size * elementTypeSize.value();
    }
  }

  return std::nullopt;
}

/// Attempt to extract a filename for the given loc.
FileLineColLoc LLVMDIUtils::extractFileLoc(Location loc, bool getCaller) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    return fileLoc;
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return extractFileLoc(nameLoc.getChildLoc());
  if (auto opaqueLoc = dyn_cast<OpaqueLoc>(loc))
    return extractFileLoc(opaqueLoc.getFallbackLocation());
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc))
    return extractFileLoc(fusedLoc.getLocations().front());
  if (auto callerLoc = dyn_cast<CallSiteLoc>(loc))
    return getCaller ? extractFileLoc(callerLoc.getCaller())
                     : extractFileLoc(callerLoc.getCallee());
  StringAttr unknownFile = mlir::StringAttr::get(loc.getContext(), "<unknown>");
  return mlir::FileLineColLoc::get(unknownFile, 0, 0);
}

} // namespace mlir
