// TritonAppleGPU dialect registration

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

// Pull in tablegen-generated definitions
#include "Dialect/TritonAppleGPU/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/TritonAppleGPU/IR/TritonAppleGPUAttrDefs.cpp.inc"

namespace mlir::triton::applegpu {

void TritonAppleGPUDialect::initialize() {
    registerTypes();
    addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/TritonAppleGPU/IR/TritonAppleGPUAttrDefs.cpp.inc"
    >();
}

void TritonAppleGPUDialect::registerTypes() {
    // No custom types yet — attrs only
}

// ── AppleMmaEncodingAttr methods ─────────────────────────────────────────────

LogicalResult AppleMmaEncodingAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    llvm::ArrayRef<unsigned> warpsPerCTA) {
    if (warpsPerCTA.size() != 2)
        return emitError() << "AppleMmaEncoding requires exactly 2 warpsPerCTA dims";
    if (warpsPerCTA[0] == 0 || warpsPerCTA[1] == 0)
        return emitError() << "AppleMmaEncoding warpsPerCTA must be non-zero";
    return success();
}

// Row-major iteration order: dim0 (M) first, then dim1 (N)
SmallVector<unsigned> AppleMmaEncodingAttr::getRepOrder() const {
    return {0, 1};
}

// Operand rep order: both A (opIdx=0) and B (opIdx=1) use {0, 1}
SmallVector<unsigned> AppleMmaEncodingAttr::getRepOrderForOperand(int opIdx) const {
    return {0, 1};
}

} // namespace mlir::triton::applegpu
