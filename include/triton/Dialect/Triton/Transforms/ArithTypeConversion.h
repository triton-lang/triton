#ifndef TRITON_DIALECT_TRITON_TRANSFORMS_ARITH_TYPE_CONVERSION_H_
#define TRITON_DIALECT_TRITON_TRANSFORMS_ARITH_TYPE_CONVERSION_H_
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::arith {

/**
 * @brief Provides helper patterns for converting arith operations using a type
 * converter.
 *
 * Note at of the time of writing this isn't provided in upstream mlir.
 */
void populateArithTypeConversions(const TypeConverter &converter,
                                  RewritePatternSet &patterns);

} // namespace mlir::arith

#endif // TRITON_DIALECT_TRITON_TRANSFORMS_ARITH_TYPE_CONVERSION_H_
