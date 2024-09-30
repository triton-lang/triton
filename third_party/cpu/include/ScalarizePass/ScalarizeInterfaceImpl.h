#ifndef MLIR_DIALECT_TRITON_SCALARIZEINTERFACEIMPL_H
#define MLIR_DIALECT_TRITON_SCALARIZEINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace triton {
namespace cpu {

void registerTritonOpScalarizeExternalModels(DialectRegistry &registry);

} // namespace cpu
} // namespace triton
} // namespace mlir

#endif // MLIR_DIALECT_TRITON_SCALARIZEINTERFACEIMPL_H
