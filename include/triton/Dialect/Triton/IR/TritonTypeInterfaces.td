#ifndef TRITON_TYPE_INTERFACES
#define TRITON_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TensorDescInterface
//===----------------------------------------------------------------------===//

def TT_TensorDescInterface : TypeInterface<"TensorDescInterface"> {
  let cppNamespace = "::mlir::triton";

  let description = [{
    Common interface for tensor descriptor types.

    This interface provides a unified API for different tensor descriptor
    implementations (e.g., tiled TensorDescType, im2col TensorDescIm2ColType).
    All tensor descriptors share the concept of a "block type" which describes
    the shape and element type of the data block being accessed.

    Concrete implementations:
    - TensorDescType (Triton dialect): Basic tiled tensor descriptor
    - TensorDescIm2ColType (TritonNvidiaGPU dialect): Im2col tensor descriptor
      with additional convolution parameters
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the block type of the tensor descriptor",
      /*retType=*/"mlir::RankedTensorType",
      /*methodName=*/"getBlockType",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/"Returns the block type with signless integer element type",
      /*retType=*/"mlir::RankedTensorType",
      /*methodName=*/"getSignlessBlockType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImpl=*/[{
        auto resTy = $_type.getBlockType();
        if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(resTy.getElementType())) {
          auto width = resTy.getElementTypeBitWidth();
          auto signlessTy = mlir::IntegerType::get($_type.getContext(), width);
          resTy = resTy.clone(signlessTy);
        }
        return resTy;
      }]
    >,
  ];
}

#endif // TRITON_TYPE_INTERFACES
