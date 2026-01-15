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
    InterfaceMethod<
      /*desc=*/"Returns the element type of the block",
      /*retType=*/"mlir::Type",
      /*methodName=*/"getElementType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImpl=*/[{
        return $_type.getBlockType().getElementType();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the shape of the block",
      /*retType=*/"llvm::ArrayRef<int64_t>",
      /*methodName=*/"getShape",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImpl=*/[{
        return $_type.getBlockType().getShape();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the rank of the block",
      /*retType=*/"int64_t",
      /*methodName=*/"getRank",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImpl=*/[{
        return $_type.getBlockType().getRank();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the encoding attribute of the block type, if any",
      /*retType=*/"mlir::Attribute",
      /*methodName=*/"getEncoding",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImpl=*/[{
        return $_type.getBlockType().getEncoding();
      }]
    >,
  ];
}

#endif // TRITON_TYPE_INTERFACES
