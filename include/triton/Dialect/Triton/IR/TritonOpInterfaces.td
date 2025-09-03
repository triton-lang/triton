#ifndef TRITON_OP_INTERFACES
#define TRITON_OP_INTERFACES

include "mlir/IR/OpBase.td"


def TransposeOpInterface : OpInterface<"TransposeOpInterface"> {
  let description = [{
    This interface is implemented by operations that perform a transpose.
    It provides methods to access common properties such as the order attribute
    and the source operand.
  }];

  let cppNamespace = "::mlir::triton";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the source operand of the transposition.",
      /*retType=*/"::mlir::Value",
      /*methodName=*/"getSrc",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Get the order of the transposition.",
      /*retType=*/"::mlir::ArrayRef<int32_t>",
      /*methodName=*/"getOrder",
      /*args=*/(ins)>
  ];

  let verify = [{
    return ::mlir::triton::impl::verifyTransposeOpInterface($_op);
  }];
}

def DotOpInterface : OpInterface<"DotOpInterface"> {
  let description = [{
    This interface is implemented by operations that perform a dot product.
  }];

  let cppNamespace = "::mlir::triton";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the LHS A tensor",
      /*retType=*/"::mlir::Value",
      /*methodName=*/"getA",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Get the RHS B tensor",
      /*retType=*/"::mlir::Value",
      /*methodName=*/"getB",
      /*args=*/(ins)>,
  InterfaceMethod<
      /*desc=*/"Verify the dimensions of the A and B DotOp operands.",
      /*retType=*/"bool",
      /*methodName=*/"verifyDims",
      /*args=*/(ins)>,
  InterfaceMethod<
      /*desc=*/"Verify the dimensions of the DotOp output.",
      /*retType=*/"bool",
      /*methodName=*/"verifyOutputDims",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImpl=*/ [{
        auto aTy = cast<ShapedType>($_op.getA().getType());
        auto bTy = cast<ShapedType>($_op.getB().getType());
        auto cTy = cast<ShapedType>($_op->getOperand(2).getType());
        auto aShape = aTy.getShape();
        auto bShape = bTy.getShape();
        auto cShape = cTy.getShape();
        return cShape[cShape.size() - 2] == aShape[aShape.size() - 2] &&
               cShape[cShape.size() - 1] == bShape[aShape.size() - 1];
      }]>
  ];

  let verify = [{ return ::mlir::triton::impl::verifyDotOpInterface($_op); }];
}

def TT_DescriptorOpInterface : OpInterface<"DescriptorOpInterface"> {
  let description = [{
    Common interface to get the descriptor argument from an operation on tensor descriptors.
  }];

  let cppNamespace = "::mlir::triton";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the descriptor",
      /*retType=*/"::mlir::TypedValue<mlir::triton::TensorDescType>",
      /*methodName=*/"getDesc",
      /*args=*/(ins)>,
  ];
}

def TT_DescriptorStoreLikeOpInterface : OpInterface<"DescriptorStoreLikeOpInterface", [TT_DescriptorOpInterface]> {
  let cppNamespace = "::mlir::triton";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get Source tensor",
      /*retType=*/"::mlir::TypedValue<mlir::RankedTensorType>",
      /*methodName=*/"getSrc",
      /*args=*/(ins)>,
  ];
}


#endif // TRITON_OP_INTERFACES
