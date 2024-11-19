#ifndef TRITON_OP_INTERFACES
#define TRITON_OP_INTERFACES

include "mlir/IR/OpBase.td"


def TransposeOpInterface : OpInterface<"TransposeOpInterface"> {
  let description = [{
    This interface is implemented by operations that perform a transpose.
    It provides methods to access common properties such as the order attribute and the source operand.
  }];

  let cppNamespace = "::mlir::triton";

  let methods = [
    InterfaceMethod<
			/*desc=*/[{
			  Get the source operand of the transposition.
		  }],
			/*retType=*/"::mlir::Value",
			/*methodName=*/"getSrc",
			/*args=*/(ins)>,
		InterfaceMethod<
			/*desc=*/[{
				Get the rank of the source operand.
			}],
			/*retType=*/"int64_t",
			/*methodName=*/"getRank",
			/*args=*/(ins)>,
    InterfaceMethod<
			/*desc=*/[{
			  Get the order of the transposition.
		  }],
			/*retType=*/"::mlir::ArrayRef<int32_t>",
			/*methodName=*/"getOrder",
			/*args=*/(ins)>
  ];

	let verify = [{ return ::mlir::triton::impl::verifyTransposeOpInterface($_op); }];
}


#endif // TRITON_OP_INTERFACES
