#ifndef TRITON_TYPES
#define TRITON_TYPES

include "triton/Dialect/Triton/IR/TritonDialect.td"

//
// Types
//
class TritonTypeDef<string name, string _mnemonic>
    : TypeDef<Triton_Dialect, name> {
    // Used by printer/parser
    let mnemonic = _mnemonic;
}

// Floating-point Type
def F8 : TritonTypeDef<"Float8", "f8">;

def TT_Float : AnyTypeOf<[F8, F16, BF16, F32, F64], "floating-point">;
def TT_FloatTensor : TensorOf<[TT_Float]>;
def TT_FloatLike : AnyTypeOf<[TT_Float, TT_FloatTensor]>;

// Boolean Type
// TT_Bool -> I1
def TT_BoolTensor : TensorOf<[I1]>;
def TT_BoolLike : AnyTypeOf<[I1, TT_BoolTensor]>;

// Integer Type
def TT_Int : AnyTypeOf<[I1, I8, I16, I32, I64], "integer">;
def TT_IntTensor : TensorOf<[TT_Int]>;
def TT_IntLike : AnyTypeOf<[TT_Int, TT_IntTensor]>;

// I32 Type
// TT_I32 -> I32
// TT_I32Tensor -> I32Tensor
def TT_I32Like: AnyTypeOf<[I32, I32Tensor]>;

// I64 Type
// TT_I64 -> I64
// TT_I64Tensor -> I64Tensor
def TT_I64Like: AnyTypeOf<[I64, I64Tensor]>;

// Pointer Type
def TT_Ptr : TritonTypeDef<"Pointer", "ptr"> {
    let summary = "pointer type";

    let description = [{
        Triton PointerType
    }];

    let parameters = (ins "Type":$pointeeType, "int":$addressSpace);

    let builders = [
        TypeBuilderWithInferredContext<(ins
            "Type":$pointeeType,
            "int":$addressSpace
        ), [{
            return $_get(pointeeType.getContext(), pointeeType, addressSpace);
        }]>
    ];

    let skipDefaultBuilders = 1;
}
def TT_PtrTensor : TensorOf<[TT_Ptr]>;
def TT_PtrLike : AnyTypeOf<[TT_Ptr, TT_PtrTensor]>;

def TT_FpIntTensor : AnyTypeOf<[TT_FloatTensor, TT_IntTensor]>;
def TT_Tensor : AnyTypeOf<[TT_FpIntTensor, TT_PtrTensor]>;

def TT_Type : AnyTypeOf<[TT_FloatLike, TT_IntLike, TT_PtrLike]>;

#endif
