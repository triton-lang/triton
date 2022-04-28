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

def F8 : TritonTypeDef<"Float8", "f8">;
def BF8 : TritonTypeDef<"BFloat8", "bf8">;

def TT_Float : AnyTypeOf<[F16, BF16, F32, F64], "floating-point">;
def TT_FloatTensor : TensorOf<[TT_Float]>;

// IntegerType
def TT_Int : AnyTypeOf<[I1, I8, I16, I32, I64], "integer">;
def TT_IntegerTensor : TensorOf<[TT_Int]>;

// PointerType
def TT_Pointer : TritonTypeDef<"Pointer", "ptr"> {
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
def TT_PtrTensor : TensorOf<[TT_Pointer]>;

def TT_FpIntTensor : AnyTypeOf<[TT_FloatTensor, TT_IntegerTensor]>;
def TT_Tensor : AnyTypeOf<[TT_FpIntTensor, TT_PtrTensor]>;

def TT_Type : AnyTypeOf<[TT_Float, TT_FloatTensor, TT_Int, TT_IntegerTensor, 
                         TT_Pointer, TT_PtrTensor]>;

#endif
