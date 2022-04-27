#ifndef Triton_OPS
#define Triton_OPS

include "TritonDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // NoSideEffect
include "mlir/Interfaces/ControlFlowInterfaces.td" // BranchOpInterface
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType

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

def TensorSizeTrait : NativeOpTrait<"TensorSizeTrait">;
//
// Op Base
//
class TT_Op<string mnemonic, list<Trait> traits = []> :
    Op<Triton_Dialect, mnemonic, !listconcat(traits, [TensorSizeTrait])>;

//
// CastOps
//
// Use cast ops in arith:
//   bitcast
//   fptoui, fptosi, uitofp, sitofp, 
//   extf, tructf,
//   extui, extsi, tructi
def TT_IntToPtrOp : TT_Op<"int_to_ptr", [SameOperandsAndResultShape, NoSideEffect,
                                         /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Cast int64 to pointer";

    let arguments = (ins I64Tensor:$from);

    let results = (outs TT_PtrTensor:$result);
}

def TT_PtrToIntOp : TT_Op<"ptr_to_int", [SameOperandsAndResultShape, NoSideEffect,
                                      /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Cast pointer to int64";

    let arguments = (ins TT_PtrTensor:$from);

    let results = (outs I64Tensor:$result);
}

def TT_FpToFp : TT_Op<"fp_to_fp", [SameOperandsAndResultShape, NoSideEffect,
                                /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Floating point casting for custom types";

    let description = [{
        Floating point casting for custom types (F8, BF8).

        F8 <-> BF8, FP16, FP32
        BF8 <-> F8, FP16, FP32
    }];

    let arguments = (ins TT_FloatTensor:$from);

    let results = (outs TT_FloatTensor:$result);

    // TODO: We need a verifier here.
}

//
// Load/Store Ops
//
def TT_CacheModifierAttr : I32EnumAttr<
    "CacheModifier", "",
    [
        I32EnumAttrCase<"NONE", 1, "none">,
        I32EnumAttrCase<"CA", 2, "ca">,
        I32EnumAttrCase<"CG", 3, "cg">,
    ]> {
    let cppNamespace = "::mlir::triton";
}
def TT_EvictionPolicyAttr : I32EnumAttr<
    "EvictionPolicy", "",
    [
        I32EnumAttrCase<"NORMAL", 1, "normal">,
        I32EnumAttrCase<"EVICT_FIRST", 2, "evict_first">,
        I32EnumAttrCase<"EVICT_LAST", 3, "evict_last">
    ]> {
    let cppNamespace = "::mlir::triton";
}

def TT_LoadOp : TT_Op<"load",
                      [SameOperandsAndResultShape,
                       MemoryEffects<[MemRead]>,
                       TypesMatchWith<"infer ptr type from result type",
                                      "result", "ptr",
                                      "getPointerTypeFromTensor($_self)">,
                       TypesMatchWith<"infer mask type from result type",
                                      "result", "mask",
                                      "getI1SameShape($_self)">,
                       TypesMatchWith<"infer other type from result type",
                                      "result", "other",
                                      "$_self">]> {
    let summary = "load";

    let arguments = (ins TT_PtrTensor:$ptr, BoolLike:$mask, TT_Type:$other,
                         TT_CacheModifierAttr:$cache, TT_EvictionPolicyAttr:$evict,
                         BoolAttr:$isVolatile);

    let results = (outs TT_Type:$result);

    let builders = [
        // for args with default values
        OpBuilder<(ins "Value":$ptr, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        OpBuilder<(ins "Value":$ptr, "Value":$mask, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>
    ];

    let assemblyFormat = "$ptr`,` $mask`,` $other attr-dict `:` type($result)";
}

def TT_StoreOp : TT_Op<"store",
                       [SameOperandsShape,
                        MemoryEffects<[MemWrite]>,
                        TypesMatchWith<"infer ptr type from value type",
                                       "value", "ptr",
                                       "getPointerTypeFromTensor($_self)">,
                        TypesMatchWith<"infer mask type from value type",
                                       "value", "mask",
                                       "getI1SameShape($_self)">]> {
    let summary = "store";

    let arguments = (ins TT_PtrTensor:$ptr, TT_Type:$value, BoolLike:$mask);

    let builders = [
        OpBuilder<(ins "Value":$ptr, "Value":$value)>,
    ];

    let assemblyFormat = "$ptr `,` $value `,` $mask `,` attr-dict `:` type($value)";
}

def TT_GEPOp : TT_Op<"getelementptr",
                     [NoSideEffect, SameOperandsAndResultShape,
                      TypesMatchWith<"result type matches ptr type",
                                     "result", "ptr", "$_self">,
                      TypesMatchWith<"result shape matches offset shape",
                                     "result", "offset",
                                     "getI32SameShape($_self)">]> {
    let arguments = (ins TT_PtrTensor:$ptr, I32Tensor:$offset);

    let results = (outs TT_PtrTensor:$result);

    let assemblyFormat = "$ptr `,` $offset `,` attr-dict `:` type($result)";
}


//
// Shape Manipulation Ops
//
def TT_ReshapeOp : TT_Op<"reshape", [NoSideEffect, SameOperandsAndResultElementType]> {
    let summary = "reshape";

    let arguments = (ins TT_Tensor:$src);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

def TT_BroadcastOp : TT_Op<"broadcast", [NoSideEffect, SameOperandsAndResultElementType]> {
    let summary = "broadcast";

    let arguments = (ins TT_Type:$src);

    let results = (outs TT_Type:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";

    let hasFolder = 1;
}

def TT_CatOp : TT_Op<"cat", [NoSideEffect, SameOperandsAndResultElementType]> {
    let summary = "concatenate 2 tensors";

    let arguments = (ins TT_Tensor:$lhs, TT_Tensor:$rhs);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

//
// builtin Ops
//
def TT_GetProgramIdOp : TT_Op<"get_program_id"> {
    let arguments = (ins I32Attr:$axis);

    let results = (outs I32:$result);

    let assemblyFormat = "attr-dict `:` type($result)";
}

def TT_GetNumProgramsOp : TT_Op<"get_num_programs"> {
    let arguments = (ins I32Attr:$axis);

    let results = (outs I32:$result);

    let assemblyFormat = "attr-dict `:` type($result)";
}

def TT_DotOp : TT_Op<"dot", [NoSideEffect,
                             TypesMatchWith<"result's type matches accumulator's type",
                                            "d", "c", "$_self">]> {
    let summary = "dot";

    let description = [{
        $d = matrix_multiply($a, $b) + $c
    }];

    let arguments = (ins TT_FpIntTensor:$a, TT_FpIntTensor:$b, TT_FpIntTensor:$c, BoolAttr:$allowTF32);

    let results = (outs TT_FpIntTensor:$d);

    let assemblyFormat = "$a`,` $b`,` $c attr-dict `:` type($a) `*` type($b) `->` type($d)";

    // let hasCanonicalizer = 1;
}

// reduction
def TT_RedOpAttr : I32EnumAttr<
    /*name*/"RedOp", /*summary*/"",
    /*case*/
    [
        I32EnumAttrCase</*sym*/"ADD", 1, /*str*/"add">,
        I32EnumAttrCase<"MAX", 2, "max">,
        I32EnumAttrCase<"MIN", 3, "min">,
        I32EnumAttrCase<"FADD", 4, "fadd">,
        I32EnumAttrCase<"FMAX", 5, "fmax">,
        I32EnumAttrCase<"FMIN", 6, "fmin">,
        I32EnumAttrCase<"XOR", 7, "xor">
    ]> {
    let cppNamespace = "::mlir::triton";
}

def TT_ReduceOp : TT_Op<"reduce"> {
    let summary = "reduce";

    let arguments = (ins TT_RedOpAttr:$reduce_op, TT_Type:$operand, I32Attr:$axis);
}

// atomic
def TT_AtomicRMWAttr : I32EnumAttr<
    "RMWOp", "",
    [
        I32EnumAttrCase<"AND", 1, "and">,
        I32EnumAttrCase<"OR", 2, "or">,
        I32EnumAttrCase<"XOR", 3, "xor">,
        I32EnumAttrCase<"ADD", 4, "add">,
        I32EnumAttrCase<"FADD", 5, "fadd">,
        I32EnumAttrCase<"MAX", 6, "max">,
        I32EnumAttrCase<"MIN", 7, "min">,
        I32EnumAttrCase<"UMAX", 8, "umax">,
        I32EnumAttrCase<"UMIN", 9, "umin">
    ]> {
    let cppNamespace = "::mlir::triton";
}

def TT_AtomicRMWOp : TT_Op<"atomic_rmw"> {
    let summary = "atomic rmw";

    let description = [{
        load data at $ptr, do $rmw_op with $val, and store result to $ptr.

        return old value at $ptr
    }];

    let arguments = (ins TT_AtomicRMWAttr:$atomic_rmw_op, TT_PtrTensor:$ptr,
                         TT_Type:$val, I1Tensor:$mask);

    let results = (outs TT_Type:$result);
}

def TT_AtomicCASOp : TT_Op<"atomic_cas"> {
    let summary = "atomic cas";

    let description = [{
        compare $cmp with data $old at location $ptr, 

        if $old == $cmp, store $val to $ptr,

        else store $old to $ptr,

        return $old
    }];

    let arguments = (ins TT_Pointer:$ptr, TT_Type:$cmp, TT_Type:$val);

    let results = (outs TT_Type:$result);
}

//
// Intrinsics
//
// TODO: should have ConstantLike as Trait
def TT_MakeRangeOp : TT_Op<"make_range", [NoSideEffect]> {
    let summary = "make range";

    let description = [{
        Returns an 1D int32 tensor.

        Values span from $start to $end (exclusive), with step = 1
    }];

    let arguments = (ins I32Attr:$start, I32Attr:$end);

    let results = (outs TT_IntegerTensor:$result);

    let assemblyFormat = "attr-dict `:` type($result)";
}

#endif // Triton_OPS
