#ifndef TRITON_PATTERNS
#define TRITON_PATTERNS

include "mlir/Dialect/StandardOps/IR/Ops.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "triton/Dialect/Triton/IR/TritonOps.td"


// AddIOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)
// AddFOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)

// AddIOp(d, DotOp(a, b, c)) and c==0 => DotOp(a, b, d)
// AddFOp(d, DotOp(a, b, c)) and c==0 => DotOp(a, b, d)
def CombineDotAddIPattern : Pat<
        (Arith_AddIOp $d, (TT_DotOp:$res $a, $b, $c, $allowTF32, $transA, $transB)),
        (TT_DotOp $a, $b, $d, $allowTF32, $transA, $transB),
        [(Constraint<CPred<"isZero($0)">> $c)]>;
def CombineDotAddFPattern : Pat<
        (Arith_AddFOp $d, (TT_DotOp:$res $a, $b, $c, $allowTF32, $transA, $transB)),
        (TT_DotOp $a, $b, $d, $allowTF32, $transA, $transB),
        [(Constraint<CPred<"isZero($0)">> $c)]>;

def CombineDotAddIRevPattern : Pat<
        (Arith_AddIOp (TT_DotOp:$res $a, $b, $c, $allowTF32, $transA, $transB), $d),
        (TT_DotOp $a, $b, $d, $allowTF32, $transA, $transB),
        [(Constraint<CPred<"isZero($0)">> $c)]>;
def CombineDotAddFRevPattern : Pat<
        (Arith_AddFOp (TT_DotOp:$res $a, $b, $c, $allowTF32, $transA, $transB), $d),
        (TT_DotOp $a, $b, $d, $allowTF32, $transA, $transB),
        [(Constraint<CPred<"isZero($0)">> $c)]>;


// addptr(addptr(%ptr, %idx0), %idx1) => addptr(%ptr, AddI(%idx0, %idx1))
//   Note: leave (sub %c0, %c0) canceling to ArithmeticDialect
//         (ref: ArithmeticCanonicalization.td)
def CombineAddPtrPattern : Pat<
        (TT_AddPtrOp (TT_AddPtrOp $ptr, $idx0), $idx1),
        (TT_AddPtrOp $ptr, (Arith_AddIOp $idx0, $idx1))>;

// broadcast(cst) => cst
def getConstantValue : NativeCodeCall<"getConstantValue($_builder, $0, $1)">;
def CombineBroadcastConstantPattern : Pat<
    (TT_BroadcastOp:$bcast_res (Arith_ConstantOp $value)),
    (Arith_ConstantOp (getConstantValue $value, $bcast_res)),
    [(Constraint<CPred<"isBroadcastConstantCombinable($0)">> $value)]>;

#endif
