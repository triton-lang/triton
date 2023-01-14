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
        (Arith_AddIOp $d, (TT_DotOp:$res $a, $b, $c, $allowTF32)),
        (TT_DotOp $a, $b, $d, $allowTF32),
        [(Constraint<CPred<"isZero($0)">> $c)]>;
def CombineDotAddFPattern : Pat<
        (Arith_AddFOp $d, (TT_DotOp:$res $a, $b, $c, $allowTF32)),
        (TT_DotOp $a, $b, $d, $allowTF32),
        [(Constraint<CPred<"isZero($0)">> $c)]>;

def CombineDotAddIRevPattern : Pat<
        (Arith_AddIOp (TT_DotOp:$res $a, $b, $c, $allowTF32), $d),
        (TT_DotOp $a, $b, $d, $allowTF32),
        [(Constraint<CPred<"isZero($0)">> $c)]>;
def CombineDotAddFRevPattern : Pat<
        (Arith_AddFOp (TT_DotOp:$res $a, $b, $c, $allowTF32), $d),
        (TT_DotOp $a, $b, $d, $allowTF32),
        [(Constraint<CPred<"isZero($0)">> $c)]>;

// TODO: this fails for addptr(addptr(ptr, i32), i64)
// Commented out until fixed
// addptr(addptr(%ptr, %idx0), %idx1) => addptr(%ptr, AddI(%idx0, %idx1))
//   Note: leave (sub %c0, %c0) canceling to ArithmeticDialect
//         (ref: ArithmeticCanonicalization.td)
// def CombineAddPtrPattern : Pat<
//         (TT_AddPtrOp (TT_AddPtrOp $ptr, $idx0), $idx1),
//         (TT_AddPtrOp $ptr, (Arith_AddIOp $idx0, $idx1))>;

// broadcast(cst) => cst
def getBroadcastConstantValue : NativeCodeCall<"getBroadcastConstantValue($0, $1)">;
def CombineBroadcastConstantPattern : Pat<
    (TT_BroadcastOp:$bcastRes (Arith_ConstantOp $value)),
    (Arith_ConstantOp (getBroadcastConstantValue $value, $bcastRes)),
    [(Constraint<CPred<"isBroadcastConstantCombinable($0)">> $value)]>;

// extf/truncf/extsi/extui(constant(cst)) => constant(cst)
def CombineExtFConstantPattern : Pat<
    (Arith_ExtFOp:$res (Arith_ConstantOp $value)),
    (Arith_ConstantOp (getBroadcastConstantValue $value, $res)),
    [(Constraint<CPred<"isBroadcastConstantCombinable($0)">> $value)]>;

def CombineTruncFConstantPattern : Pat<
    (Arith_TruncFOp:$res (Arith_ConstantOp $value)),
    (Arith_ConstantOp (getBroadcastConstantValue $value, $res)),
    [(Constraint<CPred<"isBroadcastConstantCombinable($0)">> $value)]>;
    
def CombineExtSIConstantPattern : Pat<
    (Arith_ExtSIOp:$res (Arith_ConstantOp $value)),
    (Arith_ConstantOp (getBroadcastConstantValue $value, $res)),
    [(Constraint<CPred<"isBroadcastConstantCombinable($0)">> $value)]>;

def CombineExtUIConstantPattern : Pat<
    (Arith_ExtUIOp:$res (Arith_ConstantOp $value)),
    (Arith_ConstantOp (getBroadcastConstantValue $value, $res)),
    [(Constraint<CPred<"isBroadcastConstantCombinable($0)">> $value)]>;

// add(cst, constant(0)) => constant(cst)
def getAddConstantValue : NativeCodeCall<"getAddConstantValue($0, $1)">;
def CombineAddIConstantPattern : Pat<
    (Arith_AddIOp $a, (Arith_ConstantOp $value)),
    (Arith_ConstantOp (getAddConstantValue $value, $a)),
    [(Constraint<CPred<"isAddConstantCombinable($0, $1)">> $value, $a)]>;

def CombineAddFConstantPattern : Pat<
    (Arith_AddFOp $a, (Arith_ConstantOp $value)),
    (Arith_ConstantOp (getAddConstantValue $value, $a)),
    [(Constraint<CPred<"isAddConstantCombinable($0, $1)">> $value, $a)]>;

def CombineAddIRevConstantPattern : Pat<
    (Arith_AddIOp (Arith_ConstantOp $value), $b),
    (Arith_ConstantOp (getAddConstantValue $value, $b)),
    [(Constraint<CPred<"isAddConstantCombinable($0, $1)">> $value, $b)]>;

def CombineAddFRevConstantPattern : Pat<
    (Arith_AddFOp (Arith_ConstantOp $value), $b),
    (Arith_ConstantOp (getAddConstantValue $value, $b)),
    [(Constraint<CPred<"isAddConstantCombinable($0, $1)">> $value, $b)]>;

#endif
