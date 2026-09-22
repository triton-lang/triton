#ifndef TRITONNVIDIAGPU_ENUMS
#define TRITONNVIDIAGPU_ENUMS

include "mlir/IR/EnumAttr.td"

def TTNG_CacheEvictionPriorityAttr : I32EnumAttr<
    "CacheEvictionPriority", "NVIDIA cache eviction priority",
    [
      I32EnumAttrCase<"NONE", 0, "none">,
      I32EnumAttrCase<"EVICT_NORMAL", 1, "evict_normal">,
      I32EnumAttrCase<"EVICT_UNCHANGED", 2, "evict_unchanged">,
      I32EnumAttrCase<"EVICT_FIRST", 3, "evict_first">,
      I32EnumAttrCase<"EVICT_LAST", 4, "evict_last">,
      I32EnumAttrCase<"NO_ALLOCATE", 5, "no_allocate">
    ]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
  let genSpecializedAttr = 0;
}

def TTNG_TMEMWaitKindAttr : I32EnumAttr<
    "TMEMWaitKind", "tensor memory wait kind",
    [
        I32EnumAttrCase<"LOAD", 0, "load">,
        I32EnumAttrCase<"STORE", 1, "store">,
    ]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
}

def TTNG_TMEMLoadReduceModifierAttr : I32EnumAttr<
    "TMEMLoadReduceModifier", "",
    [
        I32EnumAttrCase<"MIN", 1, "min">,
        I32EnumAttrCase<"MAX", 2, "max">,
    ]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
  let genSpecializedAttr = 0;
}

def TTNG_PackedArithOpKindAttr : I32EnumAttr<
    "PackedArithOpKind", "packed arithmetic operation",
    [
        I32EnumAttrCase<"ADD", 0, "add">,
        I32EnumAttrCase<"SUB", 1, "sub">,
        I32EnumAttrCase<"MUL", 2, "mul">,
        I32EnumAttrCase<"FMA", 3, "fma">,
        I32EnumAttrCase<"MIN", 4, "min">,
        I32EnumAttrCase<"MAX", 5, "max">,
    ]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
}

def TTNG_TensorMemoryScalesBlockRepOrderAttr : I32EnumAttr<
    "TensorMemoryScalesBlockRepOrder", "",
    [
        I32EnumAttrCase<"MN_THEN_K", 0, "mnThenK">,
        I32EnumAttrCase<"K_THEN_MN", 1, "kThenMn">,
    ]> {
  let cppNamespace = "::mlir::triton::nvidia_gpu";
  let genSpecializedAttr = 0;
}

#endif // TRITONNVIDIAGPU_ENUMS
