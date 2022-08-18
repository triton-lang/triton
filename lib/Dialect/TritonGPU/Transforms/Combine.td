#ifndef TRITONGPU_PATTERNS
#define TRITONGPU_PATTERNS

include "triton/Dialect/TritonGPU/IR/TritonGPUOps.td"
include "triton/Dialect/Triton/IR/TritonOps.td"

// convert_layout(load(...), #L) => copy_async(...); barrier 
//   if #L is smem_layout
def CopyAsyncOptPattern : Pat<
  (TTG_ConvertLayoutOp:$res (TT_LoadOp $ptr, $mask, $other, $cache, $evict, $isVolatile, $isOtherUnspecified)),
  (TTG_CopyAsyncOp $ptr, $mask, $other, $cache, $evict, $isVolatile, $isOtherUnspecified),
  [(Constraint<CPred<"isSharedLayout($0)">> $res)]>;

// ConvertLayout(ConvertLayout(x, #L0), #L1) => ConvertLayout(x, #L1)
def ConvertLayoutOptPattern : Pat<
  (TTG_ConvertLayoutOp (TTG_ConvertLayoutOp $x)),
  (TTG_ConvertLayoutOp $x)>;

// TODO: can we replace this with ConvertLayoutOp's folder?
// ConvertLayout(x, #L) => x if x.layout() == #L
def RedundantConvertLayoutOptPattern : Pat<
  (TTG_ConvertLayoutOp:$res $x), (replaceWithValue $x),
  [(Constraint<CPred<"$0.getType() == $1.getType()">> $res, $x)]>;

#endif
