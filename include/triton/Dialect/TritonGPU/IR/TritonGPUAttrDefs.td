#ifndef TRITONGPU_ATTRDEFS
#define TRITONGPU_ATTRDEFS

include "TritonGPUDialect.td"

class TritonGPU_Attr<string name, list<Trait> traits = []>
  : AttrDef<TritonGPU_Dialect, name, traits>;

def TritonGPUSharedEncodingAttr : TritonGPU_Attr<"TritonGPUSharedEncoding"> {
  let mnemonic = "shared (memory) encoding";

  let description = [{
An encoding for tensors whose elements may be simultaneously accessed by different warps in the programs, via shared memory.

In order to avoid shared memory bank conflicts, elements may be stored in a swizzled layout.
For example, a swizzled row-major layout stores would store data as follows:

A_{0, 0}  A_{0, 1}  A_{0, 2}  A_{0, 3} ...   [phase 0] \ per_phase = 2
A_{1, 0}  A_{0, 1}  A_{1, 2}  A_{1, 3} ...   [phase 0] /

groups of vec=2 elements
are stored contiguously
_ _ _ _ /\_ _ _ _
A_{2, 2}  A_{2, 3}  A_{2, 0}  A_{2, 1} ...   [phase 1] \ per phase = 2
A_{3, 2}  A_{3, 3}  A_{3, 0}  A_{3, 1} ...   [phase 1] /


And the associated TritonGPU MLIR

```mlir
#SMEM = #triton_gpu.encoding<{
  vec = 2,
  perPhase = 2,
  maxPhase = 4
}>
```
  }];

  let parameters = (
    ins
    // swizzle info
    "unsigned":$vec, "unsigned":$perPhase, "unsigned":$maxPhase
  );
}

def TritonGPUCoalescedEncodingAttr : TritonGPU_Attr<"TritonGPUCoalescedEncoding"> {
  let mnemonic = "coalesced encoding";

  let description = [{
An encoding where each warp owns a contiguous portion of the target tensor. This is typically the kind of data layout
consumed (and returned) by LoadInst.
For example, a row-major coalesced layout may distribute a 64x16 tensor over 2 warps (i.e. 64 threads) as follows:

                          thread tile size 2
                        - - - - - - /\ - - - - - -
block|  thread      || A_{0,  0}[T0]   A_{0,  1}[T0]   ... A_{0,  6}[T3]  A_{0,  7}[T3]      A_{0,  8}[T0]  A_{0,  9}[T0]  ... A_{0,  14}[T3]  A_{0,  15}[T3]
tile |  tile size 2 || A_{1,  0}[T0]   A_{1,  1}[T0]   ... A_{1,  6}[T3]  A_{1,  7}[T3]      A_{1,  8}[T0]  A_{1,  9}[T0]  ... A_{1,  14}[T3]  A_{1,  15}[T3]
size }  ....
16   |                 A_{30, 0}[T60]  A_{14, 1}[T60]  ... A_{14, 6}[T63] A_{14, 7}[T63]     A_{14, 8}[T60] A_{14, 9}[T60] ... A_{14, 14}[T63] A_{14, 15}[T63]
     |                 A_{31, 0}[T60]  A_{15, 1}[T60]  ... A_{15, 6}[T63] A_{15, 7}[T63]     A_{15, 8}[T60] A_{15, 9}[T60] ... A_{15, 14}[T63] A_{15, 15}[T63]
                      -----------------------------/\-----------------------------------
                                            block tile size 8

      
                      A_{32, 0}[T0]  A_{32, 1}[T0]  ...  A_{32, 6}[T3]  A_{32, 7}[T3]        A_{32, 8}[T0]  A_{32, 9}[T0]  ... A_{32, 14}[T3]  A_{32, 15}[T3]
                      A_{33, 0}[T0]  A_{33, 1}[T0]  ...  A_{33, 6}[T3]  A_{33, 7}[T3]        A_{33, 8}[T0]  A_{33, 9}[T0]  ... A_{33, 14}[T3]  A_{33, 15}[T3]

                      A_{62, 0}[T60]  A_{62, 1}[T60]  ...  A_{62, 6}[T63] A_{62, 7}[T63]     A_{62, 8}[T60] A_{62, 9}[T60] ... A_{62, 14}[T63] A_{62, 15}[T63]
                      A_{63, 0}[T60]  A_{63, 1}[T60]  ...  A_{63, 6}[T63] A_{63, 7}[T63]     A_{63, 8}[T60] A_{63, 9}[T60] ... A_{63, 14}[T63] A_{63, 15}[T63]

And the associated TritonGPU MLIR
#SMEM = #triton_gpu.encoding<{
  threadTileSize = {2, 2}
  blockTileSize = {16, 8}
}>
 
// note to Da: In current Triton codebase, `nanoTileSize = threadTileSize`,  and `macro-tile size = blockTileSize / threadTileSize`
   probably clearer to have easier semantics (i.e., size of each tile owned by a thread or a block)
}];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$threadTileSize,
    ArrayRefParameter<"unsigned">:$blockTileSize,
  );

  // let genVerifyDecl = 1;
}

def TritonGPUMmaEncodingAttr : TritonGPU_Attr<"TritonGPUMmaEncoding"> {
  let mnemonic = "mma encoding";

  let description = [{TODO: I think we may be able to implement it as a special-case of Coalesced encoding with maybe one more warpTileSize attribute!}];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$fragmentPerWarp,
    ArrayRefParameter<"unsigned">:$shapePerWarp,
    ArrayRefParameter<"unsigned">:$warpPerTile,
    ArrayRefParameter<"unsigned">:$shapePerTile,
    ArrayRefParameter<"unsigned">:$reptitions,
    ArrayRefParameter<"unsigned">:$contigPerThread
  );

  // let genVerifyDecl = 1;
}

#endif
