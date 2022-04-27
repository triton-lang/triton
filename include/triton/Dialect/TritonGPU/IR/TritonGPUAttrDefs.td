#ifndef TRITONGPU_ATTRDEFS
#define TRITONGPU_ATTRDEFS

include "TritonGPUDialect.td"

class TritonGPU_Attr<string name, list<Trait> traits = []>
  : AttrDef<TritonGPU_Dialect, name, traits>;

def TritonGPUSharedEncodingAttr : TritonGPU_Attr<"TritonGPUSharedEncoding"> {
  let mnemonic = "shared (memory) encoding";

  let description = [{
    Example:

    ```mlir
    #SMEM = #triton_gpu.encoding<{
      vec = 8,
      perPhase = 8,
      maxPhase = 1
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

  let description = [{}];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$nanoTileSize,
    ArrayRefParameter<"unsigned">:$microTileSize,
    ArrayRefParameter<"unsigned">:$shapePerCTA,
    ArrayRefParameter<"unsigned">:$reptitions
  );

  // let genVerifyDecl = 1;
}

def TritonGPUMmaEncodingAttr : TritonGPU_Attr<"TritonGPUMmaEncoding"> {
  let mnemonic = "mma encoding";

  let description = [{}];

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
