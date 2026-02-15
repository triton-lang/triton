#ifndef GLUON_OPS
#define GLUON_OPS

include "triton/Dialect/Gluon/IR/GluonDialect.td"
include "triton/Dialect/Gluon/IR/GluonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

class Gluon_Op<string mnemonic, list<Trait> traits = []> :
    Op<Gluon_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

def Gluon_SetAutoLayoutOp : Gluon_Op<"set_auto_layout",
                                 [SameOperandsAndResultShape,
                                  SameOperandsAndResultElementType]> {
  let summary = "set auto encoding to a concrete encoding type";

  let arguments = (ins TT_Tensor:$src);

  let results = (outs TT_Tensor:$result);

  let builders = [
    OpBuilder<(ins "Attribute":$encoding, "Value":$value)>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

#endif // GLUON_OPS
