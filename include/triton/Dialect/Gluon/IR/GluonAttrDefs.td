#ifndef GLUON_ATTRDEFS
#define GLUON_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/Gluon/IR/GluonDialect.td"

def Gluon_AutoEncodingAttr : AttrDef<Gluon_Dialect, "AutoEncoding"> {
  let mnemonic = "auto_encoding";
  let attrName = "gluon.auto_encoding";
  let description = [{
    An encoding that is inferred from neighboring ops in the graph.
  }];
}

#endif
