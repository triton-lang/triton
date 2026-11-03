//===----------------------------------------------------------------------===//
// CGA encoding attribute definition emitted early to break interface cycles.
//===----------------------------------------------------------------------===//

#ifndef TRITONGPU_CGAENCODING_ATTR_TD
#define TRITONGPU_CGAENCODING_ATTR_TD

include "triton/Dialect/TritonGPU/IR/TritonGPUAttrBase.td"

//===----------------------------------------------------------------------===//
// CGA Layout
//===----------------------------------------------------------------------===//

def CGAEncodingAttr : TritonGPU_Attr<"CGAEncoding", "cga_encoding"> {
  let parameters = (ins LinearLayoutParam:$linearLayout);

  let description = [{
Describes how blocks (CTAs) in a cooperative thread array (CGA) map onto logical
tensor dimensions. The `LinearLayout` maps from `block` into `dim0`, `dim1`...
  }];

  let extraClassDeclaration = [{
    // Map with empty bases and dims [dim0, dim1, ...]
    static CGAEncodingAttr get1CTALayout(MLIRContext *context, int rank);
    // Map with bases = [[1,], [2,], ..., [numCTAs/2]] into dim0
    static CGAEncodingAttr get1DLayout(MLIRContext *context, int numCTAs);
    // Legacy, we should kill this! Note that it is not true in general that
    // fromSplitParams(enc.getCTAsPerCGA(), enc.getCTASplitNum(), enc.getCTAOrder()) == enc!!
    static CGAEncodingAttr fromSplitParams(MLIRContext *context,
                                           ArrayRef<unsigned> CTAsPerCGA,
                                           ArrayRef<unsigned> CTASplitNum,
                                           ArrayRef<unsigned> CTAOrder);

    unsigned getRank() const { return getLinearLayout().getNumOutDims(); }
    SmallVector<unsigned> getCTAsPerCGA() const;
    SmallVector<unsigned> getCTASplitNum() const;
    SmallVector<unsigned> getCTAOrder() const;
  }];

  let genVerifyDecl = 1;
}

#endif // TRITONGPU_CGAENCODING_ATTR_TD
