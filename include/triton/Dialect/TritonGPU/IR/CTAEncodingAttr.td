//===----------------------------------------------------------------------===//
// CTA encoding attribute definition emitted early to break interface cycles.
//===----------------------------------------------------------------------===//

#ifndef TRITONGPU_CTAENCODING_ATTR_TD
#define TRITONGPU_CTAENCODING_ATTR_TD

include "triton/Dialect/TritonGPU/IR/TritonGPUAttrBase.td"

//===----------------------------------------------------------------------===//
// CTA Layout
//===----------------------------------------------------------------------===//

def CTAEncodingAttr : TritonGPU_Attr<"CTAEncoding", "cta_encoding"> {
  let parameters = (ins LinearLayoutParam:$linearLayout);

  let description = [{
Describes how blocks (CTAs) in a cooperative thread array (CGA) map onto logical
tensor dimensions. The `LinearLayout` maps from `block` into `dim0`, `dim1`...
  }];

  let extraClassDeclaration = [{
    static CTAEncodingAttr getDefault(MLIRContext *context, int rank);
    // Legacy, we should kill this! Note that it is not true in general that
    // fromSplitParams(enc.getCTAsPerCGA(), enc.getCTASplitNum(), enc.getCTAOrder()) == enc!!
    static CTAEncodingAttr fromSplitParams(MLIRContext *context,
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

#endif // TRITONGPU_CTAENCODING_ATTR_TD
