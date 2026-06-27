//===----------------------------------------------------------------------===//
// Shared attr interface definitions for TritonGPU and TritonNvidiaGPU.
//===----------------------------------------------------------------------===//

#ifndef TRITONGPU_ATTRINTERFACES_TD
#define TRITONGPU_ATTRINTERFACES_TD

include "triton/Dialect/TritonGPU/IR/TritonGPUAttrBase.td"

def LayoutEncodingTrait : AttrInterface<"LayoutEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";
  let description = [{
    Common trait for all TTGIR layouts.
  }];
  let methods = [
    InterfaceMethod<"Get the CGA layout backing this encoding.",
                    "CGAEncodingAttr", "getCGALayout">,
    InterfaceMethod<"Get the rank of the layout.", "unsigned", "getRank",
                    (ins), [{}], [{
      return $_attr.getCGALayout().getRank();
    }]>
  ];
}
def DeclareLayoutEncodingMethods : DeclareAttrInterfaceMethods<
  LayoutEncodingTrait, ["getCGALayout"]>;

def SharedEncodingTrait : AttrInterface<"SharedEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";

  let description = [{
    Common trait describing shared memory.
  }];
  let methods = [
    InterfaceMethod<"Return the default alignment for the layout.",
                    "int32_t", "getAlignment", (ins), [{}], [{ return 16; }]>,
  ];
}
def DeclareSharedEncodingMethods : DeclareAttrInterfaceMethods<
  SharedEncodingTrait, ["getAlignment"]>;

def DistributedEncodingTrait : AttrInterface<"DistributedEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";

  let description = [{
The Distributed encoding describes the layout L with the 4-level compute hierarchy on GPU.
It is abstracted from the top to the bottom as CTAs Per CGA->Warps Per CTA->Threads Per Warp->Values Per Thread.

For CTAs Per CGA and Warps Per CTA level, the linear id is distributed contiguously with the shape and order.
For example, for a shape/order pair defines a distribution layout
shape = [4, 4]
order = [0, 1] // The fastest-changing axis first
->
layout = [0  4  8  12]
         [1  5  9  13]
         [2  6  10 14]
         [3  7  11 15]

For the Threads Per Warp and Values Per Thread level, the linear id distribution is variant for each sub-class encoding.

If the layout does not completely cover the tensor, we tile it until we cover the entire tensor.
We call each individual tile "rep".
  }];

  let methods = [
    InterfaceMethod<"Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first",
                    "SmallVector<unsigned>",
                    "getRepOrder">,
    InterfaceMethod<"Return total element size per thread.",
                    "unsigned",
                    "getTotalElemsPerThread",
                     (ins "ArrayRef<int64_t>":$shape),
                     /*defaultImplementation=*/[{
                         return toLinearEncoding($_self, shape).getTotalElemsPerThread(shape);
                     }]>,
    InterfaceMethod<"Return element size per thread in each dimension.",
                    "SmallVector<unsigned>",
                    "getElemsPerThread",
                     (ins "ArrayRef<int64_t>":$shape),
                     /*defaultImplementation=*/[{
                         return toLinearEncoding($_self, shape).getElemsPerThread(shape);
                     }]>,
    InterfaceMethod<"Convert to LinearLayout.",
                    "LinearLayout",
                    "toLinearLayout",
                    (ins "ArrayRef<int64_t>":$shape)>,
  ];
}

def LinearEncodingTrait : AttrInterface<"LinearEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";
  let description = [{
    Common trait for distributed encodings backed by a LinearLayout
    (LinearEncodingAttr and GenericLinearEncodingAttr).
  }];
  let methods = [
    InterfaceMethod<"Get the underlying linear layout.",
                    "const LinearLayout &", "getLinearLayout">,

    InterfaceMethod<[{
      Generalizes getThreadsPerWarp, getWarpsPerCTA, getCTAsPerCGA to linear layouts.
      Returns the bases of the dimensions `dimName` of the layout.
      If skipBroadcast is false, we count a base zero.
    }],
                    "SmallVector<unsigned>", "basesPerDim",
                    (ins "StringAttr":$dimName,
                         "bool":$skipBroadcast),
                    [{}], [{
      return LinearEncodingTrait::basesPerDim(
          $_attr.getLinearLayout(), dimName, skipBroadcast);
    }]>,
    InterfaceMethod<[{
      Generalizes get{Warp,Thread,CTA}Order to linear layouts.
      Returns the order of the dimensions `dimName` of the layout.
      If more than one dimension is of size one, it uses defaultOrder to determine the order
      of the dimensions of size one.
    }],
                    "SmallVector<unsigned>", "orderPerDim",
                    (ins "StringAttr":$dimName,
                         "ArrayRef<unsigned>":$defaultOrder),
                    [{}], [{
      return LinearEncodingTrait::orderPerDim(
          $_attr.getLinearLayout(), dimName, defaultOrder);
    }]>,
    InterfaceMethod<"Get the layout order.",
                    "SmallVector<unsigned>", "getOrder",
                    (ins), [{}], [{
      const auto &ll = $_attr.getLinearLayout();
      auto rank = ll.getNumOutDims();
      SmallVector<unsigned> order(rank);
      // Choose [rank-1, rank-2, ... 0] as the default order in case
      // there are dims that do not move in the register.
      // This order is as good as any.
      std::iota(order.rbegin(), order.rend(), 0);
      return $_attr.orderPerDim(
          StringAttr::get($_attr.getContext(), "register"), order);
    }]>,
    InterfaceMethod<"Get contiguity for a given input dimension.",
                    "SmallVector<unsigned>", "getContig",
                    (ins "const char *":$inDim,
                         "SmallVector<unsigned>":$lowerContig),
                    [{}], [{
      return LinearEncodingTrait::getContig(
          $_attr.getLinearLayout(), inDim, lowerContig, $_attr.getOrder());
    }]>,
    // [FIXME LL] Supports legacy behaviour. We should remove these functions.
    InterfaceMethod<"Get size per thread.",
                    "SmallVector<unsigned>", "getSizePerThread",
                    (ins), [{}], [{
      auto cgaSplitNum = $_attr.getCGALayout().getCTASplitNum();
      return LinearEncodingTrait::getSizePerThread(
          $_attr.getLinearLayout(), cgaSplitNum);
    }]>,
    InterfaceMethod<"Get the CGA layout.",
                    "CGAEncodingAttr", "getCGALayout",
                    (ins), [{}], [{
      return LinearEncodingTrait::getCGALayout($_attr.getLinearLayout());
    }]>,
    InterfaceMethod<"Convert to LinearLayout.",
                    "LinearLayout", "toLinearLayout",
                    (ins "ArrayRef<int64_t>":$shape),
                    [{}], [{
      auto repOrder = $_attr.getRepOrder();
      return LinearEncodingTrait::toLinearLayout(
          $_attr.getLinearLayout(), repOrder, shape);
    }]>,

    InterfaceMethod<"Get threads per warp in each dimension.",
                    "SmallVector<unsigned>", "getThreadsPerWarp",
                    (ins), [{}], [{
      return $_attr.basesPerDim(
          StringAttr::get($_attr.getContext(), "lane"), /*skipBroadcast=*/true);
    }]>,
    InterfaceMethod<"Get the thread order.",
                    "SmallVector<unsigned>", "getThreadOrder",
                    (ins), [{}], [{
      return $_attr.orderPerDim(
          StringAttr::get($_attr.getContext(), "lane"), $_attr.getOrder());
    }]>,
    InterfaceMethod<"Get contiguity per thread.",
                    "SmallVector<unsigned>", "getContigPerThread",
                    (ins), [{}], [{
      SmallVector<unsigned> c($_attr.getOrder().size(), 1);
      return $_attr.getContig("register", c);
    }]>,
    InterfaceMethod<"Get contiguity per warp.",
                    "SmallVector<unsigned>", "getContigPerWarp",
                    (ins), [{}], [{
      return $_attr.getContig("lane", $_attr.getContigPerThread());
    }]>,
    InterfaceMethod<"Get the rep order (also satisfies DistributedEncodingTrait).",
                    "SmallVector<unsigned>", "getRepOrder",
                    (ins), [{}], [{
      // This is not correct, but:
      // - It happens to agree in most places with the legacy layout
      // - getRepOrder does not make sense for LinearEncodingAttr as it already has
      //   the same shape as the tensor that uses it
      return $_attr.getOrder();
    }]>,
    InterfaceMethod<"Get elements per thread (overrides DistributedEncodingTrait default).",
                    "SmallVector<unsigned>", "getElemsPerThread",
                    (ins "ArrayRef<int64_t>":$shape),
                    [{}], [{
      return LinearEncodingTrait::getElemsPerThread(
          $_attr.getLinearLayout(), $_attr.getOrder(), shape);
    }]>,
    InterfaceMethod<"Get total elements per thread (overrides DistributedEncodingTrait default).",
                    "unsigned", "getTotalElemsPerThread",
                    (ins "ArrayRef<int64_t>":$shape),
                    [{}], [{
      return LinearEncodingTrait::getTotalElemsPerThread(
          $_attr.getLinearLayout(), $_attr.getOrder(), shape);
    }]>,
  ];
  let extraClassDeclaration = [{
    static SmallVector<unsigned> basesPerDim(const LinearLayout &ll,
                                             StringAttr dimName,
                                             bool skipBroadcast);
    static SmallVector<unsigned> orderPerDim(const LinearLayout &ll,
                                             StringAttr dimName,
                                             ArrayRef<unsigned> defaultOrder);
    static SmallVector<unsigned> getContig(const LinearLayout &ll,
                                           const char *inDim,
                                           SmallVector<unsigned> lowerContig,
                                           ArrayRef<unsigned> order);
    static SmallVector<unsigned> getSizePerThread(
        const LinearLayout &ll, ArrayRef<unsigned> cgaSplitNum);
    static CGAEncodingAttr getCGALayout(const LinearLayout &ll);
    static LinearLayout toLinearLayout(const LinearLayout &ll,
                                       ArrayRef<unsigned> repOrder,
                                       ArrayRef<int64_t> shape);
    static SmallVector<unsigned> getElemsPerThread(
        const LinearLayout &ll, ArrayRef<unsigned> repOrder,
        ArrayRef<int64_t> shape);
    static unsigned getTotalElemsPerThread(
        const LinearLayout &ll, ArrayRef<unsigned> repOrder,
        ArrayRef<int64_t> shape);
  }];
}
def DeclareLinearEncodingMethods : DeclareAttrInterfaceMethods<
  LinearEncodingTrait, ["getLinearLayout"]>;

def MmaEncodingTrait : AttrInterface<"MmaEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";
  let methods = [
    InterfaceMethod<"Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first",
                    "SmallVector<unsigned>",
                    "getRepOrderForOperand",
                    (ins "int":$opIdx)>,
  ];
}

#endif // TRITONGPU_ATTRINTERFACES_TD
