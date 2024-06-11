#ifndef TRITONGPU_ATTRDEFS
#define TRITONGPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"

//===----------------------------------------------------------------------===//
// TritonGPU Attribute Definitions
//===----------------------------------------------------------------------===//
def TritonGPU_AttrTrait : AttrInterface<"TritonGPU_AttrTrait"> {
  let cppNamespace = "::mlir::triton::gpu";

  let methods = [
    InterfaceMethod<"Return total element size per thread.",
                    "unsigned",
                    "getTotalElemsPerThread",
                     (ins "ArrayRef<int64_t>":$tensorShape,
                          "Type":$eltTy)>,

    InterfaceMethod<"Return element size per thread in each dimension.",
                    "SmallVector<unsigned>",
                    "getElemsPerThread",
                     (ins "ArrayRef<int64_t>":$tensorShape,
                          "Type":$eltTy)>,
  ];
}

class TritonGPU_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                     Dialect dialect = TritonGPU_Dialect,
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<dialect, name, !listconcat([TritonGPU_AttrTrait], traits), baseCppClass> {

  let description = [{
TritonGPU tensors differ from usual tensors in that they contain a _layout_ attribute which determines
how the data should be partitioned across CUDA threads. Formally speaking, we define a layout as a function
\mathcal{L} that maps a multi-dimensional tensor index $i \in \mathbb{Z}^d$ to a set of integers T corresponding
to the indices of the CUDA threads allowed to access some data at index $i$.

For example, let us consider the layout function:
\mathcal{L}(0, 0) = {0, 4}
\mathcal{L}(0, 1) = {1, 5}
\mathcal{L}(1, 0) = {2, 6}
\mathcal{L}(1, 1) = {3, 7}

Then, attaching $\mathcal{L} to a tensor $T$ would mean that:
- T[0,0] is owned by both cuda thread 0 and 4
- T[0,1] is owned by both cuda thread 1 and 5
- T[1,0] is owned by both cuda thread 2 and 6
- T[1,1] is owned by both cuda thread 3 and 7

Right now, Triton implements two main classes of layouts: shared, and distributed.
  }];
  let attrName = "triton.gpu." # attrMnemonic;

  code extraBaseClassDeclaration = [{
    unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const;
    SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const;
    ::mlir::LogicalResult verifyLayoutForArg(::mlir::Operation* op, unsigned argNo) const;
  }];
}

//===----------------------------------------------------------------------===//
// CTA Layout
//===----------------------------------------------------------------------===//

def CTALayoutAttr : TritonGPU_Attr<"CTALayout", "cta_layout"> {
  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$CTAsPerCGA,
    ArrayRefParameter<"unsigned">:$CTASplitNum,
    ArrayRefParameter<"unsigned">:$CTAOrder
  );

  let description = [{
Describes how blocks are distributed among the cooperate thread arrays (aka
CTAs, aka thread blocks) in a cooperate thread group (aka CTG, aka thread group
cluster).  CGAs were introduced in Hopper (sm90).

The tensor is divided up into CTASplitNum pieces, which are distributed among
the CTAsPerCGA thread blocks.  Each CTA processes a subtensor of shape
`tensor_shape / CTASplitNum`.

Example 0: The tensor shape is [64, 128] and, there are two CTAs, each
processing half the tensor [64, 64]. Then CTAsPerCGA = [1, 2] and
CTASplitNum = [1, 2].

Example 1: The tensor shape is [64, 128] and, there are two CTAs, both
processing the complete tensor [64, 128]. This happens when multicast is
enabled. In this case, CTAsPerCTA = [1, 2] but CTASplitNum = [1, 1].

Example 2: Consider a matmul AxB=C, where A=[M,K], B=[K,N], C=[M,N].  The
CTAsPerCGA for A, B, C are the same, [SplitM, SplitN], but the CTASplitNum are
different. CTASplitNum_A = [SplitM, 1], which means multicast on dim1,
CTASplitNum_B = [1, SplitN], which means multicast on dim0, CTASplitNum_C =
[SplitM, SplitN]  which means no multicast.

Currently programs with multiple CTAs per CGA are an experimental feature in
Triton, not enabled by default.

You can leave off the CTALayout properties in the textual IR and Triton will
fill in the "default" CTALayout of CTAsPerCGA = CTASplitNum = [1...1].  In
addition, if there's only one CTA per CGA, then Triton canonicalizes CTAOrder to
[n-1,...,0] (it doesn't matter in this case).
  }];

  // CTALayout::get canonicalizes CTAOrder to [n,n-1,...,0] if CTAsPerCGA is
  // [1...1].  The CTAOrder doesn't matter in this case.
  //
  // This is a little weird because if you write textual IR with a one order and
  // then print it back out, you might get a different order.  But it seems this
  // is the best way to canonicalize an attribute in MLIR.
  let builders = [
    AttrBuilder<(ins "ArrayRef<unsigned>":$CTAsPerCGA,
                     "ArrayRef<unsigned>":$CTASplitNum,
                     "ArrayRef<unsigned>":$CTAOrder), [{
        if (llvm::all_of(CTAsPerCGA, [](unsigned x) { return x == 1; })) {
          SmallVector<unsigned> order;
          for (int i = CTAsPerCGA.size() - 1; i >= 0; --i)
            order.push_back(i);
          return $_get(context, CTAsPerCGA, CTASplitNum, order);
        }
        return $_get(context, CTAsPerCGA, CTASplitNum, CTAOrder);
    }]>,
  ];

  let extraClassDeclaration = [{
    SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const {
      llvm::report_fatal_error(
        "Unsupported getElemsPerThread in CTALayoutAttr.");
    }
    unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const {
      llvm::report_fatal_error(
        "Unsupported getTotalElemsPerThread in CTALayoutAttr.");
    }

    static CTALayoutAttr getDefault(MLIRContext *context, int rank) {
      SmallVector<unsigned> CTAsPerCGA(rank, 1);
      SmallVector<unsigned> CTASplitNum(rank, 1);
      SmallVector<unsigned> CTAOrder;
      for (int i = rank - 1; i >= 0; --i)
        CTAOrder.push_back(i);
      return get(context, CTAsPerCGA, CTASplitNum, CTAOrder);
    }
  }];

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// Shared Layout Encoding
//===----------------------------------------------------------------------===//

def SharedEncodingAttr : TritonGPU_Attr<"SharedEncoding", "shared_encoding"> {
  let mnemonic = "shared";

  let description = [{
An encoding for tensors whose elements may be simultaneously accessed by
different cuda threads in the programs, via shared memory. In other words,
for all indices i \in Z^d, \mathcal{L}(i) = {0, 1, ..., 32*num_warps - 1}.

In order to avoid shared memory bank conflicts, elements may be swizzled.
Here are some examples.  In all cases, the input tensor is [0, 1, ..., n-1].

1. Basic swizzling

  #shared<{vec=1, perPhase=1, maxPhase=4, order=[1,0]}>
  [ 0,  1,  2,  3],  // xor with 0
  [ 5,  4,  7,  6],  // xor with 1
  [10, 11,  8,  9],  // xor with 2
  [15, 14, 13, 12]   // xor with 3

Here elements of row r are xor'ed with r (or more properly, in[r][c] ->
out[r][c^r]).

2. Multiple rows per phase

  #shared<{vec=1, perPhase=2, maxPhase=4, order=[1,0]}>
  [ 0,  1,  2,  3],  // phase 0 (xor with 0)
  [ 4,  5,  6,  7],
  [ 9,  8, 11, 10],  // phase 1 (xor with 1)
  [13, 12, 15, 14]

Elements of row r are xor'ed with r/2.  In other words, perPhase=2
means that pairs of 2 rows get the same swizzling.

3. Max-phase applied

  $shared<{vec=1, perPhase=1, maxPhase=2, order=[1,0]}>
  [ 0,  1,  2,  3],  // phase 0 (xor with 0)
  [ 5,  4,  7,  6],  // phase 1 (xor with 1)
  [ 8,  9, 10, 11],  // phase 0
  [13, 12, 15, 14],  // phase 1
  [16, 17, 18, 19],  // ...
  [21, 20, 23, 22],
  [24, 25, 26, 27],
  [29, 28, 31, 30]

Elements of row r are xor'ed with (r/2) % 2.  In other words, maxPhase=m has the
effect of limiting the maximum value of the xor to m-1.

4. Max-phase and per-phase

  #shared<{vec=1, perPhase=2, maxPhase=2, order=[1,0]}>
  [ 0,  1,  2,  3],  // phase 0 (xor with 0)
  [ 4,  5,  6,  7],  // phase 0
  [ 9,  8, 11, 10],  // phase 1 (xor with 1)
  [13, 12, 15, 14],  // phase 1
  [16, 17, 18, 19],  // phase 0
  [20, 21, 22, 23],  // phase 0
  [25, 24, 27, 26],  // phase 1
  [29, 28, 31, 30]]  // phase 1

Here the xor value (the "phase", I guess?) changes every perPhase rows, up to a
maximum value of maxPhase-1.  In other words, elements of row r are xor'ed with
(r/2) % 2.

5. Adding vec

  #shared<{vec=2, perPhase=1, maxPhase=4, order=[1,0]}>
  [ 0,  1,  2,  3,  4,  5,  6,  7],
  [10, 11,  8,  9, 14, 15, 12, 13],
  [20, 21, 22, 23, 16, 17, 18, 19],
  [30, 31, 28, 29, 26, 27, 24, 25]

When vec=2, elements are swizzled in pairs of 2.  In other words, the element at
(r,c) has value

  ((c / 2) ^ r) * 2 + (c % 2).

For MMAv3 eg Hopper GMMA, hasLeadingOffset should be true. In this case,
when the matrix is stored in shared memory, there will be an offset not
only in the stride dimension, but also in the leading dimension. For example,
a matrix of size 16x128 and data type I8 is stored in the shared memory with
64B-swizzle mode. The offset of the element with index (0, 64) will be 16*64,
compared to 1*64 when the hasLeadingOffset is false.
  }];

  // swizzle info: vec, perPhase, maxPhase
  // order: the fastest-changing axis first
  let parameters = (
    ins
    "unsigned":$vec,
    "unsigned":$perPhase,
    "unsigned":$maxPhase,
    ArrayRefParameter<"unsigned">:$order,
    "CTALayoutAttr":$CTALayout,
    "bool":$hasLeadingOffset
  );

  let builders = [
    AttrBuilder<(ins "unsigned":$vec,
                     "unsigned":$perPhase,
                     "unsigned":$maxPhase,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout), [{
        bool hasLeadingOffset = false; // default value
        return $_get(context, vec, perPhase, maxPhase, order, CTALayout, hasLeadingOffset);
    }]>,

    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout,
                     "unsigned":$typeWidthInBit), [{
        bool needTrans = false; // default value
        return get(context, dotOpEnc, shape, order, CTALayout, typeWidthInBit, needTrans);
    }]>,

    // TODO(jlebar): This should not be an overload of
    // SharedEncodingAttr::get().  It's misleading, because it does a bunch of
    // nontrivial work based on the given dotOpEnc.
    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout,
                     "unsigned":$typeWidthInBit,
                     "bool":$needTrans), [{

        // ---- begin GFX908/GFX90A ----
        if (auto mfmaEnc = mlir::dyn_cast<AMDMfmaEncodingAttr>(dotOpEnc.getParent())) {
          int kDimNum = dotOpEnc.getOpIdx() == 0 ? 1 : 0;
          if (needTrans)
            kDimNum = 1 - kDimNum;
          bool isKDimInner = (order[0] == kDimNum);
          if (isKDimInner) {
            const int numBanks = 32;
            const int bankBitWidth = 32;
            const int SIMDWidth = 16;

            // number of inner dimension rows per one pattern repeat
            int innerDimLength = shape[order[0]];
            int elemsPerOneBanksRow = (numBanks * bankBitWidth) / typeWidthInBit;

            int perPhase = std::max(1, elemsPerOneBanksRow / innerDimLength);
            // vecSize is set to kWidth of the dotop layout
            int vecSize = dotOpEnc.getKWidth();
            int maxPhase = std::min(SIMDWidth / perPhase, innerDimLength / vecSize);

            // TODO (zhanglx): figure out better parameters for mfma4
            if (mfmaEnc.getMDim() == 4)
              maxPhase = 4;

            return get(context, vecSize, perPhase, maxPhase, order, CTALayout);
          } else {
            // Do not swizzle in case k dimension is not innermost.
            // In this case accesses will go in different banks even without swizzling.
            return get(context, 1, 1, 1, order, CTALayout);
          }
        }

        // ---- begin GFX11 ----
        if (mlir::isa<AMDWmmaEncodingAttr>(dotOpEnc.getParent())) {
          if (dotOpEnc.getOpIdx() == 0) {
            const int numBanks = 32;
            const int bankBitWidth = 32;

            // number of inner dimension rows per one pattern repeat
            int innerDimLength = shape[order[0]];
            int elemsPerOneBanksRow = (numBanks * bankBitWidth) / typeWidthInBit;

            int perPhase = std::max(1, elemsPerOneBanksRow / innerDimLength);
            int vecSize = ((typeWidthInBit == 16) ? 64 : 32 ) / typeWidthInBit;
            int maxPhase = 16 / perPhase;

            return get(context, vecSize, perPhase, maxPhase, order, CTALayout);
          } else {
            // Do not swizzle in case k dimension is not innermost.
            // In this case accesses will go in different banks even without swizzling.
            return get(context, 1, 1, 1, order, CTALayout);
          }
        }


        auto mmaEnc = mlir::dyn_cast<NvidiaMmaEncodingAttr>(dotOpEnc.getParent());

        if(!mmaEnc)
          return get(context, 1, 1, 1, order, CTALayout);

        int opIdx = dotOpEnc.getOpIdx();
        auto shapePerCTA = getShapePerCTA(CTALayout.getCTASplitNum(), shape);

        // number of rows per phase

        // index of the inner dimension in `order`
        unsigned inner = (opIdx == 0) ? 0 : 1;

        // ---- begin Volta ----
        if (mmaEnc.isVolta()) {
          int perPhase = 128 / (shapePerCTA[order[0]] * (typeWidthInBit / 8));
          perPhase = std::max<int>(perPhase, 1);
          bool is_row = order[0] != 0;
          bool is_vec4 = opIdx == 0 ? !is_row && (shapePerCTA[order[0]] <= 16) :
              is_row && (shapePerCTA[order[0]] <= 16);
          int pack_size = opIdx == 0 ? ((is_row || is_vec4) ? 1 : 2) :
                                       ((is_row && !is_vec4) ? 2 : 1);
          int rep = 2 * pack_size;
          int maxPhase = (order[inner] == 1 ? 8 : 4) / perPhase;
          int vec = 2 * rep;
          return get(context, vec, perPhase, maxPhase, order, CTALayout);
        }

        // ---- begin Ampere ----
        if (mmaEnc.isAmpere()) {
          int perPhase = 128 / (shapePerCTA[order[0]] * 4 / dotOpEnc.getKWidth());
          perPhase = std::max<int>(perPhase, 1);
          std::vector<size_t> matShape = {8, 8, 4 * dotOpEnc.getKWidth()};
          int vecWidth = 32 / typeWidthInBit;
          if (vecWidth != dotOpEnc.getKWidth() && order[0] == inner) {
              perPhase = std::max<int>(perPhase, 2 * vecWidth);
          }
          int rank = order.size();
          // --- handle A operand ---
          if (opIdx == 0) { // compute swizzling for A operand
              int m = (needTrans) ? matShape[2] : matShape[0];
              int k = (needTrans) ? matShape[0] : matShape[2];
              int vec = (order[0] == rank-1) ? k : m;
              int mmaStride = (order[0] == rank-1) ? m : k;
              int maxPhase = mmaStride / perPhase;
              return get(context, vec, perPhase, maxPhase, order, CTALayout);
          }

          // --- handle B operand ---
          if (opIdx == 1) {
              // we compute vec and maxPhase m, n and k size of the mma
              // instruction. when matmul operands is transposed, we should
              // consider that to get m, n and k.
              int n = needTrans ? matShape[2] : matShape[1];
              int k = needTrans ? matShape[1] : matShape[2];
              int vec = (order[0] == rank-1) ? n : k;
              int mmaStride = (order[0] == rank-1) ? k : n;
              int maxPhase = mmaStride / perPhase;
              return get(context, vec, perPhase, maxPhase, order, CTALayout);
          }

          llvm_unreachable("invalid operand index");
        }

        // ---- begin version 3 ----
        if (mmaEnc.isHopper()) {
          llvm_unreachable("SharedEncodingAttr builder when the MMAEncodingAttr"
                           " is Hopper has not been implemented yet");
          return $_get(context, 1, 1, 1, order, CTALayout, true);
        }

        // ---- not implemented ----
        llvm_unreachable("unsupported swizzling for provided MMA version");
    }]>,

    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout,
                     "Type":$eltTy), [{
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      return get(context, dotOpEnc, shape, order, CTALayout, bitwidth);
    }]>,

    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout,
                     "Type":$eltTy,
                     "bool":$needTrans), [{
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      return get(context, dotOpEnc, shape, order, CTALayout, bitwidth, needTrans);
    }]>,

    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout,
                     "Type":$eltTy), [{
        auto shapePerCTA = getShapePerCTA(CTALayout.getCTASplitNum(), shape);

        int32_t eleBitWidth = eltTy.getIntOrFloatBitWidth();
        int32_t vec = 128 / eleBitWidth, perPhase = 1, maxPhase = 1;

        // get proper shared memory swizzling mode from the contiguous dimension
        // size of the origin blocked layout.
        auto contigDimSizeInByte = shapePerCTA[order[0]] * eleBitWidth / 8;
        if (contigDimSizeInByte >= 128 && contigDimSizeInByte % 128 == 0) {
          perPhase = 1;
          maxPhase = 8;
        } else if (contigDimSizeInByte >= 64 && contigDimSizeInByte % 64 == 0) {
          perPhase = 2;
          maxPhase = 4;
        } else if (contigDimSizeInByte >= 32 && contigDimSizeInByte % 32 == 0) {
          perPhase = 4;
          maxPhase = 2;
        } else {
          llvm_unreachable("unsupported shared memory layout for MMAv3");
        }

        return $_get(context, vec, perPhase, maxPhase, order, CTALayout, true);
    }]>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Distributed Layout Encoding
//===----------------------------------------------------------------------===//
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
  }];

  let methods = [
    // Interface for the meta information about the multiple thread hierarchy.
    InterfaceMethod<"Get the shape of the CTAs per CGA.",
                    "SmallVector<unsigned>",
                    "getCTAsPerCGA">,

    InterfaceMethod<"Get the order of the CTAs per CGA. The fastest-changing axis first",
                    "SmallVector<unsigned>",
                    "getCTAOrder">,

    InterfaceMethod<"Get the shape of the warps per CTA.",
                    "SmallVector<unsigned>",
                    "getWarpsPerCTA">,

    InterfaceMethod<"Get the order of the warps per CTA. The fastest-changing axis first",
                    "SmallVector<unsigned>",
                    "getWarpOrder">,

    InterfaceMethod<"Get the shape of the threads per warp",
                    "SmallVector<unsigned>",
                    "getThreadsPerWarp">,

    InterfaceMethod<"Get the order of the threads per warp. The fastest-changing axis first",
                    "SmallVector<unsigned>",
                    "getThreadOrder">,

    InterfaceMethod<"Get the shape of the values per thread.",
                    "SmallVector<unsigned>",
                    "getSizePerThread">,

    InterfaceMethod<"Each CTA processes 1/CTASplitNum of the tensor.",
                    "SmallVector<unsigned>",
                    "getCTASplitNum">,

    InterfaceMethod<"Gets the shape of the encoding's tile, e.g. sizePerThread * threadsPerWarp * warpsPerCTA",
                    "SmallVector<unsigned>",
                    "getShapePerCTATile",
                     (ins "ArrayRef<int64_t>":$tensorShape)>,

    InterfaceMethod<"Gets the number of contiguous elements per thread.",
                    "SmallVector<unsigned>",
                    "getContigPerThread">,
  ];
}

class DistributedEncoding<string name, string attrMnemonic, list<Trait> traits = [],
                     Dialect dialect = TritonGPU_Dialect>
  : TritonGPU_Attr<name, attrMnemonic, !listconcat([DistributedEncodingTrait], traits), dialect> {

  let description = [{
Distributed encodings have a layout function L that is entirely characterized
by a d-dimensional tensor T. Note that L doesn't need to have the same shape
(or even the same rank) as the tensor it is encoding.

The layout function \mathcal{L} of this layout is then defined, for an
index `i` \in Z^d, as follows:

\mathcal{L}(T)[i_d] = L[(i_d + k_d*T.shape[d]) % L.shape[d]] \forall k_d such as i_d + k_d*T.shape[d] < L.shape[d]

Intuitively, when the tensor dim size T.shape[d] is larger than the layout
dim size L.shape[d], on that particular dim, we distribute values from the
tensor to threads mapped in the layout in a "wrapped around" manner, with
each thread owning multiple values.

OTOH, when the tensor dim size T.shape[d] is smaller than the layout
dim size L.shape[d], on that particular dim, we distribute values from the
tensor to threads mapped in the layout in a "broadcasted" manner, with
each value owned by multiple threads.

For example, for a tensor/layout pair
T = [x  x  x  x  x  x  x  x]
    [x  x  x  x  x  x  x  x]
L = [0  1  2  3 ]
    [4  5  6  7 ]
    [8  9  10 11]
    [12 13 14 15]

Then the data of T would be distributed as follow between the 16 CUDA threads:
L(T) = [ {0,8} , {1,9} , {2,10}, {3,11}, {0,8} , {1, 9} , {2, 10}, {3, 11},
         {4,12}, {5,13}, {6,14}, {7,15}, {4,12}, {5, 13}, {6, 14}, {7, 15} ]
  }];

  code extraDistributedDeclaration  = extraBaseClassDeclaration # [{
    SmallVector<unsigned> getCTAsPerCGA() const;
    SmallVector<unsigned> getCTAOrder() const;
    SmallVector<unsigned> getCTASplitNum() const;
    SmallVector<unsigned> getWarpsPerCTA() const;
    SmallVector<unsigned> getWarpOrder() const;
    SmallVector<unsigned> getThreadsPerWarp() const;
    SmallVector<unsigned> getThreadOrder() const;

    SmallVector<unsigned> getSizePerThread() const;
    SmallVector<unsigned> getShapePerCTATile(ArrayRef<int64_t> tensorShape = ArrayRef<int64_t>()) const;
  }];
}

//===----------------------------------------------------------------------===//
// Blocked Layout Encoding
//===----------------------------------------------------------------------===//

def BlockedEncodingAttr : DistributedEncoding<"BlockedEncoding", "blocked_encoding"> {
  let mnemonic = "blocked";

  let description = [{
An encoding where each warp owns a contiguous portion of the target tensor. This is typically the kind of data layout
used to promote memory coalescing in LoadInst and StoreInst.
It is characterized by three tuples -- thread tile size, warp tile size, and block tile size -- which
specify the amount of elements owned by each CUDA thread, warp and CTA respectively.

Example 1, a row-major coalesced layout may partition a 16x16 tensor over 2 warps (i.e. 64 threads) as follows:

[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]

for

#triton_gpu.blocked_layout<{
  sizePerThread = {2, 2}
  threadsPerWarp = {8, 4}
  warpsPerCTA = {1, 2}
  CTAsPerCGA = {1, 1}
  CTASplitNum = {1, 1}
}>

Example 2, a row-major coalesced layout may partition a 32x32 tensor over 2 warps (i.e. 64 threads) as follows:

[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                 ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                 ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
for

#triton_gpu.blocked_layout<{
  sizePerThread = {2, 2}
  threadsPerWarp = {8, 4}
  warpsPerCTA = {1, 2}
  CTAsPerCGA = {1, 1}
  CTASplitNum = {1, 1}
}>

Example 3, A row-major coalesced layout may partition a 32x32 tensor over 2 warps (i.e. 64 threads) and
4 CTAs (taking 2x2 for example) as follows:

CTA [0,0]                                              CTA [0,1]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                    ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]

CTA [1,0]                                              CTA [1,1]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                    ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
for

#triton_gpu.blocked_layout<{
  sizePerThread = {2, 2}
  threadsPerWarp = {8, 4}
  warpsPerCTA = {1, 2}
  CTAsPerCGA = {2, 2}
  CTASplitNum = {2, 2}
}>
}];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$sizePerThread__,
    ArrayRefParameter<"unsigned">:$threadsPerWarp__,
    ArrayRefParameter<"unsigned">:$warpsPerCTA__,
    ArrayRefParameter<"unsigned">:$order, // the fastest-changing axis first

    // CTALayout is optional in the textual IR.  If omitted, we infer it to be a
    // single CTA (so CTAsPerCGA = [1,...,1], CTASplitNum = [1,...,1],
    // CTAOrder=[n,n-1,...,0]).
    "CTALayoutAttr":$CTALayout
  );
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$sizePerThread,
                     "ArrayRef<unsigned>":$order,
                     "unsigned":$numWarps,
                     "unsigned":$numThreadsPerWarp,
                     "CTALayoutAttr":$CTALayout), [{
      unsigned rank = sizePerThread.size();
      SmallVector<unsigned, 4> threadsPerWarp(rank);
      SmallVector<unsigned, 4> warpsPerCTA(rank);
      SmallVector<int64_t> shapePerCTA = getShapePerCTA(CTALayout.getCTASplitNum(), shape);

      unsigned remainingLanes = numThreadsPerWarp;
      unsigned remainingThreads = numWarps * numThreadsPerWarp;
      unsigned remainingWarps = numWarps;
      unsigned prevLanes = 1;
      unsigned prevWarps = 1;

      // starting from the contiguous dimension
      for (unsigned d = 0; d < rank - 1; ++d) {
        unsigned i = order[d];
        unsigned threadsPerCTA = std::clamp<unsigned>(remainingThreads, 1, shapePerCTA[i] / sizePerThread[i]);
        threadsPerWarp[i] = std::clamp<unsigned>(threadsPerCTA, 1, remainingLanes);
        warpsPerCTA[i] = std::clamp<unsigned>(threadsPerCTA / threadsPerWarp[i], 1, remainingWarps);
        remainingWarps /= warpsPerCTA[i];
        remainingLanes /= threadsPerWarp[i];
        remainingThreads /= threadsPerCTA;
        prevLanes *= threadsPerWarp[i];
        prevWarps *= warpsPerCTA[i];
      }

      // Expand the last dimension to fill the remaining lanes and warps
      threadsPerWarp[order[rank - 1]] = numThreadsPerWarp / prevLanes;
      warpsPerCTA[order[rank - 1]] = numWarps / prevWarps;

      return $_get(context, sizePerThread, threadsPerWarp, warpsPerCTA, order, CTALayout);
    }]>,

    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$sizePerThread,
                     "ArrayRef<unsigned>":$order,
                     "unsigned":$numWarps,
                     "unsigned":$numThreadsPerWarp,
                     "unsigned":$numCTAs), [{
      unsigned rank = sizePerThread.size();
      SmallVector<unsigned, 4> CTAsPerCGA(rank);
      SmallVector<unsigned, 4> CTASplitNum(rank);
      ArrayRef<unsigned> CTAOrder = order;

      unsigned remainingCTAs = numCTAs;

      // starting from the most strided dimension
      for (int d = rank - 1; d >= 0; --d) {
        unsigned i = order[d];
        CTAsPerCGA[i] = std::clamp<unsigned>(remainingCTAs, 1, shape[i] / sizePerThread[i]);
        CTASplitNum[i] = CTAsPerCGA[i];
        remainingCTAs /= CTAsPerCGA[i];
      }

      CTAsPerCGA[rank - 1] *= remainingCTAs; // wrap at CTA level

      CTALayoutAttr CTALayout = CTALayoutAttr::get(context, CTAsPerCGA, CTASplitNum, CTAOrder);
      return get(context, shape, sizePerThread, order, numWarps, numThreadsPerWarp, CTALayout);
    }]>
  ];

  let extraClassDeclaration = extraDistributedDeclaration # [{
    SliceEncodingAttr squeeze(int axis);

    SmallVector<unsigned> getContigPerThread() {
      // Block encoding is dense stride layout. The elements per thread are contiguous.
      return getSizePerThread();
    };
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// MMA Layout Encoding
//===----------------------------------------------------------------------===//
// TODO: MMAv1 and MMAv2 should be two instances of the same class
def MmaEncodingTrait : AttrInterface<"MmaEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";
  let methods = [

    InterfaceMethod<"Return whether the layout support reduction op.",
                    "bool",
                    "supportReduction">,

    InterfaceMethod<"Return shape per CTA.",
                    "SmallVector<unsigned>",
                    "getShapePerCTATileForDotOperands",
                    (ins "ArrayRef<int64_t>":$tensorShape,
                         "unsigned":$opIdx)>,

    InterfaceMethod<"Return total element size per thread for dot operands.",
                    "unsigned",
                    "getTotalElemsPerThreadForOperands",
                    (ins "ArrayRef<int64_t>":$tensorShape,
                         "Type":$eltTy,
                         "unsigned":$kWidth,
                         "unsigned":$opIdx)>,

    InterfaceMethod<"Return size per thread for dot operands.",
                    "SmallVector<unsigned>",
                    "getSizePerThreadForOperands",
                    (ins "unsigned":$opIdx)>,
  ];
}

def AMDMfmaEncodingAttr : DistributedEncoding<"AMDMfmaEncoding", "amd_mfma_encoding", [MmaEncodingTrait]> {
  let mnemonic = "amd_mfma";

  let description = [{
An encoding for tensors that have been produced by MFMA matrix core instructions,
available on AMD Instinct GPUs of CDNA architectures.

It is characterized by the following parameters:
- `versionMajor` and `versionMinor` indicates the GPU architecture:
  - 1.0: gfx908, i.e. MI100
  - 2.0: gfx90a: i.e. MI200, MI210, MI250
  - 3.0: gfx940, gfx941, gfx942: MI300
- `warpsPerCTA` indicates the warp layout in the block.
- `MDim` and `NDim` indicate the dimension of the output of the mfma instruction.
- `isTransposed` indicates the result tensor is transposed so that it can be converted to dotOperand layout
without going to shared memory. This is used in the case of chained dot (E.g. Flash-Attention kernel).

Example 1:
Suppose we have a tensor with a shape of [32, 64], warpsPerCTA set to [1, 2] and MDim=NDim=32.
The data will be distributed between threads as follows:

                warp 0                                 warp 1
-----------------/\--------------      -----------------/\--------------
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]

Example 2:
Suppose we have a tensor with a shape of [16, 32], warpsPerCTA set to [1, 2] and MDim=NDim=16.
The data will be distributed between threads as follows:

                warp 0                                 warp 1
-----------------/\-------------      ------------------/\---------------
[ 0   1   2   3  ...... 14  15 ]      [ 64  65  66  67  ...... 78   79  ]
[ 0   1   2   3  ...... 14  15 ]      [ 64  65  66  67  ...... 78   79  ]
[ 0   1   2   3  ...... 14  15 ]      [ 64  65  66  67  ...... 78   79  ]
[ 0   1   2   3  ...... 14  15 ]      [ 64  65  66  67  ...... 78   79  ]
[ 16  17  18  19 ...... 30  31 ]      [ 80  81  82  83  ...... 94   95  ]
[ 16  17  18  19 ...... 30  31 ]      [ 80  81  82  83  ...... 94   95  ]
[ 16  17  18  19 ...... 30  31 ]      [ 80  81  82  83  ...... 94   95  ]
[ 16  17  18  19 ...... 30  31 ]      [ 80  81  82  83  ...... 94   95  ]
[ 32  33  34  35 ...... 46  47 ]      [ 96  97  98  99  ...... 110  111 ]
[ 32  33  34  35 ...... 46  47 ]      [ 96  97  98  99  ...... 110  111 ]
[ 32  33  34  35 ...... 46  47 ]      [ 96  97  98  99  ...... 110  111 ]
[ 32  33  34  35 ...... 46  47 ]      [ 96  97  98  99  ...... 110  111 ]
[ 48  49  50  51 ...... 62  63 ]      [ 112 113 114 115 ...... 126  127 ]
[ 48  49  50  51 ...... 62  63 ]      [ 112 113 114 115 ...... 126  127 ]
[ 48  49  50  51 ...... 62  63 ]      [ 112 113 114 115 ...... 126  127 ]
[ 48  49  50  51 ...... 62  63 ]      [ 112 113 114 115 ...... 126  127 ]

Example 3:
Suppose we have a tensor with a shape of [8, 8], warpsPerCTA set to [2, 2] and nonKDim set to 4.
The data will be distributed between threads as follows(note that each element is duploicated in 16 threads):
Suppose we have a tensor with a shape of [8, 8], warpsPerCTA set to [2, 2] and MDim=NDim=4.
The data will be distributed between threads as follows(note that each element is duplicated in 16 threads):

M  N ->                    warp 0                                                       warp 2
| --------------------------/\--------------------------   ------------------------------/\------------------------------
V [ 0,4,8...60   1,5...61     2,6...62     3,7...63    ]   [ 128,132...188  129,133...189  130,134...190  131,135...191 ]
  [ 0,4,8...60   1,5...61     2,6...62     3,7...63    ]   [ 128,132...188  129,133...189  130,134...190  131,135...191 ]
  [ 0,4,8...60   1,5...61     2,6...62     3,7...63    ]   [ 128,132...188  129,133...189  130,134...190  131,135...191 ]
  [ 0,4,8...60   1,5...61     2,6...62     3,7...63    ]   [ 128,132...188  129,133...189  130,134...190  131,135...191 ]
                           warp 1                                                       warp 3
  --------------------------/\--------------------------   ------------------------------/\------------------------------
  [ 64,68...124  65,69...125  66,70...126  67,71...127 ]   [ 192,196...252  193,197...253  194,198...254  195,199...255 ]
  [ 64,68...124  65,69...125  66,70...126  67,71...127 ]   [ 192,196...252  193,197...253  194,198...254  195,199...255 ]
  [ 64,68...124  65,69...125  66,70...126  67,71...127 ]   [ 192,196...252  193,197...253  194,198...254  195,199...255 ]
  [ 64,68...124  65,69...125  66,70...126  67,71...127 ]   [ 192,196...252  193,197...253  194,198...254  195,199...255 ]
}];

  let parameters = (
    ins
    "unsigned": $versionMajor,
    "unsigned": $versionMinor,
    ArrayRefParameter<"unsigned">:$warpsPerCTA__,
    "unsigned":$MDim,
    "unsigned":$NDim,
    "bool":$isTransposed,
    "CTALayoutAttr":$CTALayout
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{
    bool supportReduction() const {
      return true;
    }
    SmallVector<unsigned> getSizePerThreadForOperands(unsigned opIdx) const;
    SmallVector<unsigned> getShapePerCTATileForDotOperands(ArrayRef<int64_t> shape, int opIdx) const;
    unsigned getTotalElemsPerThreadForOperands(ArrayRef<int64_t> shape, Type eltTy, int kWidth, int opIdx) const;
    SmallVector<int64_t> getMFMAInstrShapeForOperands(int kWidth, int opIdx) const;
    SmallVector<int64_t> getMFMARepForOperands(ArrayRef<int64_t> operandShape, int kWidth, int opIdx) const;

    SmallVector<unsigned> getContigPerThread() {
      auto rank = getWarpsPerCTA().size();
      SmallVector<unsigned> contigPerThread(rank, 1);
      if (getIsTransposed())
        contigPerThread[rank - 1] = 4;
      else
        contigPerThread[rank - 2] = 4;
      return contigPerThread;
    };

  }];

  let hasCustomAssemblyFormat = 1;
}

def AMDWmmaEncodingAttr : DistributedEncoding<"AMDWmmaEncoding", "amd_wmma_encoding", [MmaEncodingTrait]> {
  let mnemonic = "amd_wmma";

  let description = [{
An encoding for tensors that have been produced by WMMA instructions,
available on AMD Radeon GPUs of RDNA architectures.

A `warpsPerCTA` parameter characterizes data distribution between warps.
An important limitation of WMMA for layout is a shape for tiles proccessed
by a single warp. It is [16, 16].
This encoding assumes specific access to matrix elements by threads.

Example:
Suppose we have a tensor with shape [32, 48], `warpsPerCTA` set to [2, 3].

    warp 0 [16, 16]           warp 1 [16, 16]           warp 2 [16, 16]
-----------/\----------   -----------/\----------   -----------/\----------
[0   1   2  ... 14  15]   [0   1   2  ... 14  15]   [0   1   2  ... 14  15]
[16  17  18 ... 30  31]   [16  17  18 ... 30  31]   [16  17  18 ... 30  31]
[0   1   2  ... 14  15]   [0   1   2  ... 14  15]   [0   1   2  ... 14  15]
[16  17  18 ... 30  31]   [16  17  18 ... 30  31]   [16  17  18 ... 30  31]
...                       ...                       ...
[0   1   2  ... 14  15]   [0   1   2  ... 14  15]   [0   1   2  ... 14  15]
[16  17  18 ... 30  31]   [16  17  18 ... 30  31]   [16  17  18 ... 30  31]

    warp 3 [16, 16]           warp 4 [16, 16]           warp 5 [16, 16]
-----------/\----------   -----------/\----------   -----------/\----------
[0   1   2  ... 14  15]   [0   1   2  ... 14  15]   [0   1   2  ... 14  15]
[16  17  18 ... 30  31]   [16  17  18 ... 30  31]   [16  17  18 ... 30  31]
[0   1   2  ... 14  15]   [0   1   2  ... 14  15]   [0   1   2  ... 14  15]
[16  17  18 ... 30  31]   [16  17  18 ... 30  31]   [16  17  18 ... 30  31]
...                       ...                       ...
[0   1   2  ... 14  15]   [0   1   2  ... 14  15]   [0   1   2  ... 14  15]
[16  17  18 ... 30  31]   [16  17  18 ... 30  31]   [16  17  18 ... 30  31]
  }];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$warpsPerCTA__,
    "CTALayoutAttr":$CTALayout
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = extraDistributedDeclaration # [{
    bool supportReduction() const {
      return true;
    }
    SmallVector<unsigned> getSizePerThreadForOperands(unsigned opIdx) const;
    SmallVector<unsigned> getShapePerCTATileForDotOperands(ArrayRef<int64_t> shape, int opIdx) const;
    unsigned getTotalElemsPerThreadForOperands(ArrayRef<int64_t> shape, Type eltTy, int kWidth, int opIdx) const;
    SmallVector<int64_t> getWMMAElemsPerInstrForOperands() const;
    SmallVector<int64_t> getWMMARepForOperands(ArrayRef<int64_t> operandShape,
                                      Type elemType, int kWidth, int opIdx) const;
    static SmallVector<unsigned> getMNKDimPerWMMAInstr();

    SmallVector<unsigned> getContigPerThread() {
      auto rank = getWarpsPerCTA().size();
      SmallVector<unsigned> contigPerThread(rank, 1);
      return contigPerThread;
    };
  }];
}

def NvidiaMmaEncodingAttr : DistributedEncoding<"NvidiaMmaEncoding", "nvidia_mma_encoding", [MmaEncodingTrait]> {
  let mnemonic = "nvidia_mma";

  let description = [{
An encoding for tensors that have been produced by tensor cores.

It is characterized by two parameters:
- A 'versionMajor' which specifies the generation the tensor cores
  whose output is being partitioned:
  - 1 for first-gen tensor cores (Volta), and
  - 2 for second-gen tensor cores (Turing/Ampere).
- A 'versionMinor' which indicates the specific layout of a tensor core
  generation, e.g. for Volta, there might be multiple kinds of layouts
  annotated by 0,1,2 and so on.
- A `blockTileSize` to indicate how data should be partitioned between warps.

// -------------------------------- version = 1 --------------------------- //

For first-gen tensor cores, the implicit warpTileSize is [16, 16].
Note: the layout is different from the recommended in PTX ISA
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
(mma.884 section, FP32 accumulator).

For example, when versionMinor=1, the matrix L corresponding to
blockTileSize=[32,16] is:

                               warp 0
--------------------------------/\-------------------------------
[ 0   0   2   2   8   8   10  10   0   0   2   2   8   8   10  10 ]
[ 1   1   3   3   9   9   11  11   1   1   3   3   9   9   11  11 ]
[ 0   0   2   2   8   8   10  10   0   0   2   2   8   8   10  10 ]
[ 1   1   3   3   9   9   11  11   1   1   3   3   9   9   11  11 ]
[ 4   4   6   6   12  12  14  14   4   4   6   6   12  12  14  14 ]
[ 5   5   7   7   13  13  15  15   5   5   7   7   13  13  15  15 ]
[ 4   4   6   6   12  12  14  14   4   4   6   6   12  12  14  14 ]
[ 5   5   7   7   13  13  15  15   5   5   7   7   13  13  15  15 ]
[ 16  16  18  18  20  20  22  22   16  16  18  18  20  20  22  22 ]
[ 17  17  19  19  21  21  23  23   17  17  19  19  21  21  23  23 ]
[ 16  16  18  18  20  20  22  22   16  16  18  18  20  20  22  22 ]
[ 17  17  19  19  21  21  23  23   17  17  19  19  21  21  23  23 ]
[ 24  24  26  26  28  28  30  30   24  24  26  26  28  28  30  30 ]
[ 25  25  27  27  29  29  31  31   25  25  27  27  29  29  31  31 ]
[ 24  24  26  26  28  28  30  30   24  24  26  26  28  28  30  30 ]
[ 25  25  27  27  29  29  31  31   25  25  27  27  29  29  31  31 ]

                          warp 1 = warp0 + 32
--------------------------------/\-------------------------------
[ 32  32  34  34  40  40  42  42   32  32  34  34  40  40  42  42 ]
[ 33  33  35  35  41  41  43  43   33  33  35  35  41  41  43  43 ]
[ ............................................................... ]


// -------------------------------- version = 2 --------------------------- //

For second-gen tensor cores, the implicit warpTileSize is [16, 8].
Information about this layout can be found in the official PTX documentation
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
(mma.16816 section, FP32 accumulator).

For example, the matrix L corresponding to blockTileSize=[32,16] is:
                warp 0                          warp 2
-----------------/\-------------  ----------------/\-------------
[ 0   0   1   1   2   2   3   3   32  32  33  33  34  34  35  35
[ 4   4   5   5   6   6   7   7   36  36  37  37  38  38  39  39
[ ..............................  ..............................
[ 28  28  29  29  30  30  31  31  60  60  61  61  62  62  63  63
[ 0   0   1   1   2   2   3   3   32  32  33  33  34  34  35  35
[ 4   4   5   5   6   6   7   7   36  36  37  37  38  38  39  39
[ ..............................  ..............................
[ 28  28  29  29  30  30  31  31  60  60  61  61  62  62  63  63

              warp 1                           warp 3
----------------/\-------------   ----------------/\-------------
[ 64  64  65  65  66  66  67  67  96  96  97  97  98  98  99  99
[ 68  68  69  69  70  70  71  71  100 100 101 101 102 102 103 103
[ ..............................  ...............................
[ 92  92  93  93  94  94  95  95  124 124 125 125 126 126 127 127
[ 64  64  65  65  66  66  67  67  96  96  97  97  98  98  99  99
[ 68  68  69  69  70  70  71  71  100 100 101 101 102 102 103 103
[ ..............................  ...............................
[ 92  92  93  93  94  94  95  95  124 124 125 125 126 126 127 127

}];

  let parameters = (
    ins
    "unsigned":$versionMajor,
    "unsigned":$versionMinor,
    ArrayRefParameter<"unsigned">:$warpsPerCTA__,
    "CTALayoutAttr":$CTALayout,
    ArrayRefParameter<"unsigned">:$instrShape
  );

  let builders = [
    // Specially for MMAV1(Volta)
    AttrBuilder<(ins "int":$versionMajor,
                     "int":$numWarps,
                     "CTALayoutAttr":$CTALayout,
                     "ArrayRef<unsigned>":$instrShape,
                     "ArrayRef<int64_t>":$shapeC,
                     "bool":$isARow,
                     "bool":$isBRow,
                     "bool":$isAVec4,
                     "bool":$isBVec4,
                     "int":$id), [{
      assert(versionMajor == 1 && "This builder is specially for versionMajor==1");
      // 4-bits to encode 4 booleans: [isARow, isBRow, isAVec4, isBVec4]
      int versionMinor = (isARow * (1<<0)) |\
                         (isBRow * (1<<1)) |\
                         (isAVec4 * (1<<2)) |\
                         (isBVec4 * (1<<3));

      // TODO: Share code with
      // DotOpMmaV1ConversionHelper::AParam/BParam, since same code to compute the
      // rep,spw and fpw.
      SmallVector<unsigned> wpt({1, 1});
      SmallVector<unsigned> wpt_nm1;

      SmallVector<int, 2> rep(2), spw(2);
      std::array<int, 3> fpw{{2, 2, 1}};
      int packSize0 = (isARow || isAVec4) ? 1 : 2;
      rep[0] = 2 * packSize0;
      spw[0] = fpw[0] * 4 * rep[0];

      int packSize1 = (isBRow && !isBVec4) ? 2 : 1;
      rep[1] = 2 * packSize1;
      spw[1] = fpw[1] * 4 * rep[1];

      do {
        wpt_nm1 = wpt;
        if (wpt[0] * wpt[1] < numWarps)
          wpt[0] = std::clamp<int>(wpt[0] * 2, 1, shapeC[0] / spw[0]);
        if (wpt[0] * wpt[1] < numWarps)
          wpt[1] = std::clamp<int>(wpt[1] * 2, 1, shapeC[1] / spw[1]);
      } while (wpt_nm1 != wpt);

      return $_get(context, versionMajor, versionMinor, wpt, CTALayout, instrShape);
    }]>,


    AttrBuilder<(ins "int":$versionMajor,
                     "int":$numWarps,
                     "CTALayoutAttr":$CTALayout,
                     "ArrayRef<unsigned>":$instrShape,
                     "ArrayRef<int64_t>":$shapeA,
                     "ArrayRef<int64_t>":$shapeB,
                     "ArrayRef<int64_t>":$shapeC,
                     "bool":$isARow,
                     "bool":$isBRow,
                     "int":$id), [{
      assert(versionMajor == 1 && "This builder is specially for versionMajor==1");
      bool isAVec4 = !isARow && (shapeA[isARow] <= 16);
      bool isBVec4 = isBRow && (shapeB[isBRow] <= 16);
      return get(context, versionMajor, numWarps, CTALayout, instrShape, shapeC, isARow, isBRow, isAVec4, isBVec4, id);
    }]>
  ];

  let extraClassDeclaration = extraDistributedDeclaration # [{
    bool isVolta() const;
    bool isTuring() const;
    bool isAmpere() const;
    bool isHopper() const;

    unsigned getElemsPerThreadOfOperand(int opIdx, ArrayRef<int64_t> shape) const;

    // Get [isARow, isBRow, isAVec4, isBVec4, id] from versionMinor
    std::tuple<bool, bool, bool, bool, int> decodeVoltaLayoutStates() const;

    // Number of bits in versionMinor to hold the ID of the MMA encoding instance.
    // Here 5 bits can hold 32 IDs in a single module.
    static constexpr int numBitsToHoldMmaV1ID{5};

    // For MMA v1, method `getMMAv1IsRow` returns whether e.g. the a operand is used
    // in the context of an mma.884.row.col or an mma.884.col.col operation. See the PTX ISA documentation
    // section 9.7.13.4.1 for more details.
    bool getMMAv1IsRow(int opIdx) const;
    bool getMMAv1IsVec4(int opIdx) const;
    int getMMAv1NumOuter(ArrayRef<int64_t> shape, int opIdx) const;
    SmallVector<int> getMMAv1Rep(int opIdx) const;
    SmallVector<int> getMMAv1ShapePerWarp(int opIdx) const;
    int getMMAv1Vec(int opIdx) const;
    SmallVector<int64_t> getMMAv2Rep(ArrayRef<int64_t> shape,
                                     int bitwidth, int opIdx) const;

    bool supportReduction() const {
      if (isAmpere() || isHopper()) {
        return true;
      }
      return false;
    };
    SmallVector<unsigned> getSizePerThreadForOperands(unsigned opIdx) const;
    SmallVector<unsigned> getShapePerCTATileForDotOperands(ArrayRef<int64_t> shape, int opIdx) const;
    unsigned getTotalElemsPerThreadForOperands(ArrayRef<int64_t> shape, Type eltTy, int kWidth, int opIdx) const;

    SmallVector<unsigned> getContigPerThread() {
      assert(isVolta() || isAmpere() || isHopper());
      auto rank = getWarpsPerCTA().size();
      SmallVector<unsigned> contigPerThread(rank, 1);
      contigPerThread[rank - 1] = 2;
      return contigPerThread;
    };

  }];

  let hasCustomAssemblyFormat = 1;
}

def SliceEncodingAttr : DistributedEncoding<"SliceEncoding", "slice_encoding"> {
  let mnemonic = "slice";

  let description = [{
    Given a `parent` layout and a `dim`, squeezes the given `dim` in the `parent`
    layout and distributes values in a tensor T according to the new layout.

    For example, given

    T = [x  x  x  x  x  x  x  x]
    L_parent = [0  1  2  3 ]
               [4  5  6  7 ]
               [8  9  10 11]
               [12 13 14 15] (with 16 CUDA threads)

    With dim = 0, squeezing out dim 0, we have
    L = [{0,4,8,12},  {1,5,9,13}, {2,6,10,14},  {3,7,11,15} ]

    Then the data of T would be distributed as follow between the 16 CUDA threads:
    L(T) = [ {0,4,8,12} , {1,5,9,13} , ... {3,7,11,15}, {0,4,8,12} , ..., {3,7,11,15} ]

    With dim = 1, squeezing out dim 1, we have
    L = [ {0,1,2,3}, {4,5,6,7}, {8,9,10,11}, {12,13,14,15} ]

    Then the data of T would be distributed as follow between the 16 CUDA threads:
    L = [ {0,1,2,3}, {4,5,6,7}, ..., {12,13,14,15}, {0,1,2,3}, ..., {12,13,14,15} ]

    This is useful for constructing the inverse layout of an expand_dims operation
    during some optimization passes.
  }];

  let parameters = (
    ins
    "unsigned":$dim,
    // TODO: constraint here to only take distributed encodings
    "Attribute":$parent
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{
    template<class T>
    SmallVector<T> paddedShape(ArrayRef<T> shape) const;

    SmallVector<unsigned> getContigPerThread() {
      auto parentLayout = mlir::cast<DistributedEncodingTrait>(getParent());
      auto parentContigPerThread = parentLayout.getContigPerThread();
      parentContigPerThread.erase(parentContigPerThread.begin() + getDim());
      return parentContigPerThread;
    };
  }];

  let hasCustomAssemblyFormat = 1;
}

def DotOperandEncodingAttr : DistributedEncoding<"DotOperandEncoding", "dot_operand_encoding"> {
  let mnemonic = "dot_op";

  let description = [{
In the TritonGPU dialect, given `d = tt.dot a, b, c` tt.dot's operands a and b
must be of DotOperandEncodingAttr layout, if the dot is MMA v1 or v2 (i.e.
pre-Hopper).  For MMA v3, the operands are *almost always* in a regular shared
encoding, but sometimes the LHS is also a dot-operand encoding.

a's opIdx is 0, b's opIdx is 1.

The parent field is the layout of d.

kWidth defines number of consecutive elements stored by one thread along k dimension.
Some layouts do not use this parameter, either because they have a fixed number of
elements along the K dim, or they use all elements of the tensor along the K dim.
  }];

  let parameters = (
    ins
    "unsigned":$opIdx,
    "Attribute":$parent,
    DefaultValuedParameter<"unsigned", "0">:$kWidth
  );

  let builders = [
        // Specially for MMAV1(Volta)
    AttrBuilder<(ins "unsigned":$opIdx,
                     "Attribute":$parent,
                     "Type":$eltTy), [{
      NvidiaMmaEncodingAttr parentAttr = mlir::dyn_cast<NvidiaMmaEncodingAttr>(parent);
      if (!parentAttr || !parentAttr.isAmpere())
        return $_get(context, opIdx, parent, 0);
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      unsigned MMAv2kWidth = 32 / bitwidth;
      return $_get(context, opIdx, parent, MMAv2kWidth);
    }]>
  ];

  let assemblyFormat = "`<` `{` struct(params) `}` `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = extraDistributedDeclaration # [{
    SmallVector<unsigned> getContigPerThread() {
      return getSizePerThread();
    };
  }];
}

def TTG_SharedMemorySpace : AttrDef<TritonGPU_Dialect, "SharedMemorySpace"> {
  let mnemonic = "shared_memory";
  let description = [{
    Attribute to indicate that the memory descriptor points to shared memory.
  }];
}
#endif
