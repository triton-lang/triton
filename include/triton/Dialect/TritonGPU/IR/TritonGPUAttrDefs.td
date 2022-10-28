#ifndef TRITONGPU_ATTRDEFS
#define TRITONGPU_ATTRDEFS

include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"

//===----------------------------------------------------------------------===//
// TritonGPU Attribute Definitions
//===----------------------------------------------------------------------===//

class TritonGPU_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<TritonGPU_Dialect, name, traits, baseCppClass> {

  let description = [{
TritonGPU Tensors differ from usual tensors in that they contain a _layout_ attribute which determines
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

Right now, Triton implements two classes of layouts: shared, and distributed.
  }];

  code extraBaseClassDeclaration = [{
    unsigned getElemsPerThread(ArrayRef<int64_t> shape) const;
    ::mlir::LogicalResult verifyLayoutForArg(::mlir::Operation* op, unsigned argNo) const;
  }];
}

//===----------------------------------------------------------------------===//
// Shared Layout Encoding
//===----------------------------------------------------------------------===//

def SharedEncodingAttr : TritonGPU_Attr<"SharedEncoding"> {
  let mnemonic = "shared";

  let description = [{
An encoding for tensors whose elements may be simultaneously accessed by
different cuda threads in the programs, via shared memory. In other words,
for all indices i \in R^d, \mathcal{L}(i) = {0, 1, ..., 32*num_warps - 1}.

In order to avoid shared memory bank conflicts, elements may be swizzled
in memory. For example, a swizzled row-major layout could store its data
as follows:

A_{0, 0}  A_{0, 1}  A_{0, 2}  A_{0, 3} ...   [phase 0] \ per_phase = 2
A_{1, 0}  A_{1, 1}  A_{1, 2}  A_{1, 3} ...   [phase 0] /
groups of vec=2 elements
are stored contiguously
_ _ _ _ /\_ _ _ _
A_{2, 2}  A_{2, 3}  A_{2, 0}  A_{2, 1} ...   [phase 1] \ per phase = 2
A_{3, 2}  A_{3, 3}  A_{3, 0}  A_{3, 1} ...   [phase 1] /
  }];

  let parameters = (
    ins
    // swizzle info
    "unsigned":$vec, "unsigned":$perPhase, "unsigned":$maxPhase,
    ArrayRefParameter<"unsigned", "order of axes by the rate of changing">:$order
  );

  let extraClassDeclaration = extraBaseClassDeclaration;
}

//===----------------------------------------------------------------------===//
// Distributed Layout Encoding
//===----------------------------------------------------------------------===//

class DistributedEncoding<string name> : TritonGPU_Attr<name> {
  let description = [{
Distributed encodings have a layout function that is entirely characterized
by a d-dimensional tensor L. Note that L doesn't need to have the same shape
(or even the same rank) as the tensor it is encoding.

The layout function \mathcal{L} of this layout is then defined, for an
index `i` \in R^D, as follows:

\mathcal{L}(A)[i_d] = L[(i_d + k_d*A.shape[d]) % L.shape[d]] \forall k_d such as i_d + k_d*A.shape[d] < L.shape[d]

For example, for a tensor/layout pair
A = [x  x  x  x  x  x  x  x]
    [x  x  x  x  x  x  x  x]
L = [0  1  2  3 ]
    [4  5  6  7 ]
    [8  9  10 11]
    [12 13 14 15]

Then the data of A would be distributed as follow between the 16 CUDA threads:
L(A) = [ {0,8} , {1,9} , {2,10}, {3,11}, {0,8} , {1, 9} , {2, 10}, {3, 11},
         {4,12}, {5,13}, {6,14}, {7,15}, {4,12}, {5, 13}, {6, 14}, {7, 15} ]
  }];

  let extraClassDeclaration = extraBaseClassDeclaration;
}

//===----------------------------------------------------------------------===//
// Blocked Layout Encoding
//===----------------------------------------------------------------------===//

def BlockedEncodingAttr : DistributedEncoding<"BlockedEncoding"> {
  let mnemonic = "blocked";

  let description = [{
An encoding where each warp owns a contiguous portion of the target tensor. This is typically the kind of data layout
used to promote memory coalescing in LoadInst and StoreInst.
It is characterized by three tuples -- thread tile size, warp tile size, and block tile size -- which
specify the amount of elements owned by each CUDA thread, warp and CTA respectively.

For example, a row-major coalesced layout may partition a 16x16 tensor over 2 warps (i.e. 64 threads) as follows.

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
}>
}];


  let builders = [
    // Custom builder initializes sizePerWarp and sizePerCTA automatically
    // TODO: compiles on MacOS but not linux?
    // AttrBuilder<(ins "ArrayRef<unsigned>":$sizePerThread,
    //                  "ArrayRef<unsigned>":$threadsPerWarp,
    //                  "ArrayRef<unsigned>":$warpsPerCTA,
    //                  "ArrayRef<unsigned>":$order), [{
    //   int rank = threadsPerWarp.size();
    //   SmallVector<unsigned, 4> sizePerWarp(rank);
    //   SmallVector<unsigned, 4> sizePerCTA(rank);
    //   for (unsigned i = 0; i < rank; i++) {
    //     sizePerWarp.push_back(sizePerThread[i] * threadsPerWarp[i]);
    //     sizePerCTA.push_back(sizePerWarp[i] * warpsPerCTA[i]);
    //   }
    //   return $_get(context, sizePerThread, threadsPerWarp, warpsPerCTA, order, sizePerWarp, sizePerCTA);
    // }]>,
    // Custom builder initializes sizePerWarp and sizePerCTA automatically
    // Default builder takes sizePerThread, order and numWarps, and tries to
    // pack numWarps*32 threads in the provided order for use in a type
    // of the given shape.
    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$sizePerThread,
                     "ArrayRef<unsigned>":$order,
                     "unsigned":$numWarps), [{
      int rank = sizePerThread.size();
      int remainingWarps = numWarps;
      int remainingLanes = 32;
      SmallVector<unsigned, 4> threadsPerWarp(rank);
      SmallVector<unsigned, 4> warpsPerCTA(rank);
      for (int _dim = 0; _dim < rank; ++_dim) {
        int dim = order[_dim];
        int maxNumThreads = int(shape[dim]) / sizePerThread[dim];
        warpsPerCTA[dim] = std::clamp(remainingWarps, 1, maxNumThreads);
        maxNumThreads = maxNumThreads / warpsPerCTA[dim];
        threadsPerWarp[dim] = std::clamp(remainingLanes, 1, maxNumThreads);
        remainingWarps /= warpsPerCTA[dim];
        remainingLanes /= threadsPerWarp[dim];
      }

      return $_get(context, sizePerThread, threadsPerWarp, warpsPerCTA, order);

    }]>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    SliceEncodingAttr squeeze(int axis);
  }];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$sizePerThread,
    ArrayRefParameter<"unsigned">:$threadsPerWarp,
    ArrayRefParameter<"unsigned">:$warpsPerCTA,
    // fastest-changing axis first
    ArrayRefParameter<
      "unsigned",
      "order of axes by the rate of changing"
    >:$order
    // These attributes can be inferred from the rest
    // ArrayRefParameter<"unsigned">:$sizePerWarp,
    // ArrayRefParameter<"unsigned">:$sizePerCTA
  );

}

//===----------------------------------------------------------------------===//
// MMA Layout Encoding
//===----------------------------------------------------------------------===//
// TODO: MMAv1 and MMAv2 should be two instances of the same class

def MmaEncodingAttr : DistributedEncoding<"MmaEncoding"> {
  let mnemonic = "mma";

  let description = [{
An encoding for tensors that have been produced by tensor cores.
It is characterized by two parameters:
- A 'version' which specifies the generation the tensor cores
whose output is being partitioned: 1 for first-gen tensor cores (Volta),
and 2 for second-gen tensor cores (Turing/Ampere).
- A `blockTileSize` to indicate how data should be
partitioned between warps.

// -------------------------------- version = 1 --------------------------- //

For first-gen tensor cores, the implicit warpTileSize is [16, 16].
Information about this layout can be found in the official PTX documentation
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
(mma.884 section, FP32 accumulator).

For example, the matrix L corresponding to blockTileSize=[32,16] is:

                               warp 0
--------------------------------/\-------------------------------
[ 0   0   2   2   0   0   2   2    4   4   6   6   4   4   6   6 ]
[ 1   1   3   3   1   1   3   3    5   5   7   7   5   5   7   7 ]
[ 0   0   2   2   0   0   2   2    4   4   6   6   4   4   6   6 ]
[ 1   1   3   3   1   1   3   3    5   5   7   7   5   5   7   7 ]
[ 16  16  18  18  16  16  18  18   20  20  22  22  20  20  22  22]
[ 17  17  19  19  17  17  19  19   21  21  23  23  21  21  23  23]
[ 16  16  18  18  16  16  18  18   20  20  22  22  20  20  22  22]
[ 17  17  19  19  17  17  19  19   21  21  23  23  21  21  23  23]
[ 8   8   10  10  8   8   10  10   12  12  14  14  12  12  14  14]
[ 9   9   11  11  9   9   11  11   13  13  15  15  13  13  15  15]
[ ..............................................................
[ ..............................................................
[ 24  24  26  26  24  24  26  26   28  28  30  30  28  28  30  30]
[ 25  25  27  27  25  25  27  27   29  29  31  31  29  29  31  31]

                         warp 1 = warp0 + 32
--------------------------------/\-------------------------------
[ 32  32  34  34  32  32  34  34   36  36  38  38  36  36  38  38]
[ 33  33  35  35  33  33  35  35   37  37  39  39  37  37  39  39]
[ ..............................................................
[ ..............................................................
[ 56  56  58  58  56  56  58  58   60  60  62  62  60  60  62  62]
[ 57  57  59  59  57  57  59  59   61  61  63  63  61  61  63  63]

// -------------------------------- version = 2 --------------------------- //

For second-gen tensor cores, the implicit warpTileSize is [16, 8].
Information about this layout can be found in the official PTX documentation
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
(mma.16816 section, FP32 accumulator).

For example, the matrix L corresponding to blockTileSize=[32,16] is:
                warp 0                          warp 1
-----------------/\-------------  ----------------/\-------------
[ 0   0   1   1   2   2   3   3   32  32  33  33  34  34  35  35
[ 4   4   5   5   6   6   7   7   36  36  37  37  38  38  39  39
[ ..............................  ..............................
[ 28  28  29  29  30  30  31  31  60  60  61  61  62  62  63  63
[ 0   0   1   1   2   2   3   3   32  32  33  33  34  34  35  35
[ 4   4   5   5   6   6   7   7   36  36  37  37  38  38  39  39
[ ..............................  ..............................
[ 28  28  29  29  30  30  31  31  60  60  61  61  62  62  63  63

              warp 3                           warp 4
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
    "unsigned":$version,
    ArrayRefParameter<"unsigned">:$warpsPerCTA
  );

  let extraClassDeclaration = extraBaseClassDeclaration;
}

def SliceEncodingAttr : DistributedEncoding<"SliceEncoding"> {
  let mnemonic = "slice";

  let description = [{
    TODO: improve docs

    A = [x  x  x  x  x  x  x  x]

    parent = [0  1  2  3 ]
             [4  5  6  7 ]
             [8  9  10 11]
             [12 13 14 15]
    dim = 0

    Then the data of A would be distributed as follow between the 16 CUDA threads:
    L(A) = [ {0,4,8,12} , {1,5,9,13} , ... {3,7,11,15}, {0,4,8,12} , ..., {3,7,11,15} ]

    This is useful for constructing the inverse layout of an expand_dims operation during some optimization passes.

  }];

  let parameters = (
    ins
    "unsigned":$dim,
    // TODO: constraint here to only take distributed encodings
    "Attribute":$parent
  );

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    SmallVector<int64_t> paddedShape(ArrayRef<int64_t> shape) const;
  }];
}


def DotOperandEncodingAttr : DistributedEncoding<"DotOperandEncoding"> {
  let mnemonic = "dot_op";

  let description = [{
In TritonGPU dialect, considering `d = tt.dot a, b, c`
tt.dot's operands a and b must be of DotOperandEncodingAttr layout.
a's opIdx is 0, b's opIdx is 1.
The parend field in DotOperandEncodingAttr is the layout of d.
  }];

  let parameters = (
    ins
    "unsigned":$opIdx,
    "Attribute":$parent
  );

  let extraClassDeclaration = extraBaseClassDeclaration;
}

#endif
