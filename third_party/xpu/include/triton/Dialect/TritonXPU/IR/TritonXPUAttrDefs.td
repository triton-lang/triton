//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TRITONXPU_ATTRDEFS
#define TRITONXPU_ATTRDEFS

include "mlir/IR/Interfaces.td" // AttrInterface
include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonXPU/IR/TritonXPUDialect.td"

//===----------------------------------------------------------------------===//
// TritonXPU Attribute Definitions
//===----------------------------------------------------------------------===//
def TritonXPU_AttrTrait : AttrInterface<"TritonXPU_AttrTrait"> {
  let cppNamespace = "::mlir::triton::xpu";

  let methods = [
    InterfaceMethod<"Get the shape of the values per core.",
                    "SmallVector<unsigned>",
                    "getSizePerCoreInterface">,

    InterfaceMethod<"Get the shape of the core per group",
                    "SmallVector<unsigned>",
                    "getCoresPerGroupInterface">,

    InterfaceMethod<"Get the shape of the groups per cluster.",
                    "SmallVector<unsigned>",
                    "getGroupsPerClusterInterface">,

    InterfaceMethod<"Get the shape of the cores per cluster.",
                    "SmallVector<unsigned>",
                    "getCoresPerClusterInterface">,

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

class TritonXPU_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                     Dialect dialect = TritonXPU_Dialect,
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<dialect, name, !listconcat([TritonXPU_AttrTrait], traits), baseCppClass> {

  let description = [{
    The base class of TritonXPU Encoding Attribute.
  }];
  let attrName = "triton.xpu." # attrMnemonic;

  code extraBaseClassDeclaration = [{
    SmallVector<unsigned> getSizePerCoreInterface() const;
    SmallVector<unsigned> getCoresPerGroupInterface() const;
    SmallVector<unsigned> getGroupsPerClusterInterface() const;

    SmallVector<unsigned> getCoresPerClusterInterface() const;

    unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const;
    SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const;
    ::mlir::LogicalResult verifyLayoutForArg(::mlir::Operation* op, unsigned argNo) const;
  }];
}


//===----------------------------------------------------------------------===//
// Cluster Layout
//===----------------------------------------------------------------------===//


def ClusterLayoutAttr : TritonXPU_Attr<"ClusterLayout", "cluster_layout"> {
  let mnemonic = "cluster";

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$sizePerCore,
    ArrayRefParameter<"unsigned">:$coresPerGroup,
    ArrayRefParameter<"unsigned">:$groupsPerCluster,
    ArrayRefParameter<"unsigned">:$order, // the fastest-changing axis first
    "bool":$isReduceOpt
  );

  let description = [{
For XPU hardware, the number of cores is the number of threads used.
We need try to make one thread to cover a continuous memory as long as
possible.

Example 1, a row-major coalesced layout may partition a 32x16 tensor over 64 threads as follows.

[ 0  0  0  0  0  0  0  0  ; 1 1 1 1 1 1 1 1 ]
[ 2  2  2  2  2  2  2  2  ; 3 3 3 3 3 3 3 3 ]
...
[ 60 60 60 60 60 60 60 60 ; 61 61 61 61 61 61 ]
[ 62 62 62 62 62 62 62 62 ; 63 63 63 63 63 63 ]

for

#triton_xpu.cluster_layout<{
  sizePerCore = {8}
  coresPerGroup = {4}
  groupsPerCluster = {16}
}>



Example 2, a row-major coalesced layout may partition a 32x16 1D tensor over 64 threads as follows.

[ 0  0  0  0  0  0  0  0  ; 0  0  0  0  0  0  0  0 ]
[ 1  1  1  1  1  1  1  1  ; 1  1  1  1  1  1  1  1 ]
...
[ 30  30  30  30  30  30  30  30  ; 30  30  30  30  30  30  30  30  ]
[ 31  31  31  31  31  31  31  31  ; 31  31  31  31  31  31  31  31  ]

for

#triton_xpu.cluster_layout<{
  sizePerCore = {16}
  coresPerGroup = {4}
  groupsPerCluster = {16}
}>

core_32-core_63 will be idle



Example 3, a row-major coalesced layout may partition a [32, 16] 2D tensor over 64 threads as follows.

[ 0  0  0  0  0  0  0  0  ; 1 1 1 1 1 1 1 1 ]
[ 2  2  2  2  2  2  2  2  ; 3 3 3 3 3 3 3 3 ]
...
[ 60 60 60 60 60 60 60 60 ; 61 61 61 62 62 62 ]
[ 62 62 62 62 62 62 62 62 ; 63 63 63 63 63 63 ]

for

#triton_xpu.cluster_layout<{
  sizePerCore = {1, 8}
  coresPerGroup = {1, 4}
  groupsPerCluster = {1, 16}
}>



Example 4, a row-major coalesced layout may partition a [32, 16] 2D tensor over 64 threads as follows.

[ 0  0  0  0  0  0  0  0  ; 0  0  0  0  0  0  0  0 ]
[ 1  1  1  1  1  1  1  1  ; 1  1  1  1  1  1  1  1 ]
...
[ 30  30  30  30  30  30  30  30  ; 30  30  30  30  30  30  30  30  ]
[ 31  31  31  31  31  31  31  31  ; 31  31  31  31  31  31  31  31  ]

for

#triton_xpu.cluster_layout<{
  sizePerCore = {1, 16}
  coresPerGroup = {4, 1}
  groupsPerCluster = {16, 1}
}>

core_32-core_63 will be idle


  }];

  let genVerifyDecl = 1;

  let builders = [
    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "unsigned":$buffer_size,
                     "unsigned":$core_num), [{
        int rank = shape.size();
        SmallVector<unsigned, 4> sizePerCore(rank, 1u);
        SmallVector<unsigned, 4> coresPerGroup(rank, 1u);
        SmallVector<unsigned, 4> groupsPerCluster(rank, 1u);

        coresPerGroup[rank-1] = core_num;
        groupsPerCluster[rank-1] = 1;
        sizePerCore[rank-1] = std::min(buffer_size, static_cast<unsigned>(
                        std::ceil(static_cast<float>(shape[rank-1]) / core_num)));

        return $_get(context, sizePerCore, coresPerGroup, groupsPerCluster, order, /*isReduceOpt*/false);
    }]>,
  ];

  let extraClassDeclaration = extraBaseClassDeclaration;

  //   let skipDefaultBuilders = 1; // will skip get method(use parameters directly)
  let hasCustomAssemblyFormat = 1;
}

#endif // TRITONXPU_ATTRDEFS
