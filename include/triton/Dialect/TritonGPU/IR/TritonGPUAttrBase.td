//===----------------------------------------------------------------------===//
// Base definitions shared by TritonGPU attribute TableGen files.
// Splitting these out lets us emit certain attributes (e.g. CGAEncodingAttr)
// before interface headers without creating circular dependencies.
//===----------------------------------------------------------------------===//

#ifndef TRITONGPU_ATTRBASE_TD
#define TRITONGPU_ATTRBASE_TD

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"

// Traits used across several attrs.
def MemDescViewTrait : NativeOpTrait<"MemDescViewTrait">;
def LocalLoadTrait : NativeOpTrait<"LocalLoadTrait">;
def MemWaitOpTrait : NativeOpTrait<"MemWaitOpTrait">;

// Common parameter helpers.
def LinearLayoutParam : AttrOrTypeParameter<"LinearLayout",
                                            "linear layout"> {
  let cppAccessorType = "const LinearLayout &";
}

// Base class for all TritonGPU attributes.
class TritonGPU_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<TritonGPU_Dialect, name, traits> {

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
  }];
}

#endif // TRITONGPU_ATTRBASE_TD
