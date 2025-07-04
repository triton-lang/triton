#ifndef TRITON_GPU_TEST_DIALECT
#define TRITON_GPU_TEST_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"

def TritonTest_Dialect : Dialect {
  let name = "tt_test";
}


def TestAttr : DistributedEncoding<"TestAttr", "test_attr",
                                  [MmaEncodingTrait], TritonTest_Dialect> {
  let mnemonic = "test_mma";

  let parameters = (
    ins
    "bool":$fail
  );

  let assemblyFormat = "`<` `{` struct(params) `}` `>`";

  let extraClassDeclaration = [{

    SmallVector<unsigned> getRepOrder() const {
      return {1, 0};
    }

    SmallVector<unsigned> getCTAOrder() const {
      return {1, 0};
    }

    SmallVector<unsigned> getCTAsPerCGA() const {
      return {1, 1};
    }

    SmallVector<unsigned> getCTASplitNum() const {
      return {1, 1};
    }

    SmallVector<unsigned> getRepOrderForOperand(unsigned opIdx) const {
      return {1, 1};
    }

    triton::LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const {
      return triton::LinearLayout{};
    }

  }];
}


#endif
