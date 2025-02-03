#ifndef TRITONGPU_DIALECT
#define TRITONGPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonGPU_Dialect : Dialect {
  let name = "ttg";

  let cppNamespace = "::mlir::triton::gpu";

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton GPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "mlir::gpu::GPUDialect",
  ];

  let extraClassDeclaration = [{
    static std::string getNumWarpsAttrName() { return "ttg.num-warps"; }
    static int getNumWarps(ModuleOp mod) {
      if (!mod->hasAttr("ttg.num-warps"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a ttg.num-warps attribute");
      return cast<IntegerAttr>(mod->getAttr("ttg.num-warps")).getInt();
    }
    static int getNumCTAs(ModuleOp mod) {
      if (!mod->hasAttr("ttg.num-ctas"))
        return 1;
      return cast<IntegerAttr>(mod->getAttr("ttg.num-ctas")).getInt();
    }
    void registerTypes();

    static std::string getThreadsPerWarpAttrName() { return "ttg.threads-per-warp"; }

    static int getThreadsPerWarp(ModuleOp mod) {
      Attribute threadsPerWarp = mod->getDiscardableAttr("ttg.threads-per-warp");
      if(!threadsPerWarp) {
        return 32;
      }
      return cast<IntegerAttr>(threadsPerWarp).getInt();
    }

    LinearLayout toLinearLayout(ArrayRef<int64_t> shape, Attribute layout);

    private:
      LinearLayoutCache llCache;
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
