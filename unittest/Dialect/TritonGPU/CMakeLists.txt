set(LLVM_TARGET_DEFINITIONS TestDialect.td)
mlir_tablegen(TestDialect.h.inc -gen-dialect-decls -dialect=tt_test)
mlir_tablegen(TestDialect.cpp.inc -gen-dialect-defs -dialect=tt_test)
mlir_tablegen(TestGPUAttrDefs.h.inc -gen-attrdef-decls --attrdefs-dialect=tt_test)
mlir_tablegen(TestGPUAttrDefs.cpp.inc -gen-attrdef-defs --attrdefs-dialect=tt_test)
add_public_tablegen_target(TritonTestGPUAttrDefsIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR}) # Tablegen'd files

add_triton_library(TritonGPUTestIR

    DEPENDS
    TritonGPUTableGen
    TritonGPUAttrDefsIncGen
    TritonGPUTypeInterfacesIncGen
    TritonTestGPUAttrDefsIncGen

    LINK_LIBS PUBLIC
    MLIRGPUDialect
    TritonIR
    TritonTools
)

add_triton_ut(
  NAME TestSwizzling
  SRCS SwizzleTest.cpp
  LIBS
    TritonTools
    LLVMSupport
    MLIRSupport
)
add_triton_ut(
  NAME Dialect
  SRCS DialectTest.cpp
  LIBS
    MLIRParser
    TritonGPUIR
    TritonGPUTestIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)
add_triton_ut(
  NAME LinearLayoutConversions
  SRCS LinearLayoutConversionsTest.cpp
  LIBS
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)

add_triton_ut(
  NAME DumpLayoutTest
  SRCS DumpLayoutTest.cpp
  LIBS
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)
