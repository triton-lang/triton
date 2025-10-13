set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Extensions)
add_public_tablegen_target(TritonGPUExtensionIncGen)

llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

set(GPU_EXTENSION_PASSES
    GPUExtensionTestLib
    )

set(GPUExtensionTestLib_SOURCES
    MLIRPluginHello.cpp
    )

get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
get_property(f2reduce_libs GLOBAL PROPERTY F2REDUCE2_LIBS)

foreach( plugin ${GPU_EXTENSION_PASSES} )
    add_mlir_library(${plugin}
        ${${plugin}_SOURCES}
        SHARED

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_BINARY_DIR}/lib

        DEPENDS
        TritonTableGen
        TritonCanonicalizeIncGen
        TritonGPUExtensionIncGen

        LINK_LIBS PUBLIC
        ${triton_libs}
        ${f2reduce_libs}
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFunctionInterfaces
        MLIRFuncDialect
        MLIRPass
        LLVMSupport
        MLIRSupport
        TritonGPUIR
        TritonNvidiaGPUIR
        TritonInstrumentIR
        MLIROptLib
        MLIRPass
        MLIRRegisterAllDialects
        MLIRRegisterAllPasses
        MLIRTransforms
        Python3::Module
        pybind11::headers
        "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
        # ${TRITON_LIB_DIR}/libtriton.so
        )

    # CMAKE_LIBRARY_OUTPUT_DIRECTORY is only set during the Python
    # build. It is empty if building directly from the root
    # CMakeLists.txt file. Therefore if not building from Python just
    # use the default CMake shared lib path otherwise this causes a hard
    # build error
    if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(${plugin} PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY
      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../mlir-plugin")
    endif(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)

    target_link_options(${plugin} PRIVATE -rdynamic)

    # This is set to -fvisibility=hidden in the top level CMake file
    # which causes the llvmGetPassPluginInfo symbol to be hidden and
    # an "entry point not found" error. Reset it just for this target
    if(NOT MSVC)
        target_compile_options(${plugin} PRIVATE -fvisibility=default -fPIC)
    endif()

    target_compile_definitions(${plugin} PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
    target_link_libraries(${plugin} ${Python_LIBRARIES})
    target_link_options(${plugin} PRIVATE ${PYTHON_LDFLAGS} -Wl,--no-as-needed)
endforeach()
