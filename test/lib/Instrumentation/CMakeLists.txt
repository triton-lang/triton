set(GPU_INSTRUMENTATION_PASSES
	GPUInstrumentationTestLib
    )

set(GPUInstrumentationTestLib_SOURCES
    GPUHello.cpp
    )


foreach( plugin ${GPU_INSTRUMENTATION_PASSES} )
    add_library(
      ${plugin}
      SHARED
      ${${plugin}_SOURCES}
      )

    target_link_libraries(
      ${plugin}
      PRIVATE
      LLVMCore
      "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
      )
     # This is set to -fvisibility=hidden in the top level CMake file
     # which causes the llvmGetPassPluginInfo symbol to be hidden and
     # an "entry point not found" error. Reset it just for this target
     target_compile_options(${plugin} PRIVATE -fvisibility=default)

     set_target_properties(${plugin} PROPERTIES
	     LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/instrumentation/triton")
endforeach()
