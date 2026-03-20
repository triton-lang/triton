# This target builds a stub (empty) libtriton.so for use with triton-opt plugins.
#
# Problem: Triton plugins are shared libraries that link against libtriton. When
# triton-opt loads a plugin via dlopen, the dynamic linker tries to find and load
# libtriton.so to satisfy that dependency. However, triton-opt already statically
# provides all the necessary LLVM/MLIR/Triton symbols, so loading the real
# libtriton.so is both unnecessary and harmful — it pulls in Python and other
# unwanted dependencies.
#
# Solution: LD_PRELOAD this empty libtriton.so when running triton-opt. The
# dynamic linker sees the dependency as satisfied, and the plugin's symbols
# resolve against triton-opt itself.

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/empty.c "")
add_library(faketriton SHARED ${CMAKE_CURRENT_BINARY_DIR}/empty.c)

set_target_properties(faketriton PROPERTIES
  OUTPUT_NAME "triton"
)

if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(faketriton PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY
  "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins")
endif(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
