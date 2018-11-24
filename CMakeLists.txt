cmake_minimum_required(VERSION 2.8.7)
project(TDL)

# LLVM
include(cmake/FindLLVM.cmake)

# Link directories
link_directories(/home/philippe/Development/llvm-tlvm/build/lib)
# Include directories
include_directories(/home/philippe/Development/llvm-tlvm/include)
include_directories(/home/philippe/Development/llvm-tlvm/build/include)

# Flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -Wno-strict-aliasing")

# Executables
foreach(PROG gemm conv)
     add_executable(${PROG} ${PROG}.cpp)
     set_target_properties(${PROG} PROPERTIES OUTPUT_NAME ${PROG})
     target_link_libraries(${PROG} ${LLVM_LIBRARIES})
endforeach()
