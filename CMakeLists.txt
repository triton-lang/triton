cmake_minimum_required(VERSION 2.8)
project(triton)
include(CTest)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# FLEX/YACC
find_package(BISON)
find_package(FLEX)
BISON_TARGET(Parser ${CMAKE_CURRENT_SOURCE_DIR}/include/triton/ast/parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp)
FLEX_TARGET(Lexer ${CMAKE_CURRENT_SOURCE_DIR}/include/triton/ast/scanner.l ${CMAKE_CURRENT_BINARY_DIR}/scanner.cpp)
get_filename_component(BISON_Parser_INCLUDE_DIRECTORIES ${BISON_Parser_OUTPUT_HEADER} DIRECTORY)
include_directories(${BISON_Parser_INCLUDE_DIRECTORIES})

#execute_process(COMMAND python -c "import tensorflow as tf; print(tf.__cxx11_abi_flag__ if \"__cxx11_abi_flag__\" in tf.__dict__ else 0)"
#                OUTPUT_VARIABLE TF_ABI OUTPUT_STRIP_TRAILING_WHITESPACE)
#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# LLVM
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
#llvm_map_components_to_libnames(llvm_libs all)

#Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Gather headers for cmake-based IDEs
file( GLOB_RECURSE ALL_SRC *.cpp *.hpp *.h *.py *.y *.l CMakeLists*)
add_custom_target( ALL SOURCES ${ALL_SRC} )

# Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS} -std=c++11")

# Triton
file(GLOB_RECURSE LIBTRITON_SRC lib/*.cpp)
add_library(triton SHARED ${LIBTRITON_SRC} ${BISON_Parser_OUTPUTS} ${FLEX_Lexer_OUTPUTS})
target_link_libraries(triton LLVM)

# Examples
add_subdirectory(examples)





