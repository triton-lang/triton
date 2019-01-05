cmake_minimum_required(VERSION 2.8)
project(TDL)
include(CTest)

# FLEX/YACC
find_package(BISON)
find_package(FLEX)
BISON_TARGET(Parser ${CMAKE_CURRENT_SOURCE_DIR}/include/ast/parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp)
FLEX_TARGET(Lexer ${CMAKE_CURRENT_SOURCE_DIR}/include/ast/scanner.l ${CMAKE_CURRENT_BINARY_DIR}/scanner.cpp)
get_filename_component(BISON_Parser_INCLUDE_DIRECTORIES ${BISON_Parser_OUTPUT_HEADER} DIRECTORY)
include_directories(${BISON_Parser_INCLUDE_DIRECTORIES})

# LLVM
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(llvm_libs support core irreader)

#Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Gather headers for cmake-based IDEs
file( GLOB_RECURSE ALL_SRC *.cpp *.hpp *.h *.py *.y *.l)
add_custom_target( ALL SOURCES ${ALL_SRC} )

# Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS} -std=c++11")

# TDL
file(GLOB_RECURSE LIBTDL_SRC lib/*.cpp)
add_library(tdl SHARED ${LIBTDL_SRC} ${BISON_Parser_OUTPUTS} ${FLEX_Lexer_OUTPUTS})
target_link_libraries(tdl ${llvm_libs})

# Examples
add_subdirectory(examples)





