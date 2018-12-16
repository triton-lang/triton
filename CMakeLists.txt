cmake_minimum_required(VERSION 2.8.7)
project(TDL)
include(CTest)
include(cmake/FindLLVM.cmake)

# FLEX/YACC
find_package(BISON)
find_package(FLEX)
BISON_TARGET(Parser ${CMAKE_CURRENT_SOURCE_DIR}/include/parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp)
FLEX_TARGET(Lexer ${CMAKE_CURRENT_SOURCE_DIR}/include/scanner.l ${CMAKE_CURRENT_BINARY_DIR}/scanner.cpp)
get_filename_component(BISON_Parser_INCLUDE_DIRECTORIES ${BISON_Parser_OUTPUT_HEADER} DIRECTORY)
include_directories(${BISON_Parser_INCLUDE_DIRECTORIES})

#Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Gather headers for cmake-based IDEs
file( GLOB_RECURSE ALL_SRC *.cpp *.hpp *.h *.py)
add_custom_target( ALL SOURCES ${ALL_SRC} )

# Compiler flags
link_directories(/home/philippe/Development/llvm-tlvm/build/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(/home/philippe/Development/llvm-tlvm/include)
include_directories(/home/philippe/Development/llvm-tlvm/build/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Library
file(GLOB_RECURSE LIBTDL_SRC lib/*.cpp)
add_library(tdl SHARED ${LIBTDL_SRC} ${BISON_Parser_OUTPUTS} ${FLEX_Lexer_OUTPUTS})
target_link_libraries(tdl "dl" ${LLVM_LIBRARIES})

# Examples
add_subdirectory(examples)





