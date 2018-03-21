cmake_minimum_required(VERSION 2.8.7)
project(isaac)
include(CTest)

#Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

#QtCreator: add visibility of headers
file( GLOB_RECURSE ALL_SRC *.cpp *.hpp *.h *.py)
add_custom_target( ALL SOURCES ${ALL_SRC} )

#Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/isaac/external/CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -Wno-strict-aliasing")

#Source
file(GLOB_RECURSE LIBISAAC_SRC lib/*.cpp)
add_library(isaac SHARED ${LIBISAAC_SRC})
target_link_libraries(isaac "dl")

#Examples
add_subdirectory(examples)

#Tests
add_subdirectory(tests)
