cmake_minimum_required(VERSION 2.8.7)
project(isaac-research)
include(CTest)

#QtCreator: add visibility of headers
file( GLOB_RECURSE MAKE_HEADERS_VISIBLE_SRC *.cpp *.hpp *.h)
add_custom_target( MAKE_HEADERS_VISIBLE SOURCES ${MAKE_HEADERS_VISIBLE_SRC} )

#Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

#Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic")

#Source
file(GLOB_RECURSE LIBISAAC_SRC lib/*.cpp)
add_library(isaac SHARED ${LIBISAAC_SRC})
target_link_libraries(isaac "dl")

#Examples
add_subdirectory(examples)

#Tests
add_subdirectory(tests)
