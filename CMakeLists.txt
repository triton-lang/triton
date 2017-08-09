cmake_minimum_required(VERSION 2.8.7)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(WIN32)
 SET(CMAKE_FIND_LIBRARY_PREFIXES "")
 SET(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
endif()
# Add visibility of headers
file( GLOB_RECURSE MAKE_HEADERS_VISIBLE_SRC *.cpp *.hpp *.h)
add_custom_target( MAKE_HEADERS_VISIBLE SOURCES ${MAKE_HEADERS_VISIBLE_SRC} )

#Modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/lib/external/)

#Compiler flags
add_definitions(${BACKEND_DEFINES})
if(WIN32)
    add_definitions("-DNOMINMAX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic")
endif()

#Includes
FIND_PACKAGE(DLFCN REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/tools/ ${CMAKE_CURRENT_SOURCE_DIR}/include/external/ ${CMAKE_CURRENT_SOURCE_DIR}/include/external/cuda ${DLFCN_INCLUDE_DIR})

#Binary to convert .cu files to const char *
if(NOT ANDROID)
    add_executable(bin2cpp ${CMAKE_MODULE_PATH}/helpers/bin2cpp.cpp)
    include("${CMAKE_MODULE_PATH}/helpers/CodeToH.cmake")
endif()

#Source files
file(GLOB_RECURSE LIBISAAC_SRC lib/*.cpp)

#Python wrapper
set(SETUP_PY_IN "${CMAKE_MODULE_PATH}/python/setup.py")
set(SETUP_PY    "${CMAKE_SOURCE_DIR}/python/setup.py")

set(LIBISAAC_SRC_STR)
foreach(FILE ${LIBISAAC_SRC})
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "src" _TMP ${FILE})
    set(LIBISAAC_SRC_STR "${_TMP} ${LIBISAAC_SRC_STR}")
endforeach()

#Include directories
set(INCLUDE_DIRECTORIES_STR)
get_property(INCLUDE_DIRECTORIES_LST DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
set(INCLUDE_DIRECTORIES_STR)
foreach(FILE ${INCLUDE_DIRECTORIES_LST})
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "src" _TMP ${FILE})
    set(INCLUDE_DIRECTORIES_STR "${INCLUDE_DIRECTORIES_STR} ${_TMP}")
endforeach()

configure_file(${SETUP_PY_IN} ${SETUP_PY})

add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/build/timestamp"
                    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/python ${CMAKE_BINARY_DIR}/python
                    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/python/src/lib/CMakeLists.txt

                    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/python/build
                    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/python/src/lib
                    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/python/src/include

                    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/lib ${CMAKE_BINARY_DIR}/python/src/lib
                    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/python/src/include

                    COMMAND ${CMAKE_COMMAND} -E tar czf isaac-1.0.tar.gz ${CMAKE_BINARY_DIR}/python
                    )

add_custom_target(package-python DEPENDS "${CMAKE_BINARY_DIR}/build/timestamp")


#Isaac
include(CTest)

add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(bench)
add_subdirectory(examples)
