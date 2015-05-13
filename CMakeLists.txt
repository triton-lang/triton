cmake_minimum_required(VERSION 2.8.10)

# Add visibility of headers
file( GLOB_RECURSE MAKE_HEADERS_VISIBLE_SRC *.hpp *.h)
add_custom_target( MAKE_HEADERS_VISIBLE SOURCES ${MAKE_HEADERS_VISIBLE_SRC} )

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(CUDA QUIET)
find_package(OpenCL QUIET REQUIRED)
if(CUDA_FOUND)
   set(BACKEND_DEFINES "-DISAAC_WITH_CUDA")
   include_directories(${CUDA_INCLUDE_DIRS})
endif()

add_definitions(-Wno-sign-compare -D__CL_ENABLE_EXCEPTIONS ${BACKEND_DEFINES} -Wall -Wextra -pedantic -std=c++11)
add_executable(bin2cpp ${CMAKE_MODULE_PATH}/helpers/bin2cpp.cpp)

file(GLOB_RECURSE LIBISAAC_SRC lib/*.cpp)

#Python wrapper
set(SETUP_PY_IN "${CMAKE_MODULE_PATH}/python/setup.py")
set(SETUP_PY    "${CMAKE_SOURCE_DIR}/python/setup.py")

set(LIBISAAC_SRC_STR)
foreach(FILE ${LIBISAAC_SRC})
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "src" _TMP ${FILE})
    set(LIBISAAC_SRC_STR "${_TMP} ${LIBISAAC_SRC_STR}")
endforeach()

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
