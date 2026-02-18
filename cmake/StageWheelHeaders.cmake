cmake_minimum_required(VERSION 3.20)

# Stage Triton C/C++ headers into the Python wheel build tree.
#
# Background:
# - `pip wheel .` invokes setuptools (`setup.py`) which builds a package staging
#   tree under `build/lib.*`.
# - During CMake configure, `setup.py` passes that staging root as
#   `TRITON_WHEEL_DIR`.
# - This script is run from a `POST_BUILD` step on target `triton` so generated
#   headers from the build tree already exist.
#
# Inputs (provided via `cmake -D... -P`):
# - `SOURCE_INCLUDE_DIR`: source tree include dir (`<repo>/include`)
# - `BINARY_INCLUDE_DIR`: generated include dir (`<build>/include`)
# - `BINARY_THIRD_PARTY_DIR`: generated third-party dir (`<build>/third_party`)
# - `WHEEL_TRITON_DIR`: destination package dir (`${TRITON_WHEEL_DIR}/triton`)
#
# Copy policy:
# - Copy only header-like artifacts needed by downstream C/C++ consumers.
# - Include: `*.h`, `*.hpp`, `*.h.inc`, `*.inc`
# - Exclude by omission: `*.cpp.inc` and other non-header files.

if(NOT DEFINED WHEEL_TRITON_DIR)
  message(FATAL_ERROR "WHEEL_TRITON_DIR must be defined")
endif()

if(NOT DEFINED SOURCE_INCLUDE_DIR)
  message(FATAL_ERROR "SOURCE_INCLUDE_DIR must be defined")
endif()

file(REMOVE_RECURSE "${WHEEL_TRITON_DIR}/include" "${WHEEL_TRITON_DIR}/third_party")

file(COPY "${SOURCE_INCLUDE_DIR}/"
     DESTINATION "${WHEEL_TRITON_DIR}/include"
     FILES_MATCHING
     PATTERN "*.h"
     PATTERN "*.h.inc"
     PATTERN "*.hpp"
     PATTERN "*.hpp.inc")

if(DEFINED BINARY_INCLUDE_DIR AND EXISTS "${BINARY_INCLUDE_DIR}")
  file(COPY "${BINARY_INCLUDE_DIR}/"
       DESTINATION "${WHEEL_TRITON_DIR}/include"
       FILES_MATCHING
       PATTERN "*.h"
       PATTERN "*.h.inc"
       PATTERN "*.hpp"
       PATTERN "*.hpp.inc")
endif()

if(DEFINED BINARY_THIRD_PARTY_DIR AND EXISTS "${BINARY_THIRD_PARTY_DIR}")
  file(COPY "${BINARY_THIRD_PARTY_DIR}/"
       DESTINATION "${WHEEL_TRITON_DIR}/third_party"
       FILES_MATCHING
       PATTERN "*.h"
       PATTERN "*.hpp"
       PATTERN "*.h.inc"
       PATTERN "*.hpp.inc")
endif()
