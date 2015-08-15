# - Find python libraries
# This module finds the libraries corresponding to the Python interpeter
# FindPythonInterp provides.
# This code sets the following variables:
#
#  PYTHONLIBS_FOUND           - have the Python libs been found
#  PYTHON_PREFIX              - path to the Python installation
#  PYTHON_LIBRARIES           - path to the python library
#  PYTHON_INCLUDE_DIRS        - path to where Python.h is found
#  PYTHON_SITE_PACKAGES       - path to installation site-packages
#  PYTHON_IS_DEBUG            - whether the Python interpreter is a debug build
#
#  PYTHON_INCLUDE_PATH        - path to where Python.h is found (deprecated)
#
# A function PYTHON_ADD_MODULE(<name> src1 src2 ... srcN) is defined to build modules for python.

#=============================================================================
# Copyright 2001-2009 Kitware, Inc.
# Copyright 2012 Continuum Analytics, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# * Neither the names of Kitware, Inc., the Insight Software Consortium,
# nor the names of their contributors may be used to endorse or promote
# products derived from this software without specific prior written
# permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# # A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

# Use the Python interpreter to find the libs.
if(PythonLibsNew_FIND_REQUIRED)
    find_package(PythonInterp REQUIRED)
else()
    find_package(PythonInterp)
endif()

if(NOT PYTHONINTERP_FOUND)
    set(PYTHONLIBS_FOUND FALSE)
    return()
endif()

# According to http://stackoverflow.com/questions/646518/python-how-to-detect-debug-interpreter
# testing whether sys has the gettotalrefcount function is a reliable, cross-platform
# way to detect a CPython debug interpreter.
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "from distutils import sysconfig as s;import sys;import struct;
print('.'.join(str(v) for v in sys.version_info));
print(s.PREFIX);
print(s.get_python_inc(plat_specific=True));
print(s.get_python_lib(plat_specific=True));
print(s.get_config_var('SO'));
print(hasattr(sys, 'gettotalrefcount')+0);
print(struct.calcsize('@P'));
"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE _PYTHON_VALUES
    ERROR_VARIABLE _PYTHON_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT _PYTHON_SUCCESS MATCHES 0)
    if(PythonLibsNew_FIND_REQUIRED)
        message(FATAL_ERROR
            "Python config failure:\n${_PYTHON_ERROR_VALUE}")
    endif()
    set(PYTHONLIBS_FOUND FALSE)
    return()
endif()

# Convert the process output into a list
string(REGEX REPLACE ";" "\\\\;" _PYTHON_VALUES ${_PYTHON_VALUES})
string(REGEX REPLACE "\n" ";" _PYTHON_VALUES ${_PYTHON_VALUES})
list(GET _PYTHON_VALUES 0 _PYTHON_VERSION_LIST)
list(GET _PYTHON_VALUES 1 PYTHON_PREFIX)
list(GET _PYTHON_VALUES 2 PYTHON_INCLUDE_DIR)
list(GET _PYTHON_VALUES 3 PYTHON_SITE_PACKAGES)
list(GET _PYTHON_VALUES 4 PYTHON_MODULE_EXTENSION)
list(GET _PYTHON_VALUES 5 PYTHON_IS_DEBUG)
list(GET _PYTHON_VALUES 6 PYTHON_SIZEOF_VOID_P)

# Make sure the Python has the same pointer-size as the chosen compiler
if(NOT ${PYTHON_SIZEOF_VOID_P} MATCHES ${CMAKE_SIZEOF_VOID_P})
    if(PythonLibsNew_FIND_REQUIRED)
        math(EXPR _PYTHON_BITS "${PYTHON_SIZEOF_VOID_P} * 8")
        math(EXPR _CMAKE_BITS "${CMAKE_SIZEOF_VOID_P} * 8")
        message(FATAL_ERROR
            "Python config failure: Python is ${_PYTHON_BITS}-bit, "
            "chosen compiler is  ${_CMAKE_BITS}-bit")
    endif()
    set(PYTHONLIBS_FOUND FALSE)
    return()
endif()

# The built-in FindPython didn't always give the version numbers
string(REGEX REPLACE "\\." ";" _PYTHON_VERSION_LIST ${_PYTHON_VERSION_LIST})
list(GET _PYTHON_VERSION_LIST 0 PYTHON_VERSION_MAJOR)
list(GET _PYTHON_VERSION_LIST 1 PYTHON_VERSION_MINOR)
list(GET _PYTHON_VERSION_LIST 2 PYTHON_VERSION_PATCH)

# Make sure all directory separators are '/'
string(REGEX REPLACE "\\\\" "/" PYTHON_PREFIX ${PYTHON_PREFIX})
string(REGEX REPLACE "\\\\" "/" PYTHON_INCLUDE_DIR ${PYTHON_INCLUDE_DIR})
string(REGEX REPLACE "\\\\" "/" PYTHON_SITE_PACKAGES ${PYTHON_SITE_PACKAGES})

# TODO: All the nuances of CPython debug builds have not been dealt with/tested.
if(PYTHON_IS_DEBUG)
    set(PYTHON_MODULE_EXTENSION "_d${PYTHON_MODULE_EXTENSION}")
endif()

if(CMAKE_HOST_WIN32)
    set(PYTHON_LIBRARY
        "${PYTHON_PREFIX}/libs/Python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}.lib")
elseif(APPLE)
    set(PYTHON_LIBRARY
        "${PYTHON_PREFIX}/lib/libpython${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}.dylib")
else()
    if(${PYTHON_SIZEOF_VOID_P} MATCHES 8)
        set(_PYTHON_LIBS_SEARCH "${PYTHON_PREFIX}/lib64" "${PYTHON_PREFIX}/lib")
    else()
        set(_PYTHON_LIBS_SEARCH "${PYTHON_PREFIX}/lib")
    endif()
    # Probably this needs to be more involved. It would be nice if the config
    # information the python interpreter itself gave us were more complete.
    find_library(PYTHON_LIBRARY
        NAMES "python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
        PATHS ${_PYTHON_LIBS_SEARCH}
        NO_SYSTEM_ENVIRONMENT_PATH)
endif()

# For backward compatibility, set PYTHON_INCLUDE_PATH, but make it internal.
SET(PYTHON_INCLUDE_PATH "${PYTHON_INCLUDE_DIR}" CACHE INTERNAL
          "Path to where Python.h is found (deprecated)")

MARK_AS_ADVANCED(
  PYTHON_LIBRARY
  PYTHON_INCLUDE_DIR
)

# We use PYTHON_INCLUDE_DIR, PYTHON_LIBRARY and PYTHON_DEBUG_LIBRARY for the
# cache entries because they are meant to specify the location of a single
# library. We now set the variables listed by the documentation for this
# module.
SET(PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_DIR}")
SET(PYTHON_LIBRARIES "${PYTHON_LIBRARY}")
SET(PYTHON_DEBUG_LIBRARIES "${PYTHON_DEBUG_LIBRARY}")


# Don't know how to get to this directory, just doing something simple :P
#INCLUDE(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
#FIND_PACKAGE_HANDLE_STANDARD_ARGS(PythonLibs DEFAULT_MSG PYTHON_LIBRARIES PYTHON_INCLUDE_DIRS)
find_package_message(PYTHON
    "Found PythonLibs: ${PYTHON_LIBRARY}"
    "${PYTHON_EXECUTABLE}${PYTHON_VERSION}")


# PYTHON_ADD_MODULE(<name> src1 src2 ... srcN) is used to build modules for python.
FUNCTION(PYTHON_ADD_MODULE _NAME )
  GET_PROPERTY(_TARGET_SUPPORTS_SHARED_LIBS
    GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS)
  OPTION(PYTHON_ENABLE_MODULE_${_NAME} "Add module ${_NAME}" TRUE)
  OPTION(PYTHON_MODULE_${_NAME}_BUILD_SHARED
    "Add module ${_NAME} shared" ${_TARGET_SUPPORTS_SHARED_LIBS})

  # Mark these options as advanced
  MARK_AS_ADVANCED(PYTHON_ENABLE_MODULE_${_NAME}
    PYTHON_MODULE_${_NAME}_BUILD_SHARED)

  IF(PYTHON_ENABLE_MODULE_${_NAME})
    IF(PYTHON_MODULE_${_NAME}_BUILD_SHARED)
      SET(PY_MODULE_TYPE MODULE)
    ELSE(PYTHON_MODULE_${_NAME}_BUILD_SHARED)
      SET(PY_MODULE_TYPE STATIC)
      SET_PROPERTY(GLOBAL  APPEND  PROPERTY  PY_STATIC_MODULES_LIST ${_NAME})
    ENDIF(PYTHON_MODULE_${_NAME}_BUILD_SHARED)

    SET_PROPERTY(GLOBAL  APPEND  PROPERTY  PY_MODULES_LIST ${_NAME})
    ADD_LIBRARY(${_NAME} ${PY_MODULE_TYPE} ${ARGN})
    TARGET_LINK_LIBRARIES(${_NAME} ${PYTHON_LIBRARIES})

    IF(PYTHON_MODULE_${_NAME}_BUILD_SHARED)
      SET_TARGET_PROPERTIES(${_NAME} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
      SET_TARGET_PROPERTIES(${_NAME} PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")
    ELSE()
    ENDIF()

  ENDIF(PYTHON_ENABLE_MODULE_${_NAME})
ENDFUNCTION(PYTHON_ADD_MODULE)

