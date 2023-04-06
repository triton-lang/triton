# - Find LLVM headers and libraries.
# This module locates LLVM and adapts the llvm-config output for use with
# CMake.
#
# A given list of COMPONENTS is passed to llvm-config.
#
# The following variables are defined:
#  LLVM_FOUND          - true if LLVM was found
#  LLVM_CXXFLAGS       - C++ compiler flags for files that include LLVM headers.
#  LLVM_ENABLE_ASSERTIONS - Whether LLVM was built with enabled assertions (ON/OFF).
#  LLVM_INCLUDE_DIRS   - Directory containing LLVM include files.
#  LLVM_IS_SHARED      - Whether LLVM is going to be linked dynamically (ON) or statically (OFF).
#  LLVM_LDFLAGS        - Linker flags to add when linking against LLVM
#                        (includes -LLLVM_LIBRARY_DIRS).
#  LLVM_LIBRARIES      - Full paths to the library files to link against.
#  LLVM_LIBRARY_DIRS   - Directory containing LLVM libraries.
#  LLVM_NATIVE_ARCH    - Backend corresponding to LLVM_HOST_TARGET, e.g.,
#                        X86 for x86_64 and i686 hosts.
#  LLVM_ROOT_DIR       - The root directory of the LLVM installation.
#                        llvm-config is searched for in ${LLVM_ROOT_DIR}/bin.
#  LLVM_TARGETS_TO_BUILD - List of built LLVM targets.
#  LLVM_VERSION_MAJOR  - Major version of LLVM.
#  LLVM_VERSION_MINOR  - Minor version of LLVM.
#  LLVM_VERSION_STRING - Full LLVM version string (e.g. 6.0.0svn).
#  LLVM_VERSION_BASE_STRING - Base LLVM version string without git/svn suffix (e.g. 6.0.0).
#
# Note: The variable names were chosen in conformance with the official CMake
# guidelines, see ${CMAKE_ROOT}/Modules/readme.txt.

# Try suffixed versions to pick up the newest LLVM install available on Debian
# derivatives.
# We also want an user-specified LLVM_ROOT_DIR to take precedence over the
# system default locations such as /usr/local/bin. Executing find_program()
# multiples times is the approach recommended in the docs.
set(llvm_config_names llvm-config-6.0 llvm-config60
                      llvm-config)
foreach(v RANGE 7 17)
    # names like llvm-config-7.0 llvm-config70 llvm-config-7 llvm-config-7-64
    list(PREPEND llvm_config_names llvm-config-${v}.0 llvm-config${v}0 llvm-config-${v} llvm-config-${v}-64)
endforeach()
find_program(LLVM_CONFIG
    NAMES ${llvm_config_names}
    PATHS ${LLVM_ROOT_DIR}/bin NO_DEFAULT_PATH
    DOC "Path to llvm-config tool.")
find_program(LLVM_CONFIG NAMES ${llvm_config_names})
if(APPLE)
    # extra fallbacks for MacPorts & Homebrew
    find_program(LLVM_CONFIG
        NAMES ${llvm_config_names}
        PATHS /opt/local/libexec/llvm-11/bin  /opt/local/libexec/llvm-10/bin  /opt/local/libexec/llvm-9.0/bin
              /opt/local/libexec/llvm-8.0/bin /opt/local/libexec/llvm-7.0/bin /opt/local/libexec/llvm-6.0/bin
              /opt/local/libexec/llvm/bin
              /usr/local/opt/llvm@11/bin /usr/local/opt/llvm@10/bin /usr/local/opt/llvm@9/bin
              /usr/local/opt/llvm@8/bin  /usr/local/opt/llvm@7/bin  /usr/local/opt/llvm@6/bin
              /usr/local/opt/llvm/bin
        NO_DEFAULT_PATH)
endif()

# Prints a warning/failure message depending on the required/quiet flags. Copied
# from FindPackageHandleStandardArgs.cmake because it doesn't seem to be exposed.
macro(_LLVM_FAIL _msg)
  if(LLVM_FIND_REQUIRED)
    message(FATAL_ERROR "${_msg}")
  else()
    if(NOT LLVM_FIND_QUIETLY)
      message(WARNING "${_msg}")
    endif()
  endif()
endmacro()


if(NOT LLVM_CONFIG)
    if(NOT LLVM_FIND_QUIETLY)
        _LLVM_FAIL("No LLVM installation (>= ${LLVM_FIND_VERSION}) found. Try manually setting the 'LLVM_ROOT_DIR' or 'LLVM_CONFIG' variables.")
    endif()
else()
    macro(llvm_set var flag)
       if(LLVM_FIND_QUIETLY)
            set(_quiet_arg ERROR_QUIET)
        endif()
        set(result_code)
        execute_process(
            COMMAND ${LLVM_CONFIG} --link-static --${flag}
            RESULT_VARIABLE result_code
            OUTPUT_VARIABLE LLVM_${var}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ${_quiet_arg}
        )
        if(result_code)
            _LLVM_FAIL("Failed to execute llvm-config ('${LLVM_CONFIG}', result code: '${result_code})'")
        else()
            if(${ARGV2})
                file(TO_CMAKE_PATH "${LLVM_${var}}" LLVM_${var})
            endif()
        endif()
    endmacro()
    macro(llvm_set_libs var flag components)
       if(LLVM_FIND_QUIETLY)
            set(_quiet_arg ERROR_QUIET)
        endif()
        set(result_code)
        execute_process(
            COMMAND ${LLVM_CONFIG} --link-static --${flag} ${components}
            RESULT_VARIABLE result_code
            OUTPUT_VARIABLE tmplibs
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ${_quiet_arg}
        )
        if(result_code)
            _LLVM_FAIL("Failed to execute llvm-config ('${LLVM_CONFIG}', result code: '${result_code})'")
        else()
            file(TO_CMAKE_PATH "${tmplibs}" tmplibs)
            string(REGEX MATCHALL "${pattern}[^ ]+" LLVM_${var} ${tmplibs})
        endif()
    endmacro()

    llvm_set(VERSION_STRING version)
    llvm_set(CXXFLAGS cxxflags)
    llvm_set(INCLUDE_DIRS includedir true)
    llvm_set(ROOT_DIR prefix true)
    llvm_set(ENABLE_ASSERTIONS assertion-mode)

    # The LLVM version string _may_ contain a git/svn suffix, so match only the x.y.z part
    string(REGEX MATCH "^[0-9]+[.][0-9]+[.][0-9]+" LLVM_VERSION_BASE_STRING "${LLVM_VERSION_STRING}")

    llvm_set(SHARED_MODE shared-mode)
    if(LLVM_SHARED_MODE STREQUAL "shared")
        set(LLVM_IS_SHARED ON)
    else()
        set(LLVM_IS_SHARED OFF)
    endif()

    llvm_set(LDFLAGS ldflags)
    llvm_set(SYSTEM_LIBS system-libs)
    string(REPLACE "\n" " " LLVM_LDFLAGS "${LLVM_LDFLAGS} ${LLVM_SYSTEM_LIBS}")
    if(APPLE) # unclear why/how this happens
        string(REPLACE "-llibxml2.tbd" "-lxml2" LLVM_LDFLAGS ${LLVM_LDFLAGS})
    endif()

    llvm_set(LIBRARY_DIRS libdir true)
    llvm_set_libs(LIBRARIES libfiles "${LLVM_FIND_COMPONENTS}")
    # LLVM bug: llvm-config --libs tablegen returns -lLLVM-3.8.0
    # but code for it is not in shared library
    if("${LLVM_FIND_COMPONENTS}" MATCHES "tablegen")
        if (NOT "${LLVM_LIBRARIES}" MATCHES "LLVMTableGen")
            set(LLVM_LIBRARIES "${LLVM_LIBRARIES};-lLLVMTableGen")
        endif()
    endif()

    llvm_set(CMAKEDIR cmakedir)
    llvm_set(TARGETS_TO_BUILD targets-built)
    string(REGEX MATCHALL "${pattern}[^ ]+" LLVM_TARGETS_TO_BUILD ${LLVM_TARGETS_TO_BUILD})

    # Parse LLVM_NATIVE_ARCH manually from LLVMConfig.cmake; including it leads to issues like
    # https://github.com/ldc-developers/ldc/issues/3079.
    file(STRINGS "${LLVM_CMAKEDIR}/LLVMConfig.cmake" LLVM_NATIVE_ARCH LIMIT_COUNT 1 REGEX "^set\\(LLVM_NATIVE_ARCH (.+)\\)$")
    string(REGEX MATCH "set\\(LLVM_NATIVE_ARCH (.+)\\)" LLVM_NATIVE_ARCH "${LLVM_NATIVE_ARCH}")
    set(LLVM_NATIVE_ARCH ${CMAKE_MATCH_1})
    message(STATUS "LLVM_NATIVE_ARCH: ${LLVM_NATIVE_ARCH}")

    # On CMake builds of LLVM, the output of llvm-config --cxxflags does not
    # include -fno-rtti, leading to linker errors. Be sure to add it.
    if(NOT MSVC AND (CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")))
        if(NOT ${LLVM_CXXFLAGS} MATCHES "-fno-rtti")
            set(LLVM_CXXFLAGS "${LLVM_CXXFLAGS} -fno-rtti")
        endif()
    endif()

    # Remove some clang-specific flags for gcc.
    if(CMAKE_COMPILER_IS_GNUCXX)
        string(REPLACE "-Wcovered-switch-default " "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
        string(REPLACE "-Wstring-conversion " "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
        string(REPLACE "-fcolor-diagnostics " "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
        # this requires more recent gcc versions (not supported by 4.9)
        string(REPLACE "-Werror=unguarded-availability-new " "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
    endif()

    # Remove gcc-specific flags for clang.
    if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        string(REPLACE "-Wno-maybe-uninitialized " "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
    endif()

    string(REGEX REPLACE "([0-9]+).*" "\\1" LLVM_VERSION_MAJOR "${LLVM_VERSION_STRING}" )
    string(REGEX REPLACE "[0-9]+\\.([0-9]+).*[A-Za-z]*" "\\1" LLVM_VERSION_MINOR "${LLVM_VERSION_STRING}" )

    if (${LLVM_VERSION_STRING} VERSION_LESS ${LLVM_FIND_VERSION})
        _LLVM_FAIL("Unsupported LLVM version ${LLVM_VERSION_STRING} found (${LLVM_CONFIG}). At least version ${LLVM_FIND_VERSION} is required. You can also set variables 'LLVM_ROOT_DIR' or 'LLVM_CONFIG' to use a different LLVM installation.")
    endif()
endif()

# Use the default CMake facilities for handling QUIET/REQUIRED.
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(LLVM
    REQUIRED_VARS LLVM_ROOT_DIR
    VERSION_VAR LLVM_VERSION_STRING)
