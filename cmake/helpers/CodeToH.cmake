#Copyright (c) 2014, ArrayFire
#All rights reserved.

# Function to turn an OpenCL source file into a C string within a source file.
# xxd uses its input's filename to name the string and its length, so we
# need to move them to a name that depends only on the path output, not its
# input.  Otherwise, builds in different relative locations would put the
# source into different variable names, and everything would fall over.
# The actual name will be filename (.s replaced with underscores), and length
# name_len.
#
# Usage example:
#
# set(KERNELS a.cl b/c.cl)
# resource_to_cxx_source(
#   SOURCES ${KERNELS}
#   VARNAME OUTPUTS
# )
# add_executable(foo ${OUTPUTS})
#
# The namespace they are placed in is taken from filename.namespace.
#
# For example, if the input file is kernel.cl, the two variables will be
#  unsigned char ns::kernel_cl[];
#  unsigned int ns::kernel_cl_len;
#
# where ns is the contents of kernel.cl.namespace.

include(CMakeParseArguments)

set(BIN2CPP_PROGRAM "bin2cpp")

function(CODE_TO_H)
    cmake_parse_arguments(ARGS "" "VARNAME;EXTENSION;OUTPUT_DIR;TARGET;NAMESPACE;EOF" "SOURCES" ${ARGN})

    set(_output_files "")
    foreach(_input_file ${ARGS_SOURCES})
        get_filename_component(_path "${_input_file}" PATH)
        get_filename_component(_name "${_input_file}" NAME)
        get_filename_component(_name_we "${_input_file}" NAME_WE)
        set(var_name ${_name_we})

        set(_namespace "${ARGS_NAMESPACE}")
        string(REPLACE "." "_" var_name ${var_name})

        set(_output_path "${ARGS_OUTPUT_DIR}")
        set(_output_file "${_output_path}/${_name_we}.${ARGS_EXTENSION}")

        add_custom_command(
            OUTPUT ${_output_file}
            DEPENDS ${_input_file} ${BIN2CPP_PROGRAM}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_output_path}"
            COMMAND ${CMAKE_COMMAND} -E echo "\\#include \\<${_path}/${_name_we}.hpp\\>"  >>"${_output_file}"
            COMMAND ${BIN2CPP_PROGRAM} --file ${_name} --namespace ${_namespace} --output ${_output_file} --name ${var_name} --eof ${ARGS_EOF} --extension ${ARGS_EXTENSION}
            WORKING_DIRECTORY "${_path}"
            COMMENT "Compiling ${_input_file} to C++ source"
        )
        list(APPEND _output_files ${_output_file})
    endforeach()
    add_custom_target(${ARGS_TARGET} ALL DEPENDS ${_output_files})
endfunction()
