include(FindPackageHandleStandardArgs)
execute_process(COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__))"
                OUTPUT_VARIABLE TORCH_INSTALL_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

find_package_handle_standard_args(TORCH DEFAULT_MSG TORCH_INSTALL_PREFIX)
if(TORCH_INSTALL_PREFIX)
  set(TORCH_INCLUDE_DIRS ${TORCH_INSTALL_PREFIX}/lib/include/ 
						 ${TORCH_INSTALL_PREFIX}/lib/include/torch/csrc/api/include
						 ${TORCH_INSTALL_PREFIX}/include/
						 ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include/)
  set(TORCH_LIBRARY_DIRS ${TORCH_INSTALL_PREFIX}/lib/)
endif()

mark_as_advanced(TORCH_INCLUDE_DIRS TORCH_LIBRARY_DIRS)
