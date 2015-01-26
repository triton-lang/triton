find_program(PYTHON "python")

get_property(LIBATIDLAS_PATH TARGET atidlas PROPERTY LOCATION)
get_filename_component(LIBATIDLAS_DIR ${LIBATIDLAS_PATH} DIRECTORY)

if(PYTHON)
  add_subdirectory(pyatidlas)
  add_subdirectory(autotune)
endif()
