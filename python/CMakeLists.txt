find_program(PYTHON "python")

if (PYTHON)
  add_subdirectory(pyatidlas)
  add_subdirectory(autotune)
endif()
