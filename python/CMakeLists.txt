find_program(PYTHON "python")

if(PYTHON)
  add_subdirectory(pyisaac)
  add_subdirectory(autotune)
endif()
