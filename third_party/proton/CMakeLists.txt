project(Proton LANGUAGES CXX)

# ============ Check for includes =============
if(NOT CUPTI_INCLUDE_DIR)
  message(FATAL_ERROR "CUPTI include directory not defined")
endif()
if(NOT ROCTRACER_INCLUDE_DIR)
  message(FATAL_ERROR "ROCTRACER include directory not defined")
endif()
if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "JSON include directory not defined")
endif()

# ============ Dependencies =============
find_package(Python3 REQUIRED Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

# ============ Setup a list for object libraries ============
set(PROTON_OBJ_LIBS "")

function(add_proton_library name)
  add_library(${name} OBJECT ${ARGN})

  target_link_libraries(${name} PRIVATE pybind11::headers)

  target_include_directories(${name}
    PRIVATE
      ${CUPTI_INCLUDE_DIR}
      ${ROCTRACER_INCLUDE_DIR}
      ${JSON_INCLUDE_DIR}
      ${PROJECT_SOURCE_DIR}/csrc/include
  )

  target_compile_definitions(${name} PRIVATE __HIP_PLATFORM_AMD__)

  list(APPEND PROTON_OBJ_LIBS ${name})
  set(PROTON_OBJ_LIBS "${PROTON_OBJ_LIBS}" PARENT_SCOPE)
endfunction()

add_subdirectory(${PROJECT_SOURCE_DIR}/csrc)

# ============ Possibly handle Mac suffixes ============
if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  set(PROTON_PYTHON_LDFLAGS "-undefined dynamic_lookup")
endif()

# ============ Create final SHARED library from object libs ============
if(NOT PROTON_OBJ_LIBS)
  message(WARNING "No object libraries were defined!")
endif()

set(PROTON_OBJ_SOURCES "")
foreach(_lib IN LISTS PROTON_OBJ_LIBS)
  list(APPEND PROTON_OBJ_SOURCES $<TARGET_OBJECTS:${_lib}>)
  message(STATUS "Collecting object files from ${_lib}")
endforeach()

add_library(proton SHARED ${PROTON_OBJ_SOURCES})

target_link_libraries(proton PRIVATE Python3::Module)
if(PROTON_PYTHON_LDFLAGS)
  target_link_options(proton PRIVATE ${PROTON_PYTHON_LDFLAGS})
endif()
