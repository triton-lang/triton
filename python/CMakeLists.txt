file( GLOB_RECURSE PYTHON_SRC *.cpp)
file( GLOB_RECURSE PYTHON_PYSRC *.py)
add_custom_target( MAKE_PYTHON_SRC_VISIBLE SOURCES ${PYTHON_SRC} ${PYTHON_PYSRC})

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/external/cmake/setup.py")
set(SETUP_PY    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build")

function(JOIN VALUES GLUE OUTPUT)
  string (REGEX REPLACE "([^\\]|^);" "\\1${GLUE}" _TMP_STR "${VALUES}")
  string (REGEX REPLACE "[\\](.)" "\\1" _TMP_STR "${_TMP_STR}") #fixes escaping
  set (${OUTPUT} "${_TMP_STR}" PARENT_SCOPE)
endfunction()
join("${LIBISAAC_SRC}" " " LIBISAAC_SRC_STR)

get_property(INCLUDE_DIRECTORIES_LST DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
join("${INCLUDE_DIRECTORIES_LST}" " " INCLUDE_DIRECTORIES_STR)

configure_file(${SETUP_PY_IN} ${SETUP_PY})

add_custom_command(OUTPUT ${OUTPUT}/timestamp
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/isaac ${CMAKE_CURRENT_BINARY_DIR}/isaac
                   COMMAND ${PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py build
                   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}/timestamp
                   DEPENDS ${LIBISAAC_SRC} ${PYTHON_SRC} ${PYTHON_PYSRC} ${SETUP_PY})
add_custom_target(python DEPENDS ${OUTPUT}/timestamp)
install(CODE "execute_process(WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND ${PYTHON} ${SETUP_PY} install)" OPTIONAL)
