file( GLOB_RECURSE PYATIDLAS_SRC *.cpp)
file( GLOB_RECURSE PYATIDLAS_PYSRC *.py)
add_custom_target( MAKE_PYATIDLAS_SRC_VISIBLE SOURCES ${PYATIDLAS_SRC} ${PYATIDLAS_PYSRC})

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")
configure_file(${SETUP_PY_IN} ${SETUP_PY})
add_custom_command(OUTPUT ${OUTPUT}
                   COMMAND ${PYTHON} ${SETUP_PY} build
                   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                   DEPENDS atidlas ${PYATIDLAS_SRC} ${PYATIDLAS_PYSRC} ${SETUP_PY})
add_custom_target(pyatidlas ALL DEPENDS ${OUTPUT})
install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
