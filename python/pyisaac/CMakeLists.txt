file( GLOB_RECURSE PYISAAC_SRC *.cpp)
file( GLOB_RECURSE PYISAAC_PYSRC *.py)
add_custom_target( MAKE_PYISAAC_SRC_VISIBLE SOURCES ${PYISAAC_SRC} ${PYISAAC_PYSRC})

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")
configure_file(${SETUP_PY_IN} ${SETUP_PY})
add_custom_command(OUTPUT ${OUTPUT}
                   COMMAND ${PYTHON} ${SETUP_PY} build
                   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                   DEPENDS isaac ${PYISAAC_SRC} ${PYISAAC_PYSRC} ${SETUP_PY})
add_custom_target(pyisaac ALL DEPENDS ${OUTPUT})
#install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
