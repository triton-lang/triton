set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")
file(GLOB DEPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/pyatidlas/*.py ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(APPEND DEPS "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")

configure_file(${SETUP_PY_IN} ${SETUP_PY})
add_custom_command(OUTPUT ${OUTPUT}
                   COMMAND ${PYTHON} ${SETUP_PY} build
                   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                   DEPENDS ${DEPS})
add_custom_target(pyatidlas ALL DEPENDS ${OUTPUT})
install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
