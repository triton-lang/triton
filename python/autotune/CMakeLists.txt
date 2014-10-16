find_program(PYINSTALLER pyinstaller)

if(PYINSTALLER)

set(SPEC_IN "${CMAKE_CURRENT_SOURCE_DIR}/pyinstaller_build.spec")
set(SPEC    "${CMAKE_CURRENT_BINARY_DIR}/pyinstaller_build.spec")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")
file(GLOB DEPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/pysrc/*.py")
LIST(APPEND DEPS "${CMAKE_CURRENT_SOURCE_DIR}/pyinstaller_build.spec")

configure_file(${SPEC_IN} ${SPEC})
add_custom_command(OUTPUT ${OUTPUT}
                   COMMAND ${PYINSTALLER} ${SPEC_IN} ${CMAKE_CURRENT_SOURCE_DIR}
                   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                   DEPENDS ${DEPS} pyatidlas)
add_custom_target(autotune ALL DEPENDS ${OUTPUT})

endif()
