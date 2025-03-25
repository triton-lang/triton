function(add_xpu_libdevice OUTPUT SRC ARCH)
  set(CLANG ${LLVM_TOOLS_BINARY_DIR}/clang)

  get_filename_component(OUTPUT_NAME ${OUTPUT} NAME_WE)

  add_custom_target(
    libdevice-${ARCH} ALL
    COMMAND ${CLANG} --xpu-arch=${ARCH} ${SRC} -c -emit-llvm --xpu-device-only -O3 -o ${OUTPUT} -std=c++11 -Wno-literal-range
    DEPENDS ${SRC}
    COMMENT "Building libdevice-${ARCH} ..."
    VERBATIM
  )
endfunction()

set(XPU_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/trigonometric.xpu)
set(XPU3_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../backend/xpu3/lib/)
file(MAKE_DIRECTORY ${XPU3_LIB_DIR})
add_xpu_libdevice(${XPU3_LIB_DIR}/libdevice-xpu3.bc ${XPU_LIB_SRC} xpu3)
