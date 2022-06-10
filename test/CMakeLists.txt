llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCe_DIR}/lit.cfg.py
)

set(TRITON_TEST_DEPENDS
  FileCheck count not
  triton-opt
)

add_lit_testsuite(check-triton "Running the triton regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${STANDALONE_TEST_DEPENDS}
  )
set_target_properties(check-triton PROPERTIES FOLDER "Tests")

add_lit_testsuites(TRITON ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TRITON_TEST_DEPENDS})
