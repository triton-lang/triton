# Create testing backend
add_library(TestBackend OBJECT TestBackend.cpp)
target_include_directories(TestBackend
  PRIVATE
    "${PROTON_COMMON_DIR}/include"
    "${PROTON_SRC_DIR}/include"
)

# Create testing backend registry
codegen_proton_backend_registry(
  "${CMAKE_CURRENT_BINARY_DIR}/RegisteredBackends.cpp"
  TestBackend
)

add_library(TestBackendRegistry OBJECT
  "${CMAKE_CURRENT_BINARY_DIR}/RegisteredBackends.cpp"
)
target_include_directories(TestBackendRegistry
  PRIVATE
    "${PROTON_COMMON_DIR}/include"
    "${PROTON_SRC_DIR}/include"
)

# Add test
add_triton_ut(
  NAME BackendRegistration
  SRCS BackendRegistrationTest.cpp
  LIBS TestBackend TestBackendRegistry
)

target_include_directories(BackendRegistration
  PRIVATE
    "${PROTON_COMMON_DIR}/include"
    "${PROTON_SRC_DIR}/include"
)
