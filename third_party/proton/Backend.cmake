# Add a backend to the Proton build and store it's namespace so a call to it's registration
# hook can be codegenned.
function(add_proton_backend name backend_namespace)
  add_proton_library(${name} ${ARGN})
  set_property(GLOBAL APPEND PROPERTY PROTON_BACKEND_REGISTRATION_HOOKS
               ${backend_namespace})
  message(STATUS "Backend registered with namespace ${backend_namespace}")
endfunction()

# Add a new device type to the DeviceType enum.
function(add_proton_device_type device_type)
  set_property(GLOBAL APPEND PROPERTY PROTON_BACKEND_DEVICE_TYPES ${device_type})
endfunction()

# Add an external library required by a backend to the Proton build.
function(add_proton_backend_external_lib lib)
  set_property(GLOBAL APPEND PROPERTY PROTON_BACKEND_EXTERNAL_LIBS ${lib})
endfunction()

# Generate a RegisteredBackends.cpp file with calls to the registration hooks of any
# registered backends.
function(codegen_proton_backend_registry output_file)
  set(BACKEND_REGISTRATION_HOOKS "")
  set(BACKEND_REGISTRATION_HOOKS_FORWARD_DECLS "")
  foreach(_namespace IN LISTS ARGN)
    # RegisteredBackends.in substitutes these snippets into a list of calls and
    # matching forward declarations.
    string(APPEND BACKEND_REGISTRATION_HOOKS
           "${_namespace}::registerProtonBackend(),\n")
    string(APPEND BACKEND_REGISTRATION_HOOKS_FORWARD_DECLS
           "namespace ${_namespace} { proton::BackendRegistration registerProtonBackend(); }\n")
  endforeach()
  configure_file(${PROTON_SRC_DIR}/lib/Backend/RegisteredBackends.in
                 ${output_file} @ONLY)
endfunction()

# Codegen the Proton headers/sources that add registered backends and device to the build.
function(codegen_proton_backend_templates)
  # Device types
  get_property(_proton_backend_device_types GLOBAL
               PROPERTY PROTON_BACKEND_DEVICE_TYPES)
  set(BACKEND_DEVICE_TAGS "")
  foreach(_device_type IN LISTS _proton_backend_device_types)
    string(APPEND BACKEND_DEVICE_TAGS "  ${_device_type},\n")
  endforeach()
  configure_file(${PROTON_COMMON_DIR}/include/DeviceType.in
                 ${PROTON_COMMON_DIR}/include/DeviceType.h @ONLY)

  # Registration Hooks
  get_property(_proton_backend_namespaces GLOBAL
               PROPERTY PROTON_BACKEND_REGISTRATION_HOOKS)
  codegen_proton_backend_registry(
    ${PROTON_SRC_DIR}/lib/Backend/RegisteredBackends.cpp
    ${_proton_backend_namespaces}
  )
endfunction()

# Find any Triton plugins which also have a /proton directory and add them to the build.
function(discover_proton_backends)
  foreach(BACKEND_DIR ${TRITON_PLUGIN_DIRS})
    if(EXISTS ${BACKEND_DIR}/proton)
      file(READ "${BACKEND_DIR}/backend/name.conf" BACKEND_NAME)
      string(STRIP ${BACKEND_NAME} BACKEND_NAME)
      # Include the backend as part of the build, placing the build output under
      # ${PROJECT_BINARY_DIR}/third_party/${BACKEND_NAME}
      set(BACKEND_BUILD_DIR "${PROJECT_BINARY_DIR}/third_party/${BACKEND_NAME}")
      message(
        STATUS
        "Adding Proton backend '${BACKEND_NAME}' to the build with output directory ${BACKEND_BUILD_DIR}")
      add_subdirectory(${BACKEND_DIR}/proton ${BACKEND_BUILD_DIR})
    endif()
  endforeach()
endfunction()

# Link every backend-requested external dependency into the final proton shared
# library.
function(link_proton_backend_external_libs)
  get_property(_proton_external_libs GLOBAL
               PROPERTY PROTON_BACKEND_EXTERNAL_LIBS)
  if(_proton_external_libs)
    message(STATUS
            "Linking external libs into 'proton': ${_proton_external_libs}")
    target_link_libraries(proton PRIVATE ${_proton_external_libs})
  endif()
endfunction()
