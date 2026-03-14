set(_driver_sources
  Device.cpp
  GPU/CudaApi.cpp
  GPU/CuptiApi.cpp
  GPU/HipApi.cpp
  GPU/HsaApi.cpp
  GPU/NvtxApi.cpp
)

if(PROTON_BUILD_ROCPROFILER_SDK)
  list(APPEND _driver_sources
    GPU/RocprofApi.cpp
  )
endif()

add_proton_library(ProtonDriver ${_driver_sources})
