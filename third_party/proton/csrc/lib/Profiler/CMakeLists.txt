set(_profiler_sources
  Profiler.cpp
  GPUProfiler.cpp
  Graph.cpp
  Cupti/CuptiPCSampling.cpp
  Cupti/CuptiProfiler.cpp
  Instrumentation/InstrumentationProfiler.cpp
  Instrumentation/Metadata.cpp
)

if(PROTON_BUILD_ROCPROFILER_SDK)
  list(APPEND _profiler_sources
    RocprofSDK/RocprofSDKProfiler.cpp
  )
else()
  list(APPEND _profiler_sources
    RocprofSDK/RocprofSDKProfilerStub.cpp
  )
endif()

add_proton_library(ProtonProfiler ${_profiler_sources})
