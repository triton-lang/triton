if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ArgumentBindingTest.cpp")
  add_triton_ut(
    NAME ArgumentBinding
    SRCS ArgumentBindingTest.cpp
    LIBS
      TritonTools
  )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/RuntimeTest.cpp")
  add_triton_ut(
    NAME RuntimeTest
    SRCS RuntimeTest.cpp
    LIBS
      TritonTools
  )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/PipelineConcurrencyTest.cpp")
  add_triton_ut(
    NAME PipelineConcurrency
    SRCS PipelineConcurrencyTest.cpp
    LIBS
      TritonTools
  )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/DriverTest.cpp")
  if(TARGET Python3::Module)
    add_triton_ut(
      NAME DriverTest
      SRCS DriverTest.cpp
      LIBS
        TritonTools
        Python3::Module
    )
  else()
    add_triton_ut(
      NAME DriverTest
      SRCS DriverTest.cpp
      LIBS
        TritonTools
    )
  endif()
endif()
