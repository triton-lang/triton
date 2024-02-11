include (${CMAKE_CURRENT_SOURCE_DIR}/googletest.cmake)

include(GoogleTest)
enable_testing()

function(add_triton_ut)
  set(options)
  set(oneValueArgs NAME)
  set(multiValueArgs SRCS LIBS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_test(NAME ${__NAME}
          COMMAND ${__NAME})
  add_executable(
          ${__NAME}
          ${__SRCS})
  target_link_libraries(
          ${__NAME}
          PRIVATE
          GTest::gtest_main
          gmock
          ${__LIBS})

  # Without the TEST_DISCOVERY_TIMEOUT, the tests randomly time out on my mac
  # laptop.  I think the issue may be that the very first time you run a program
  # it's a bit slow.
  gtest_discover_tests(${__NAME} PROPERTIES TEST_DISCOVERY_TIMEOUT 60)
endfunction()

add_subdirectory(Analysis)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
