#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "GTest::gtest" for configuration "RelWithDebInfo"
set_property(TARGET GTest::gtest APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(GTest::gtest PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libgtest.a"
  )

list(APPEND _cmake_import_check_targets GTest::gtest )
list(APPEND _cmake_import_check_files_for_GTest::gtest "${_IMPORT_PREFIX}/lib/libgtest.a" )

# Import target "GTest::gtest_main" for configuration "RelWithDebInfo"
set_property(TARGET GTest::gtest_main APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(GTest::gtest_main PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libgtest_main.a"
  )

list(APPEND _cmake_import_check_targets GTest::gtest_main )
list(APPEND _cmake_import_check_files_for_GTest::gtest_main "${_IMPORT_PREFIX}/lib/libgtest_main.a" )

# Import target "GTest::gmock" for configuration "RelWithDebInfo"
set_property(TARGET GTest::gmock APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(GTest::gmock PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libgmock.a"
  )

list(APPEND _cmake_import_check_targets GTest::gmock )
list(APPEND _cmake_import_check_files_for_GTest::gmock "${_IMPORT_PREFIX}/lib/libgmock.a" )

# Import target "GTest::gmock_main" for configuration "RelWithDebInfo"
set_property(TARGET GTest::gmock_main APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(GTest::gmock_main PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libgmock_main.a"
  )

list(APPEND _cmake_import_check_targets GTest::gmock_main )
list(APPEND _cmake_import_check_files_for_GTest::gmock_main "${_IMPORT_PREFIX}/lib/libgmock_main.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
