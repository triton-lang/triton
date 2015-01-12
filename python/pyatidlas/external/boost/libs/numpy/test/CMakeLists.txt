project(BoostNumpyTests)

if (WIN32)
  set(runCmakeTest runCmakeTest.bat)
  foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
    message( STATUS "configuring runCmakeTest for cfg=${cfg}" )
    CONFIGURE_FILE( ${runCmakeTest}.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${cfg}/${runCmakeTest} @ONLY )
  endforeach()
else()
  set(runCmakeTest runCmakeTest.sh)
  CONFIGURE_FILE( ${runCmakeTest}.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${runCmakeTest} @ONLY )
endif()

set( TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR} )
set( TestCommand ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${runCmakeTest} )

# custom macro with most of the redundant code for making a python test
macro( addPythonTest _name _srcpy )
  # make the pyd library link against boost_numpy python and boost
  TARGET_LINK_LIBRARIES(${_name} boost_numpy ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
  
  # make a test of the module using the python source file in the test directory
  ADD_TEST(${_name} ${TestCommand} ${TEST_SOURCE_DIR}/${_srcpy})

  # set the regex to use to recognize a failure since `python testfoo.py`
  # does not seem to return non-zero with a test failure
  set_property(TEST ${_name} PROPERTY FAIL_REGULAR_EXPRESSION "ERROR\\:" "ImportError\\: DLL load failed\\: " )

  # put the test target into a VS solution folder named test (should
  # be a no-op for Linux)
  SET_PROPERTY(TARGET ${_name} PROPERTY FOLDER "test")
endmacro()

PYTHON_ADD_MODULE(dtype_mod dtype_mod.cpp)
addPythonTest( dtype_mod dtype.py )

PYTHON_ADD_MODULE(indexing_mod indexing_mod.cpp)
addPythonTest( indexing_mod indexing.py )

PYTHON_ADD_MODULE(ndarray_mod ndarray_mod.cpp)
addPythonTest( ndarray_mod ndarray.py )

PYTHON_ADD_MODULE(shapes_mod shapes_mod.cpp)
addPythonTest( shapes_mod shapes.py )

PYTHON_ADD_MODULE(templates_mod templates_mod.cpp)
addPythonTest( templates_mod templates.py )

PYTHON_ADD_MODULE(ufunc_mod ufunc_mod.cpp)
addPythonTest( ufunc_mod ufunc.py )

# installation logic (skip until it is better thought out)
# set(DEST_TEST boost.numpy/test)
#
# # copy the extension modules to DEST_TEST
# install(TARGETS dtype_mod indexing_mod ndarray_mod shapes_mod templates_mod ufunc_mod LIBRARY
#   DESTINATION ${DEST_TEST}
#   ${INSTALL_PERMSSIONS_RUNTIME}
#   )
#
# # copy the source test python modules to DEST_TEST too
# install(FILES dtype.py indexing.py ndarray.py shapes.py templates.py ufunc.py
#   DESTINATION ${DEST_TEST}
#   ${INSTALL_PERMSSIONS_SRC}
#   )
