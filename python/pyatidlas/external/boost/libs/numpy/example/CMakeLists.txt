# custom macro with most of the redundant code for making a python example module
macro( addPythonExe _name _srccpp )
  ADD_EXECUTABLE(${_name} ${_srccpp})
  
  # make the pyd library link against boost_numpy python and boost
  TARGET_LINK_LIBRARIES(${_name} boost_numpy ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
  
  # put the example target into a VS solution folder named example (should
  # be a no-op for Linux)
  SET_PROPERTY(TARGET ${_name} PROPERTY FOLDER "example")
endmacro()

macro( addPythonMod _name _srccpp )
  PYTHON_ADD_MODULE(${_name} ${_srccpp})
  
  # make the pyd library link against boost_numpy python and boost
  TARGET_LINK_LIBRARIES(${_name} boost_numpy ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
  
  # put the example target into a VS solution folder named example (should
  # be a no-op for Linux)
  SET_PROPERTY(TARGET ${_name} PROPERTY FOLDER "example")
endmacro()

addPythonMod(gaussian gaussian.cpp)
addPythonExe(dtype dtype.cpp)
addPythonExe(fromdata fromdata.cpp)
addPythonExe(ndarray ndarray.cpp)
addPythonExe(simple simple.cpp)
addPythonExe(ufunc ufunc.cpp)
addPythonExe(wrap wrap.cpp)

# # installation logic (skip until it is better thought out)
# set(DEST_EXAMPLE boost.numpy/example)
#
# # install executables demonstrating embedding python
# install(TARGETS dtype fromdata ndarray simple ufunc wrap RUNTIME
#   DESTINATION ${DEST_EXAMPLE}
#   ${INSTALL_PERMSSIONS_RUNTIME}
#   )
#
# # install extension module
# install(TARGETS gaussian LIBRARY
#   DESTINATION ${DEST_EXAMPLE}
#   ${INSTALL_PERMSSIONS_RUNTIME}
#   )
#
# # install source file using the extension module
# install(FILES demo_gaussian.py 
#   DESTINATION ${DEST_EXAMPLE}
#   ${INSTALL_PERMSSIONS_SRC}
#   )
