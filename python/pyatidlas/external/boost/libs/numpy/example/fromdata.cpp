// Copyright Ankit Daftery 2011-2012.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

/**
 *  @brief An example to show how to access data using raw pointers.  This shows that you can use and
 *         manipulate data in either Python or C++ and have the changes reflected in both.
 */

#include <boost/numpy.hpp>
#include <iostream>

namespace p = boost::python;
namespace np = boost::numpy;


int main(int argc, char **argv)
{
  // Initialize the Python runtime.
  Py_Initialize();
  // Initialize NumPy
  np::initialize();
  // Create an array in C++
  int arr[] = {1,2,3,4} ; 
  // Create the ndarray in Python
  np::ndarray py_array = np::from_data(arr, np::dtype::get_builtin<int>() , p::make_tuple(4), p::make_tuple(4), p::object());
  // Print the ndarray that we just created, and the source C++ array
  std::cout << "C++ array :" << std::endl ;
  for (int j=0;j<4;j++)
  {
    std::cout << arr[j] << ' ' ;
  }
  std::cout << std::endl << "Python ndarray :" << p::extract<char const *>(p::str(py_array)) << std::endl;
  // Change an element in the python ndarray
  py_array[1] = 5 ; 
  // And see if the C++ container is changed or not
  std::cout << "Is the change reflected in the C++ array used to create the ndarray ? " << std::endl ; 
  for (int j = 0;j<4 ; j++)
  {
    std::cout << arr[j] << ' ' ;
  }
  // Conversely, change it in C++
  arr[2] = 8 ;
  // And see if the changes are reflected in the Python ndarray
  std::cout << std::endl << "Is the change reflected in the Python ndarray ?" << std::endl << p::extract<char const *>(p::str(py_array)) << std::endl;

}
