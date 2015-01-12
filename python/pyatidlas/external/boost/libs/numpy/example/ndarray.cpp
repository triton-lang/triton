// Copyright Ankit Daftery 2011-2012.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

/**
 *  @brief An example to show how to create ndarrays using arbitrary Python sequences.
 *
 *  The Python sequence could be any object whose __array__ method returns an array, or any
 *  (nested) sequence.  This example also shows how to create arrays using both unit and
 *  non-unit strides.      
 */

#include <boost/numpy.hpp>
#include <iostream>

namespace p = boost::python;
namespace np = boost::numpy;

#if _MSC_VER
using boost::uint8_t;
#endif

int main(int argc, char **argv)
{
  // Initialize the Python runtime.
  Py_Initialize();
  // Initialize NumPy
  np::initialize();
  // Create an ndarray from a simple tuple
  p::object tu = p::make_tuple('a','b','c') ;
  np::ndarray example_tuple = np::array (tu) ; 
  // and from a list
  p::list l ;
  np::ndarray example_list = np::array (l) ; 
  // Optionally, you can also specify a dtype
  np::dtype dt = np::dtype::get_builtin<int>();
  np::ndarray example_list1 = np::array (l,dt);
  // You can also create an array by supplying data.First,create an integer array
  int data[] = {1,2,3,4} ;
  // Create a shape, and strides, needed by the function
  p::tuple shape = p::make_tuple(4) ;
  p::tuple stride = p::make_tuple(4) ; 
  // The function also needs an owner, to keep track of the data array passed. Passing none is dangerous
  p::object own ;
  // The from_data function takes the data array, datatype,shape,stride and owner as arguments
  // and returns an ndarray
  np::ndarray data_ex = np::from_data(data,dt,shape,stride,own);
  // Print the ndarray we created
  std::cout << "Single dimensional array ::" << std::endl << p::extract < char const * > (p::str(data_ex)) << std::endl ; 
  // Now lets make an 3x2 ndarray from a multi-dimensional array using non-unit strides
  // First lets create a 3x4 array of 8-bit integers
  uint8_t mul_data[][4] = {{1,2,3,4},{5,6,7,8},{1,3,5,7}};
  // Now let's create an array of 3x2 elements, picking the first and third elements from each row
  // For that, the shape will be 3x2
  shape = p::make_tuple(3,2) ;
  // The strides will be 4x2 i.e. 4 bytes to go to the next desired row, and 2 bytes to go to the next desired column
  stride = p::make_tuple(4,2) ; 
  // Get the numpy dtype for the built-in 8-bit integer data type
  np::dtype dt1 = np::dtype::get_builtin<uint8_t>();
  // First lets create and print out the ndarray as is
  np::ndarray mul_data_ex = np::from_data(mul_data,dt1, p::make_tuple(3,4),p::make_tuple(4,1),p::object());
  std::cout << "Original multi dimensional array :: " << std::endl << p::extract < char const * > (p::str(mul_data_ex)) << std::endl ; 
  // Now create the new ndarray using the shape and strides
  mul_data_ex = np::from_data(mul_data,dt1, shape,stride,p::object());
  // Print out the array we created using non-unit strides
  std::cout << "Selective multidimensional array :: "<<std::endl << p::extract < char const * > (p::str(mul_data_ex)) << std::endl ; 

}


