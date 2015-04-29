// Copyright 2011 Stefan Seefeld.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

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
  // Create a 3x3 shape...
  p::tuple shape = p::make_tuple(3, 3);
  // ...as well as a type for C++ float
  np::dtype dtype = np::dtype::get_builtin<float>();
  // Construct an array with the above shape and type
  np::ndarray a = np::zeros(shape, dtype);
  // Construct an empty array with the above shape and dtype as well
  np::ndarray b = np::empty(shape,dtype);
  // Print the array
  std::cout << "Original array:\n" << p::extract<char const *>(p::str(a)) << std::endl;
  // Reshape the array into a 1D array
  a = a.reshape(p::make_tuple(9));
  // Print it again.
  std::cout << "Reshaped array:\n" << p::extract<char const *>(p::str(a)) << std::endl;
}
