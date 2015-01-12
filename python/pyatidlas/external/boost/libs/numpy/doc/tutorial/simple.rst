A simple tutorial on Arrays
===========================

Let's start with a simple tutorial to create and modify arrays.

Get the necessary headers for numpy components and set up necessary namespaces::

	#include <boost/numpy.hpp>
	#include <iostream>

	namespace p = boost::python;
	namespace np = boost::numpy;

Initialise the Python runtime, and the numpy module. Failure to call these results in segmentation errors::

	int main(int argc, char **argv)
	{
	  Py_Initialize();
	  np::initialize();


Zero filled n-dimensional arrays can be created using the shape and data-type of the array as a parameter. Here, the shape is 3x3 and the datatype is the built-in float type::

	  p::tuple shape = p::make_tuple(3, 3);
	  np::dtype dtype = np::dtype::get_builtin<float>();
	  np::ndarray a = np::zeros(shape, dtype);

You can also create an empty array like this ::

	np::ndarray b = np::empty(shape,dtype);
	
Print the original and reshaped array. The array a which is a list is first converted to a string, and each value in the list is extracted using extract< >::

	  std::cout << "Original array:\n" << p::extract<char const *>(p::str(a)) << std::endl;

	  // Reshape the array into a 1D array
	  a = a.reshape(p::make_tuple(9));
	  // Print it again.
	  std::cout << "Reshaped array:\n" << p::extract<char const *>(p::str(a)) << std::endl;
	}

