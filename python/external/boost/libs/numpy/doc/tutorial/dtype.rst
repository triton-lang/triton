How to use dtypes
=================

Here is a brief tutorial to show how to create ndarrays with built-in python data types, and extract the types and values of member variables

Like before, first get the necessary headers, setup the namespaces and initialize the Python runtime and numpy module::

	#include <boost/numpy.hpp>
	#include <iostream>

	namespace p = boost::python;
	namespace np = boost::numpy;

	int main(int argc, char **argv)
	{
	  Py_Initialize();
	  np::initialize();

Next, we create the shape and dtype. We use the get_builtin method to get the numpy dtype corresponding to the builtin C++ dtype 
Here, we will create a 3x3 array passing a tuple with (3,3) for the size, and double as the data type ::

	  p::tuple shape = p::make_tuple(3, 3);
	  np::dtype dtype = np::dtype::get_builtin<double>();
	  np::ndarray a = np::zeros(shape, dtype);

Finally, we can print the array using the extract method in the python namespace. 
Here, we first convert the variable into a string, and then extract it as a C++ character array from the python string using the <char const \* > template ::

	std::cout << "Original array:\n" << p::extract<char const *>(p::str(a)) << std::endl;

We can also print the dtypes of the data members of the ndarray by using the get_dtype method for the ndarray ::

	std::cout << "Datatype is:\n" << p::extract<char const *>(p::str(a.get_dtype())) << std::endl ;

We can also create custom dtypes and build ndarrays with the custom dtypes

We use the dtype constructor to create a custom dtype. This constructor takes a list as an argument.

The list should contain one or more tuples of the format (variable name, variable type)

So first create a tuple with a variable name and its dtype, double, to create a custom dtype ::

  p::tuple for_custom_dtype = p::make_tuple("ha",dtype) ;

Next, create a list, and add this tuple to the list. Then use the list to create the custom dtype ::

  p::list list_for_dtype ;
  list_for_dtype.append(for_custom_dtype) ;
  np::dtype custom_dtype = np::dtype(list_for_dtype) ;

We are now ready to create an ndarray with dimensions specified by \*shape\* and of custom dtpye ::

  np::ndarray new_array = np::zeros(shape,custom_dtype);

	}
