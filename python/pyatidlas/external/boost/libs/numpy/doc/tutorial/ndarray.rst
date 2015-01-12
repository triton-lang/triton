Creating ndarrays
=================

The Boost.Numpy library exposes quite a few methods to create ndarrays. ndarrays can be created in a variety of ways, include empty arrays and zero filled arrays.
ndarrays can also be created from arbitrary python sequences as well as from data and dtypes. 

This tutorial will introduce you to some of the ways in which you can create ndarrays. The methods covered here include creating ndarrays from arbitrary Python sequences, as well as from C++ containers, using both unit and non-unit strides

First, as before, initialise the necessary namepaces and runtimes ::

	#include <boost/numpy.hpp>
	#include <iostream>

	namespace p = boost::python;
	namespace np = boost::numpy;

	int main(int argc, char **argv)
	{
	  Py_Initialize();
	  np::initialize();

Let's now create an ndarray from a simple tuple. We first create a tuple object, and then pass it to the array method, to generate the necessary tuple ::

  p::object tu = p::make_tuple('a','b','c') ;
  np::ndarray example_tuple = np::array(tu) ; 

Let's now try the same with a list. We create an empty list, add an element using the append method, and as before, call the array method ::

  p::list l ;
  l.append('a') ;
  np::ndarray example_list = np::array (l) ;

Optionally, we can also specify a dtype for the array ::

  np::dtype dt = np::dtype::get_builtin<int>();
  np::ndarray example_list1 = np::array (l,dt);

We can also create an array by supplying data arrays and a few other parameters.

First,create an integer array ::

  int data[] = {1,2,3,4} ;

Create a shape, and strides, needed by the function ::

  p::tuple shape = p::make_tuple(4) ;
  p::tuple stride = p::make_tuple(4) ; 

Here, shape is 1x4 , and the stride is also 4. 
A stride is the number of bytes that must be travelled to get to the next desired element while constructing the ndarray. Here, the size of the "int" is 32 bits and hence, the stride is 4 to access each element.

The function also needs an owner, to keep track of the data array passed. Passing none is dangerous ::

  p::object own ;

The from_data function takes the data array, datatype,shape,stride and owner as arguments and returns an ndarray ::

  np::ndarray data_ex1 = np::from_data(data,dt, shape,stride,own);

Now let's print the ndarray we created ::

  std::cout << "Single dimensional array ::" << std::endl << p::extract < char const * > (p::str(data_ex)) << std::endl ; 

Let's make it a little more interesting. Lets make an 3x2 ndarray from a multi-dimensional array using non-unit strides

First lets create a 3x4 array of 8-bit integers ::

  uint8_t mul_data[][4] = {{1,2,3,4},{5,6,7,8},{1,3,5,7}};

Now let's create an array of 3x2 elements, picking the first and third elements from each row . For that, the shape will be 3x2.
The strides will be 4x2 i.e. 4 bytes to go to the next desired row, and 2 bytes to go to the next desired column ::

  shape = p::make_tuple(3,2) ;
  stride = p::make_tuple(4,2) ;
 
Get the numpy dtype for the built-in 8-bit integer data type ::

  np::dtype dt1 = np::dtype::get_builtin<uint8_t>();

Now lets first create and print out the ndarray as is.
Notice how we can pass the shape and strides in the function directly, as well as the owner. The last part can be done because we don't have any use to 
manipulate the "owner" object ::

  np::ndarray mul_data_ex = np::from_data(mul_data,dt1, p::make_tuple(3,4),p::make_tuple(4,1),p::object());
  std::cout << "Original multi dimensional array :: " << std::endl << p::extract < char const * > (p::str(mul_data_ex)) << std::endl ; 

Now create the new ndarray using the shape and strides and print out the array we created using non-unit strides ::

  mul_data_ex = np::from_data(mul_data,dt1, shape,stride,p::object());
  std::cout << "Selective multidimensional array :: "<<std::endl << p::extract < char const * > (p::str(mul_data_ex)) << std::endl ; 

Note : The from_data method will throw "error_already_set" if the number of elements dictated by the shape and the corresponding strides don't match

}
