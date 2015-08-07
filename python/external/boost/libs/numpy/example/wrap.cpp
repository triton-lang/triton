// Copyright Jim Bosch 2011-2012.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

/**
 *  A simple example showing how to wrap a couple of C++ functions that
 *  operate on 2-d arrays into Python functions that take NumPy arrays
 *  as arguments.
 *        
 *  If you find have a lot of such functions to wrap, you may want to
 *  create a C++ array type (or use one of the many existing C++ array
 *  libraries) that maps well to NumPy arrays and create Boost.Python
 *  converters.  There's more work up front than the approach here,
 *  but much less boilerplate per function.  See the "Gaussian" example
 *  included with Boost.NumPy for an example of custom converters, or
 *  take a look at the "ndarray" project on GitHub for a more complete,
 *  high-level solution.
 *
 *  Note that we're using embedded Python here only to make a convenient
 *  self-contained example; you could just as easily put the wrappers
 *  in a regular C++-compiled module and imported them in regular
 *  Python.  Again, see the Gaussian demo for an example.
 */

#include <boost/numpy.hpp>
#include <boost/scoped_array.hpp>
#include <iostream>

namespace p = boost::python;
namespace np = boost::numpy;

// This is roughly the most efficient way to write a C/C++ function that operates
// on a 2-d NumPy array - operate directly on the array by incrementing a pointer
// with the strides.
void fill1(double * array, int rows, int cols, int row_stride, int col_stride) {
    double * row_iter = array;
    double n = 0.0; // just a counter we'll fill the array with.
    for (int i = 0; i < rows; ++i, row_iter += row_stride) {
        double * col_iter = row_iter;
        for (int j = 0; j < cols; ++j, col_iter += col_stride) {
            *col_iter = ++n;
        }
    }
}

// Here's a simple wrapper function for fill1.  It requires that the passed
// NumPy array be exactly what we're looking for - no conversion from nested
// sequences or arrays with other data types, because we want to modify it
// in-place.
void wrap_fill1(np::ndarray const & array) {
    if (array.get_dtype() != np::dtype::get_builtin<double>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
        p::throw_error_already_set();
    }
    if (array.get_nd() != 2) {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
        p::throw_error_already_set();
    }
    fill1(reinterpret_cast<double*>(array.get_data()),
          array.shape(0), array.shape(1),
          array.strides(0) / sizeof(double), array.strides(1) / sizeof(double));
}

// Another fill function that takes a double**.  This is less efficient, because
// it's not the native NumPy data layout, but it's common enough in C/C++ that
// it's worth its own example.  This time we don't pass the strides, and instead
// in wrap_fill2 we'll require the C_CONTIGUOUS bitflag, which guarantees that
// the column stride is 1 and the row stride is the number of columns.  That
// restricts the arrays that can be passed to fill2 (it won't work on most
// subarray views or transposes, for instance).
void fill2(double ** array, int rows, int cols) {
    double n = 0.0; // just a counter we'll fill the array with.
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            array[i][j] = ++n;
        }
    }    
}
// Here's the wrapper for fill2; it's a little more complicated because we need
// to check the flags and create the array of pointers.
void wrap_fill2(np::ndarray const & array) {
    if (array.get_dtype() != np::dtype::get_builtin<double>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
        p::throw_error_already_set();
    }
    if (array.get_nd() != 2) {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
        p::throw_error_already_set();
    }
    if (!(array.get_flags() & np::ndarray::C_CONTIGUOUS)) {
        PyErr_SetString(PyExc_TypeError, "Array must be row-major contiguous");
        p::throw_error_already_set();
    }
    double * iter = reinterpret_cast<double*>(array.get_data());
    int rows = array.shape(0);
    int cols = array.shape(1);
    boost::scoped_array<double*> ptrs(new double*[rows]);
    for (int i = 0; i < rows; ++i, iter += cols) {
        ptrs[i] = iter;
    }
    fill2(ptrs.get(), array.shape(0), array.shape(1));
}

BOOST_PYTHON_MODULE(example) {
    np::initialize();  // have to put this in any module that uses Boost.NumPy
    p::def("fill1", wrap_fill1);
    p::def("fill2", wrap_fill2);
}

int main(int argc, char **argv)
{
    // This line makes our module available to the embedded Python intepreter.
# if PY_VERSION_HEX >= 0x03000000
    PyImport_AppendInittab("example", &PyInit_example);
# else
    PyImport_AppendInittab("example", &initexample);
# endif
    // Initialize the Python runtime.
    Py_Initialize();

    PyRun_SimpleString(
        "import example\n"
        "import numpy\n"
        "z1 = numpy.zeros((5,6), dtype=float)\n"
        "z2 = numpy.zeros((4,3), dtype=float)\n"
        "example.fill1(z1)\n"
        "example.fill2(z2)\n"
        "print z1\n"
        "print z2\n"
    );
    Py_Finalize();
}


