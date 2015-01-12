// Copyright Jim Bosch 2010-2012.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <boost/numpy.hpp>

#include <cmath>
#include <memory>

#ifndef M_PI
#include <boost/math/constants/constants.hpp>
const double M_PI = boost::math::constants::pi<double>();
#endif

namespace bp = boost::python;
namespace bn = boost::numpy;

/**
 *  A 2x2 matrix class, purely for demonstration purposes.
 *
 *  Instead of wrapping this class with Boost.Python, we'll convert it to/from numpy.ndarray.
 */
class matrix2 {
public:

    double & operator()(int i, int j) {
        return _data[i*2 + j];
    }

    double const & operator()(int i, int j) const {
        return _data[i*2 + j];
    }
    
    double const * data() const { return _data; }

private:
    double _data[4];
};

/**
 *  A 2-element vector class, purely for demonstration purposes.
 *
 *  Instead of wrapping this class with Boost.Python, we'll convert it to/from numpy.ndarray.
 */
class vector2 {
public:

    double & operator[](int i) {
        return _data[i];
    }

    double const & operator[](int i) const {
        return _data[i];
    }
    
    double const * data() const { return _data; }

    vector2 operator+(vector2 const & other) const {
        vector2 r;
        r[0] = _data[0] + other[0];
        r[1] = _data[1] + other[1];
        return  r;
    }

    vector2 operator-(vector2 const & other) const {
        vector2 r;
        r[0] = _data[0] - other[0];
        r[1] = _data[1] - other[1];
        return  r;
    }

private:
    double _data[2];
};

/**
 *  Matrix-vector multiplication.
 */
vector2 operator*(matrix2 const & m, vector2 const & v) {
    vector2 r;
    r[0] = m(0, 0) * v[0] + m(0, 1) * v[1];
    r[1] = m(1, 0) * v[0] + m(1, 1) * v[1];
    return r;
}

/**
 *  Vector inner product.
 */
double dot(vector2 const & v1, vector2 const & v2) {
    return v1[0] * v2[0] + v1[1] * v2[1];
}

/**
 *  This class represents a simple 2-d Gaussian (Normal) distribution, defined by a
 *  mean vector 'mu' and a covariance matrix 'sigma'.
 */
class bivariate_gaussian {
public:

    vector2 const & get_mu() const { return _mu; }

    matrix2 const & get_sigma() const { return _sigma; }

    /**
     *  Evaluate the density of the distribution at a point defined by a two-element vector.
     */
    double operator()(vector2 const & p) const {
        vector2 u = _cholesky * (p - _mu);
        return 0.5 * _cholesky(0, 0) * _cholesky(1, 1) * std::exp(-0.5 * dot(u, u)) / M_PI;
    }

    /**
     *  Evaluate the density of the distribution at an (x, y) point.
     */
    double operator()(double x, double y) const {
        vector2 p;
        p[0] = x;
        p[1] = y;
        return operator()(p);
    }

    /**
     *  Construct from a mean vector and covariance matrix.
     */
    bivariate_gaussian(vector2 const & mu, matrix2 const & sigma)
        : _mu(mu), _sigma(sigma), _cholesky(compute_inverse_cholesky(sigma))
    {}
    
private:

    /**
     *  This evaluates the inverse of the Cholesky factorization of a 2x2 matrix;
     *  it's just a shortcut in evaluating the density.
     */
    static matrix2 compute_inverse_cholesky(matrix2 const & m) {
        matrix2 l;
        // First do cholesky factorization: l l^t = m
        l(0, 0) = std::sqrt(m(0, 0));
        l(0, 1) = m(0, 1) / l(0, 0);
        l(1, 1) = std::sqrt(m(1, 1) - l(0,1) * l(0,1));
        // Now do forward-substitution (in-place) to invert:
        l(0, 0) = 1.0 / l(0, 0);
        l(1, 0) = l(0, 1) = -l(0, 1) / l(1, 1);
        l(1, 1) = 1.0 / l(1, 1);
        return l;
    }

    vector2 _mu;
    matrix2 _sigma;
    matrix2 _cholesky;
                        
};

/*
 *  We have a two options for wrapping get_mu and get_sigma into NumPy-returning Python methods:
 *   - we could deep-copy the data, making totally new NumPy arrays;
 *   - we could make NumPy arrays that point into the existing memory.
 *  The latter is often preferable, especially if the arrays are large, but it's dangerous unless
 *  the reference counting is correct: the returned NumPy array needs to hold a reference that
 *  keeps the memory it points to from being deallocated as long as it is alive.  This is what the
 *  "owner" argument to from_data does - the NumPy array holds a reference to the owner, keeping it
 *  from being destroyed.
 *
 *  Note that this mechanism isn't completely safe for data members that can have their internal
 *  storage reallocated.  A std::vector, for instance, can be invalidated when it is resized,
 *  so holding a Python reference to a C++ class that holds a std::vector may not be a guarantee
 *  that the memory in the std::vector will remain valid.
 */

/**
 *  These two functions are custom wrappers for get_mu and get_sigma, providing the shallow-copy
 *  conversion with reference counting described above.
 *
 *  It's also worth noting that these return NumPy arrays that cannot be modified in Python;
 *  the const overloads of vector::data() and matrix::data() return const references, 
 *  and passing a const pointer to from_data causes NumPy's 'writeable' flag to be set to false.
 */
static bn::ndarray py_get_mu(bp::object const & self) {
    vector2 const & mu = bp::extract<bivariate_gaussian const &>(self)().get_mu();
    return bn::from_data(
        mu.data(),
        bn::dtype::get_builtin<double>(),
        bp::make_tuple(2),
        bp::make_tuple(sizeof(double)),
        self
    );  
}
static bn::ndarray py_get_sigma(bp::object const & self) {
    matrix2 const & sigma = bp::extract<bivariate_gaussian const &>(self)().get_sigma();
    return bn::from_data(
        sigma.data(),
        bn::dtype::get_builtin<double>(),
        bp::make_tuple(2, 2),
        bp::make_tuple(2 * sizeof(double), sizeof(double)),
        self
    );
}

/**
 *  To allow the constructor to work, we need to define some from-Python converters from NumPy arrays
 *  to the matrix/vector types.  The rvalue-from-python functionality is not well-documented in Boost.Python
 *  itself; you can learn more from boost/python/converter/rvalue_from_python_data.hpp.
 */

/**
 *  We start with two functions that just copy a NumPy array into matrix/vector objects.  These will be used
 *  in the templated converted below.  The first just uses the operator[] overloads provided by
 *  bp::object.
 */
static void copy_ndarray_to_mv2(bn::ndarray const & array, vector2 & vec) {
    vec[0] = bp::extract<double>(array[0]);
    vec[1] = bp::extract<double>(array[1]);
}

/**
 *  Here, we'll take the alternate approach of using the strides to access the array's memory directly.
 *  This can be much faster for large arrays.
 */
static void copy_ndarray_to_mv2(bn::ndarray const & array, matrix2 & mat) {
    // Unfortunately, get_strides() can't be inlined, so it's best to call it once up-front.
    Py_intptr_t const * strides = array.get_strides();
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            mat(i, j) = *reinterpret_cast<double const *>(array.get_data() + i * strides[0] + j * strides[1]);
        }
    }
}

/**
 *  Here's the actual converter.  Because we've separated the differences into the above functions,
 *  we can write a single template class that works for both matrix2 and vector2.
 */
template <typename T, int N>
struct mv2_from_python {
    
    /**
     *  Register the converter.
     */
    mv2_from_python() {
        bp::converter::registry::push_back(
            &convertible,
            &construct,
            bp::type_id< T >()
        );
    }

    /**
     *  Test to see if we can convert this to the desired type; if not return zero.
     *  If we can convert, returned pointer can be used by construct().
     */
    static void * convertible(PyObject * p) {
        try {
            bp::object obj(bp::handle<>(bp::borrowed(p)));
            std::auto_ptr<bn::ndarray> array(
                new bn::ndarray(
                    bn::from_object(obj, bn::dtype::get_builtin<double>(), N, N, bn::ndarray::V_CONTIGUOUS)
                )
            );
            if (array->shape(0) != 2) return 0;
            if (N == 2 && array->shape(1) != 2) return 0;
            return array.release();
        } catch (bp::error_already_set & err) {
            bp::handle_exception();
            return 0;
        }
    }

    /**
     *  Finish the conversion by initializing the C++ object into memory prepared by Boost.Python.
     */
    static void construct(PyObject * obj, bp::converter::rvalue_from_python_stage1_data * data) {
        // Extract the array we passed out of the convertible() member function.
        std::auto_ptr<bn::ndarray> array(reinterpret_cast<bn::ndarray*>(data->convertible));
        // Find the memory block Boost.Python has prepared for the result.
        typedef bp::converter::rvalue_from_python_storage<T> storage_t;
        storage_t * storage = reinterpret_cast<storage_t*>(data);
        // Use placement new to initialize the result.
        T * m_or_v = new (storage->storage.bytes) T();
        // Fill the result with the values from the NumPy array.
        copy_ndarray_to_mv2(*array, *m_or_v);
        // Finish up.
        data->convertible = storage->storage.bytes;
    }

};


BOOST_PYTHON_MODULE(gaussian) {
    bn::initialize();

    // Register the from-python converters
    mv2_from_python< vector2, 1 >();
    mv2_from_python< matrix2, 2 >();

    typedef double (bivariate_gaussian::*call_vector)(vector2 const &) const;

    bp::class_<bivariate_gaussian>("bivariate_gaussian", bp::init<bivariate_gaussian const &>())

        // Declare the constructor (wouldn't work without the from-python converters).
        .def(bp::init< vector2 const &, matrix2 const & >())

        // Use our custom reference-counting getters
        .add_property("mu", &py_get_mu)
        .add_property("sigma", &py_get_sigma)

        // First overload accepts a two-element array argument
        .def("__call__", (call_vector)&bivariate_gaussian::operator())

        // This overload works like a binary NumPy universal function: you can pass
        // in scalars or arrays, and the C++ function will automatically be called
        // on each element of an array argument.
        .def("__call__", bn::binary_ufunc<bivariate_gaussian,double,double,double>::make())
        ;
}
