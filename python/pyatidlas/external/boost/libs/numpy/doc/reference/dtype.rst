dtype
=====

.. contents ::

A `dtype`_ is an object describing the type of the elements of an ndarray

.. _dtype: http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#data-type-objects-dtype

 ``<boost/numpy/dtype.hpp>`` contains the method calls necessary to generate a python object equivalent to a numpy.dtype from builtin C++ objects, as well as to create custom dtypes from user defined types


synopsis
--------

::

  namespace boost 
  {
  namespace numpy 
  {

  class dtype : public python::object 
  {
    static python::detail::new_reference convert(python::object::object_cref arg, bool align);
  public:

    // Convert an arbitrary Python object to a data-type descriptor object.
    template <typename T>
    explicit dtype(T arg, bool align=false);

    // Get the built-in numpy dtype associated with the given scalar template type.
    template <typename T> static dtype get_builtin();

    // Return the size of the data type in bytes.
    int get_itemsize() const;
  };

  } 

constructors
------------

::

  template <typename T>
  explicit dtype(T arg, bool align=false)

:Requirements: ``T`` must be either :

               * a built-in C++ typename convertible to object
               * a valid python object or convertible to object

:Effects: Constructs an object from the supplied python object / convertible 
          to object / builtin C++ data type

:Throws: Nothing

::

  template <typename T> static dtype get_builtin();
  
:Requirements: The typename supplied, ``T`` must be a builtin C++ type also supported by numpy

:Returns: Numpy dtype corresponding to builtin C++ type

accessors
---------

::

  int get_itemsize() const;

:Returns: the size of the data type in bytes.


Example(s)
----------

::

  namespace np = boost::numpy;
  np::dtype dtype = np::dtype::get_builtin<double>();
  p::tuple for_custom_dtype = p::make_tuple("ha",dtype);
  np::dtype custom_dtype = np::dtype(list_for_dtype);

