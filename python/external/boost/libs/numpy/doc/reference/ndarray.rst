ndarray
=======

.. contents ::

A `ndarray`_ is an N-dimensional array which contains items of the same type and size, where N is the number of dimensions and is specified in the form of a ``shape`` tuple. Optionally, the numpy ``dtype`` for the objects contained may also be specified.

.. _ndarray: http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
.. _dtype: http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#data-type-objects-dtype

 ``<boost/numpy/ndarray.hpp>`` contains the structures and methods necessary to move raw data between C++ and Python and create ndarrays from the data



synopsis
--------

::

  namespace boost 
  {
  namespace numpy 
  {

  class ndarray : public python::object 
  {

  public:
  
    enum bitflag 
    {
      NONE=0x0, C_CONTIGUOUS=0x1, F_CONTIGUOUS=0x2, V_CONTIGUOUS=0x1|0x2, 
      ALIGNED=0x4, WRITEABLE=0x8, BEHAVED=0x4|0x8,
      CARRAY_RO=0x1|0x4, CARRAY=0x1|0x4|0x8, CARRAY_MIS=0x1|0x8,
      FARRAY_RO=0x2|0x4, FARRAY=0x2|0x4|0x8, FARRAY_MIS=0x2|0x8,
      UPDATE_ALL=0x1|0x2|0x4, VARRAY=0x1|0x2|0x8, ALL=0x1|0x2|0x4|0x8
    };

    ndarray view(dtype const & dt) const;
    ndarray astype(dtype const & dt) const;
    ndarray copy() const;
    int const shape(int n) const;
    int const strides(int n) const;
    char * get_data() const;
    dtype get_dtype() const;
    python::object get_base() const;
    void set_base(object const & base);
    Py_intptr_t const * get_shape() const;
    Py_intptr_t const * get_strides() const;
    int const get_nd() const;
   
    bitflag const get_flags() const;
  
    ndarray transpose() const;
    ndarray squeeze() const;
    ndarray reshape(python::tuple const & shape) const;
    python::object scalarize() const;
  };

  ndarray zeros(python::tuple const & shape, dtype const & dt);
  ndarray zeros(int nd, Py_intptr_t const * shape, dtype const & dt);

  ndarray empty(python::tuple const & shape, dtype const & dt);
  ndarray empty(int nd, Py_intptr_t const * shape, dtype const & dt);

  ndarray array(python::object const & obj);
  ndarray array(python::object const & obj, dtype const & dt);

  template <typename Container>
  ndarray from_data(void * data,dtype const & dt,Container shape,Container strides,python::object const & owner);
  template <typename Container>
  ndarray from_data(void const * data, dtype const & dt, Container shape, Container strides, python::object const & owner);

  ndarray from_object(python::object const & obj, dtype const & dt,int nd_min, int nd_max, ndarray::bitflag flags=ndarray::NONE);
  ndarray from_object(python::object const & obj, dtype const & dt,int nd, ndarray::bitflag flags=ndarray::NONE);
  ndarray from_object(python::object const & obj, dtype const & dt, ndarray::bitflag flags=ndarray::NONE);
  ndarray from_object(python::object const & obj, int nd_min, int nd_max,ndarray::bitflag flags=ndarray::NONE);
  ndarray from_object(python::object const & obj, int nd, ndarray::bitflag flags=ndarray::NONE);
  ndarray from_object(python::object const & obj, ndarray::bitflag flags=ndarray::NONE)

  ndarray::bitflag operator|(ndarray::bitflag a, ndarray::bitflag b) ; 
  ndarray::bitflag operator&(ndarray::bitflag a, ndarray::bitflag b);

  }


constructors
------------

::

  ndarray view(dtype const & dt) const;

:Returns: new ndarray with old ndarray data cast as supplied dtype

::

  ndarray astype(dtype const & dt) const;

:Returns: new ndarray with old ndarray data converted to supplied dtype

::

  ndarray copy() const;
  
:Returns: Copy of calling ndarray object

:: 

  ndarray transpose() const;

:Returns:  An ndarray with the rows and columns interchanged
 
::

  ndarray squeeze() const;

:Returns:  An ndarray with all unit-shaped dimensions removed
  
::

  ndarray reshape(python::tuple const & shape) const;

:Requirements: The new ``shape`` of the ndarray must be supplied as a tuple

:Returns:  An ndarray with the same data but reshaped to the ``shape`` supplied 


::

  python::object scalarize() const;

:Returns: A scalar if the ndarray has only one element, otherwise it returns the entire array

::

  ndarray zeros(python::tuple const & shape, dtype const & dt);
  ndarray zeros(int nd, Py_intptr_t const * shape, dtype const & dt);

:Requirements: The following parameters must be supplied as required :

		* the ``shape`` or the size of all dimensions, as a tuple
		* the ``dtype`` of the data
		* the ``nd`` size for a square shaped ndarray
		* the ``shape`` Py_intptr_t 

:Returns:  A new ndarray with the given shape and data type, with data initialized to zero.

::

  ndarray empty(python::tuple const & shape, dtype const & dt);
  ndarray empty(int nd, Py_intptr_t const * shape, dtype const & dt);


:Requirements: The following parameters must be supplied :

		* the ``shape`` or the size of all dimensions, as a tuple
		* the ``dtype`` of the data
		* the ``shape`` Py_intptr_t 

:Returns:  A new ndarray with the given shape and data type, with data left uninitialized.

::

  ndarray array(python::object const & obj);
  ndarray array(python::object const & obj, dtype const & dt);

:Returns:  A new ndarray from an arbitrary Python sequence, with dtype of each element specified optionally

::

  template <typename Container>
  inline ndarray from_data(void * data,dtype const & dt,Container shape,Container strides,python::object const & owner)

:Requirements: The following parameters must be supplied :

		* the ``data`` which is a generic C++ data container
		* the dtype ``dt`` of the data
		* the ``shape`` of the ndarray as Python object
		* the ``strides`` of each dimension of the array as a Python object
		* the ``owner`` of the data, in case it is not the ndarray itself

:Returns: ndarray with attributes and data supplied

:Note: The ``Container`` typename must be one that is convertible to a std::vector or python object type

::

  ndarray from_object(python::object const & obj, dtype const & dt,int nd_min, int nd_max, ndarray::bitflag flags=ndarray::NONE);

:Requirements: The following parameters must be supplied :

		* the ``obj`` Python object to convert to ndarray
		* the dtype ``dt`` of the data
		* minimum number of dimensions ``nd_min`` of the ndarray as Python object
		* maximum number of dimensions ``nd_max`` of the ndarray as Python object
		* optional ``flags`` bitflags

:Returns: ndarray constructed with dimensions and data supplied as parameters

::

  inline ndarray from_object(python::object const & obj, dtype const & dt, int nd, ndarray::bitflag flags=ndarray::NONE);

:Requirements: The following parameters must be supplied :

		* the ``obj`` Python object to convert to ndarray
		* the dtype ``dt`` of the data
		* number of dimensions ``nd`` of the ndarray as Python object
		* optional ``flags`` bitflags

:Returns: ndarray with dimensions ``nd`` x ``nd`` and suplied parameters

::

  inline ndarray from_object(python::object const & obj, dtype const & dt, ndarray::bitflag flags=ndarray::NONE)

:Requirements: The following parameters must be supplied :

		* the ``obj`` Python object to convert to ndarray
		* the dtype ``dt`` of the data
		* optional ``flags`` bitflags

:Returns: Supplied Python object as ndarray

::

  ndarray from_object(python::object const & obj, int nd_min, int nd_max, ndarray::bitflag flags=ndarray::NONE);

:Requirements: The following parameters must be supplied :

		* the ``obj`` Python object to convert to ndarray
		* minimum number of dimensions ``nd_min`` of the ndarray as Python object
		* maximum number of dimensions ``nd_max`` of the ndarray as Python object
		* optional ``flags`` bitflags

:Returns: ndarray with supplied dimension limits and parameters

:Note: dtype need not be supplied here

::

  inline ndarray from_object(python::object const & obj, int nd, ndarray::bitflag flags=ndarray::NONE);

:Requirements: The following parameters must be supplied :

		* the ``obj`` Python object to convert to ndarray
		* the dtype ``dt`` of the data
		* number of dimensions ``nd`` of the ndarray as Python object
		* optional ``flags`` bitflags

:Returns: ndarray of ``nd`` x ``nd`` dimensions constructed from the supplied object

::

  inline ndarray from_object(python::object const & obj, ndarray::bitflag flags=ndarray::NONE)

:Requirements: The following parameters must be supplied :

		* the ``obj`` Python object to convert to ndarray
		* optional ``flags`` bitflags

:Returns: ndarray of same dimensions and dtype as supplied Python object


accessors
---------

::

  int const shape(int n) const;

:Returns: The size of the n-th dimension of the ndarray

::

  int const strides(int n) const;

:Returns: The stride of the nth dimension.

::

  char * get_data() const;

:Returns: Array's raw data pointer as a char

:Note: This returns char so stride math works properly on it.User will have to reinterpret_cast it.

::

  dtype get_dtype() const;

:Returns: Array's data-type descriptor object (dtype)


::

  python::object get_base() const;

:Returns: Object that owns the array's data, or None if the array owns its own data.  


::

  void set_base(object const & base);

:Returns: Set the object that owns the array's data. Exercise caution while using this


::

  Py_intptr_t const * get_shape() const;

:Returns: Shape of the array as an array of integers


::

  Py_intptr_t const * get_strides() const;

:Returns: Stride of the array as an array of integers


::

  int const get_nd() const;

:Returns: Number of array dimensions


::

  bitflag const get_flags() const;

:Returns: Array flags

::

  inline ndarray::bitflag operator|(ndarray::bitflag a, ndarray::bitflag b)

:Returns: bitflag logically OR-ed as (a | b)

::

  inline ndarray::bitflag operator&(ndarray::bitflag a, ndarray::bitflag b)

:Returns: bitflag logically AND-ed as (a & b)


Example(s)
----------

::

  p::object tu = p::make_tuple('a','b','c') ;
  np::ndarray example_tuple = np::array (tu) ; 

  p::list l ;
  np::ndarray example_list = np::array (l) ; 

  np::dtype dt = np::dtype::get_builtin<int>();
  np::ndarray example_list1 = np::array (l,dt);

  int data[] = {1,2,3,4} ;
  p::tuple shape = p::make_tuple(4) ;
  p::tuple stride = p::make_tuple(4) ; 
  p::object own ;
  np::ndarray data_ex = np::from_data(data,dt,shape,stride,own);

  uint8_t mul_data[][4] = {{1,2,3,4},{5,6,7,8},{1,3,5,7}};
  shape = p::make_tuple(3,2) ;
  stride = p::make_tuple(4,2) ; 
  np::dtype dt1 = np::dtype::get_builtin<uint8_t>();

  np::ndarray mul_data_ex = np::from_data(mul_data,dt1, p::make_tuple(3,4),p::make_tuple(4,1),p::object());
  mul_data_ex = np::from_data(mul_data,dt1, shape,stride,p::object());

