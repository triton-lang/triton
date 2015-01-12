binary_ufunc
============

.. contents ::

A ``binary_ufunc`` is a struct used as an intermediate step to broadcast two arguments so that a C++ function can be converted to a ufunc like function

 ``<boost/numpy/ufunc.hpp>`` contains the ``binary_ufunc`` structure definitions


synopsis
--------

::

  namespace boost
  {
  namespace numpy 
  {

  template <typename TBinaryFunctor,
            typename TArgument1=typename TBinaryFunctor::first_argument_type,
            typename TArgument2=typename TBinaryFunctor::second_argument_type,
            typename TResult=typename TBinaryFunctor::result_type>

  struct binary_ufunc 
  {

    static python::object call(TBinaryFunctor & self, 
                               python::object const & input1, 
                               python::object const & input2,
                               python::object const & output);

    static python::object make(); 
  };

  }
  }


constructors
------------

::

  struct example_binary_ufunc
  {
    typedef any_valid first_argument_type;
    typedef any_valid second_argument_type;
    typedef any_valid result_type;
  };

:Requirements: The ``any_valid`` type must be defined using typedef as a valid C++ type in order to use the struct methods correctly

:Note: The struct must be exposed as a Python class, and an instance of the class must be created to use the ``call`` method corresponding to the ``__call__`` attribute of the Python object

accessors
---------

::

  template <typename TBinaryFunctor,
            typename TArgument1=typename TBinaryFunctor::first_argument_type,
            typename TArgument2=typename TBinaryFunctor::second_argument_type,
            typename TResult=typename TBinaryFunctor::result_type>
  static python::object call(TBinaryFunctor & self, 
                             python::object const & input, 
                             python::object const & output);

:Requires: Typenames ``TBinaryFunctor`` and optionally ``TArgument1`` and ``TArgument2`` for argument type and ``TResult`` for result type

:Effects: Passes a Python object to the underlying C++ functor after broadcasting its arguments

::

  template <typename TBinaryFunctor,
            typename TArgument1=typename TBinaryFunctor::first_argument_type,
            typename TArgument2=typename TBinaryFunctor::second_argument_type,
            typename TResult=typename TBinaryFunctor::result_type>
  static python::object make(); 

:Requires: Typenames ``TBinaryFunctor`` and optionally ``TArgument1`` and ``TArgument2`` for argument type and ``TResult`` for result type

:Returns: A Python function object to call the overloaded () operator in the struct (in typical usage)

Example(s)
----------

::

  struct BinarySquare
  {
    typedef double first_argument_type;
    typedef double second_argument_type;
    typedef double result_type;

    double operator()(double a,double b) const { return (a*a + b*b) ; }
  };

  p::object ud = p::class_<BinarySquare, boost::shared_ptr<BinarySquare> >("BinarySquare").def("__call__", np::binary_ufunc<BinarySquare>::make());
  p::object inst = ud();
  result_array = inst.attr("__call__")(demo_array,demo_array) ;
  std::cout << "Square of list with binary ufunc is " << p::extract <char const * > (p::str(result_array)) << std::endl ; 

