unary_ufunc
===========

.. contents ::

A ``unary_ufunc`` is a struct used as an intermediate step to broadcast a single argument so that a C++ function can be converted to a ufunc like function

 ``<boost/numpy/ufunc.hpp>`` contains the ``unary_ufunc`` structure definitions


synopsis
--------

::

  namespace boost 
  {
  namespace numpy 
  {

  template <typename TUnaryFunctor, 
            typename TArgument=typename TUnaryFunctor::argument_type, 
            typename TResult=typename TUnaryFunctor::result_type>
  struct unary_ufunc 
  {

    static python::object call(TUnaryFunctor & self, 
                               python::object const & input, 
                               python::object const & output) ;

    static python::object make(); 

  };
  }
  }


constructors
------------

::

  struct example_unary_ufunc
  {
    typedef any_valid_type argument_type;
    typedef any_valid_type result_type;
  };

:Requirements: The ``any_valid`` type must be defined using typedef as a valid C++ type in order to use the struct methods correctly

:Note: The struct must be exposed as a Python class, and an instance of the class must be created to use the ``call`` method corresponding to the ``__call__`` attribute of the Python object

accessors
---------

::

  template <typename TUnaryFunctor, 
            typename TArgument=typename TUnaryFunctor::argument_type,
            typename TResult=typename TUnaryFunctor::result_type>
  static python::object call(TUnaryFunctor & self, 
                             python::object const & input, 
                             python::object const & output);

:Requires: Typenames ``TUnaryFunctor`` and optionally ``TArgument`` for argument type and ``TResult`` for result type

:Effects: Passes a Python object to the underlying C++ functor after broadcasting its arguments

::

  template <typename TUnaryFunctor, 
            typename TArgument=typename TUnaryFunctor::argument_type,
            typename TResult=typename TUnaryFunctor::result_type>
  static python::object make(); 

:Requires: Typenames ``TUnaryFunctor`` and optionally ``TArgument`` for argument type and ``TResult`` for result type

:Returns: A Python function object to call the overloaded () operator in the struct (in typical usage)



Example(s)
----------

::

  struct UnarySquare 
  {
    typedef double argument_type;
    typedef double result_type;
    double operator()(double r) const { return r * r;}
  };

  p::object ud = p::class_<UnarySquare, boost::shared_ptr<UnarySquare> >("UnarySquare").def("__call__", np::unary_ufunc<UnarySquare>::make());
  p::object inst = ud();
  std::cout << "Square of unary scalar 1.0 is " << p::extract <char const * > (p::str(inst.attr("__call__")(1.0))) << std::endl ; 

