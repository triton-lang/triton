/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

#include <boost/bind.hpp>
#include <boost/python.hpp>

#include "isaac/exception/api.h"
#include "isaac/exception/driver.h"

#include "common.hpp"
#include "exceptions.h"


namespace wrap
{

//Code taken from https://mail.python.org/pipermail/cplusplus-sig/2006-May/010347.html
template< typename CPP_ExceptionType
          , typename X1 = bp::detail::not_specified
          , typename X2 = bp::detail::not_specified
          , typename X3 = bp::detail::not_specified
          >
class exception
        : public bp::class_<CPP_ExceptionType, X1, X2, X3>
{
public:
    typedef bp::class_<CPP_ExceptionType, X1, X2, X3> base_type;
    typedef typename base_type::wrapped_type wrapped_type;
    typedef exception<CPP_ExceptionType, X1, X2, X3>               self;

    // Construct with the class name, with or without docstring, and default
    // __init__() function
    exception(char const* name, char const* doc = 0) : base_type(name, doc)
    {
        init();
    }

    // Construct with class name, no docstring, and an uncallable
    // __init__ function
    exception(char const* name, bp::no_init_t const& no_init_tag): base_type(name, no_init_tag)
    {
        init();
    }

    // Construct with class name, docstring, and an uncallable
    // __init__ function
    exception(char const* name, char const* doc, bp::no_init_t const& no_init_tag): base_type(name, doc, no_init_tag)
    {
        init();
    }

    // Construct with class name and init<> function
    template <class DerivedT>
    inline exception(char const* name, bp::init_base<DerivedT> const& i): base_type(name, i)
    {
        init();
    }

    // Construct with class name, docstring and init<> function
    template <class DerivedT>
    inline exception( char const* name
                      , char const* doc
                      , bp::init_base<DerivedT> const& i): base_type(name, doc, i)
    {
        init();
    }

private:

    static void to_python_exception(bp::object const& exn_type, wrapped_type const& exn)
    {
        static const bp::to_python_value<wrapped_type> convert_argument;
        PyErr_SetObject(exn_type.ptr(), convert_argument(exn));
        bp::throw_error_already_set();
    }

    void init() const
    {
        bp::register_exception_translator<wrapped_type>( std::bind(&to_python_exception, *this, std::placeholders::_1));
    }
};

}

void export_exceptions()
{
    namespace exc = isaac::exception;

#define BIND_EXCEPTION(CPPNAME, PYTHONNAME) \
    wrap::exception<isaac::CPPNAME>(PYTHONNAME, bp::init<std::string>())\
        .def("__str__", &isaac::CPPNAME::what)

    BIND_EXCEPTION(operation_not_supported_exception, "OperationNotSupported");
    BIND_EXCEPTION(semantic_error, "SemanticError");

    //OCL
    wrap::exception<exc::ocl::base>("OclException", bp::no_init);
#define BIND_OCL_EXCEPTION(CPPNAME, PYTHONNAME) \
            wrap::exception<exc::ocl::CPPNAME, bp::bases<exc::ocl::base> >(PYTHONNAME)\
                .def("__str__", &exc::ocl::CPPNAME::what)


    BIND_OCL_EXCEPTION(out_of_resources, "OclLaunchOutOfResources");
    BIND_OCL_EXCEPTION(mem_object_allocation_failure, "MemObjectAllocationFailure");
    BIND_OCL_EXCEPTION(out_of_host_memory, "OutOfHostMemory");
    BIND_OCL_EXCEPTION(invalid_work_group_size, "InvalidWorkGroupSize");
    BIND_OCL_EXCEPTION(invalid_value, "InvalidValue");

    //CUDA
    wrap::exception<exc::cuda::base>("CudaException", bp::no_init);
#define BIND_CUDA_EXCEPTION(CPPNAME, PYTHONNAME) \
            wrap::exception<exc::cuda::CPPNAME, bp::bases<exc::cuda::base> >(PYTHONNAME)\
                .def("__str__", &exc::cuda::CPPNAME::what)


    BIND_CUDA_EXCEPTION(launch_out_of_resources, "CudaLaunchOutOfResources");
}
