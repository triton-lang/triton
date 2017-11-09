/*
    pybind11/complex.h: Complex number support

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include <complex>

/// glibc defines I as a macro which breaks things, e.g., boost template names
#ifdef I
#  undef I
#endif

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

// The format codes are already in the string in common.h, we just need to provide a specialization
template <typename T> struct is_fmt_numeric<std::complex<T>> {
    static constexpr bool value = true;
    static constexpr int index = is_fmt_numeric<T>::index + 3;
};

template <typename T> class type_caster<std::complex<T>> {
public:
    bool load(handle src, bool convert) {
        if (!src)
            return false;
        if (!convert && !PyComplex_Check(src.ptr()))
            return false;
        Py_complex result = PyComplex_AsCComplex(src.ptr());
        if (result.real == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        value = std::complex<T>((T) result.real, (T) result.imag);
        return true;
    }

    static handle cast(const std::complex<T> &src, return_value_policy /* policy */, handle /* parent */) {
        return PyComplex_FromDoubles((double) src.real(), (double) src.imag());
    }

    PYBIND11_TYPE_CASTER(std::complex<T>, _("complex"));
};
NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
