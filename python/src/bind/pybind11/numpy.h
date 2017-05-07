/*
    pybind11/numpy.h: Basic NumPy support, vectorize() wrapper

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include "complex.h"
#include <numeric>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <initializer_list>
#include <functional>
#include <utility>
#include <typeindex>

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

/* This will be true on all flat address space platforms and allows us to reduce the
   whole npy_intp / size_t / Py_intptr_t business down to just size_t for all size
   and dimension types (e.g. shape, strides, indexing), instead of inflicting this
   upon the library user. */
static_assert(sizeof(size_t) == sizeof(Py_intptr_t), "size_t != Py_intptr_t");

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)
template <typename type, typename SFINAE = void> struct npy_format_descriptor { };
template <typename type> struct is_pod_struct;

struct PyArrayDescr_Proxy {
    PyObject_HEAD
    PyObject *typeobj;
    char kind;
    char type;
    char byteorder;
    char flags;
    int type_num;
    int elsize;
    int alignment;
    char *subarray;
    PyObject *fields;
    PyObject *names;
};

struct PyArray_Proxy {
    PyObject_HEAD
    char *data;
    int nd;
    ssize_t *dimensions;
    ssize_t *strides;
    PyObject *base;
    PyObject *descr;
    int flags;
};

struct PyVoidScalarObject_Proxy {
    PyObject_VAR_HEAD
    char *obval;
    PyArrayDescr_Proxy *descr;
    int flags;
    PyObject *base;
};

struct numpy_type_info {
    PyObject* dtype_ptr;
    std::string format_str;
};

struct numpy_internals {
    std::unordered_map<std::type_index, numpy_type_info> registered_dtypes;

    numpy_type_info *get_type_info(const std::type_info& tinfo, bool throw_if_missing = true) {
        auto it = registered_dtypes.find(std::type_index(tinfo));
        if (it != registered_dtypes.end())
            return &(it->second);
        if (throw_if_missing)
            pybind11_fail(std::string("NumPy type info missing for ") + tinfo.name());
        return nullptr;
    }

    template<typename T> numpy_type_info *get_type_info(bool throw_if_missing = true) {
        return get_type_info(typeid(typename std::remove_cv<T>::type), throw_if_missing);
    }
};

inline PYBIND11_NOINLINE void load_numpy_internals(numpy_internals* &ptr) {
    ptr = &get_or_create_shared_data<numpy_internals>("_numpy_internals");
}

inline numpy_internals& get_numpy_internals() {
    static numpy_internals* ptr = nullptr;
    if (!ptr)
        load_numpy_internals(ptr);
    return *ptr;
}

struct npy_api {
    enum constants {
        NPY_C_CONTIGUOUS_ = 0x0001,
        NPY_F_CONTIGUOUS_ = 0x0002,
        NPY_ARRAY_OWNDATA_ = 0x0004,
        NPY_ARRAY_FORCECAST_ = 0x0010,
        NPY_ENSURE_ARRAY_ = 0x0040,
        NPY_ARRAY_ALIGNED_ = 0x0100,
        NPY_ARRAY_WRITEABLE_ = 0x0400,
        NPY_BOOL_ = 0,
        NPY_BYTE_, NPY_UBYTE_,
        NPY_SHORT_, NPY_USHORT_,
        NPY_INT_, NPY_UINT_,
        NPY_LONG_, NPY_ULONG_,
        NPY_LONGLONG_, NPY_ULONGLONG_,
        NPY_FLOAT_, NPY_DOUBLE_, NPY_LONGDOUBLE_,
        NPY_CFLOAT_, NPY_CDOUBLE_, NPY_CLONGDOUBLE_,
        NPY_OBJECT_ = 17,
        NPY_STRING_, NPY_UNICODE_, NPY_VOID_
    };

    static npy_api& get() {
        static npy_api api = lookup();
        return api;
    }

    bool PyArray_Check_(PyObject *obj) const {
        return (bool) PyObject_TypeCheck(obj, PyArray_Type_);
    }
    bool PyArrayDescr_Check_(PyObject *obj) const {
        return (bool) PyObject_TypeCheck(obj, PyArrayDescr_Type_);
    }

    PyObject *(*PyArray_DescrFromType_)(int);
    PyObject *(*PyArray_NewFromDescr_)
        (PyTypeObject *, PyObject *, int, Py_intptr_t *,
         Py_intptr_t *, void *, int, PyObject *);
    PyObject *(*PyArray_DescrNewFromType_)(int);
    PyObject *(*PyArray_NewCopy_)(PyObject *, int);
    PyTypeObject *PyArray_Type_;
    PyTypeObject *PyVoidArrType_Type_;
    PyTypeObject *PyArrayDescr_Type_;
    PyObject *(*PyArray_DescrFromScalar_)(PyObject *);
    PyObject *(*PyArray_FromAny_) (PyObject *, PyObject *, int, int, int, PyObject *);
    int (*PyArray_DescrConverter_) (PyObject *, PyObject **);
    bool (*PyArray_EquivTypes_) (PyObject *, PyObject *);
    int (*PyArray_GetArrayParamsFromObject_)(PyObject *, PyObject *, char, PyObject **, int *,
                                             Py_ssize_t *, PyObject **, PyObject *);
    PyObject *(*PyArray_Squeeze_)(PyObject *);
private:
    enum functions {
        API_PyArray_Type = 2,
        API_PyArrayDescr_Type = 3,
        API_PyVoidArrType_Type = 39,
        API_PyArray_DescrFromType = 45,
        API_PyArray_DescrFromScalar = 57,
        API_PyArray_FromAny = 69,
        API_PyArray_NewCopy = 85,
        API_PyArray_NewFromDescr = 94,
        API_PyArray_DescrNewFromType = 9,
        API_PyArray_DescrConverter = 174,
        API_PyArray_EquivTypes = 182,
        API_PyArray_GetArrayParamsFromObject = 278,
        API_PyArray_Squeeze = 136
    };

    static npy_api lookup() {
        module m = module::import("numpy.core.multiarray");
        auto c = m.attr("_ARRAY_API");
#if PY_MAJOR_VERSION >= 3
        void **api_ptr = (void **) PyCapsule_GetPointer(c.ptr(), NULL);
#else
        void **api_ptr = (void **) PyCObject_AsVoidPtr(c.ptr());
#endif
        npy_api api;
#define DECL_NPY_API(Func) api.Func##_ = (decltype(api.Func##_)) api_ptr[API_##Func];
        DECL_NPY_API(PyArray_Type);
        DECL_NPY_API(PyVoidArrType_Type);
        DECL_NPY_API(PyArrayDescr_Type);
        DECL_NPY_API(PyArray_DescrFromType);
        DECL_NPY_API(PyArray_DescrFromScalar);
        DECL_NPY_API(PyArray_FromAny);
        DECL_NPY_API(PyArray_NewCopy);
        DECL_NPY_API(PyArray_NewFromDescr);
        DECL_NPY_API(PyArray_DescrNewFromType);
        DECL_NPY_API(PyArray_DescrConverter);
        DECL_NPY_API(PyArray_EquivTypes);
        DECL_NPY_API(PyArray_GetArrayParamsFromObject);
        DECL_NPY_API(PyArray_Squeeze);
#undef DECL_NPY_API
        return api;
    }
};

inline PyArray_Proxy* array_proxy(void* ptr) {
    return reinterpret_cast<PyArray_Proxy*>(ptr);
}

inline const PyArray_Proxy* array_proxy(const void* ptr) {
    return reinterpret_cast<const PyArray_Proxy*>(ptr);
}

inline PyArrayDescr_Proxy* array_descriptor_proxy(PyObject* ptr) {
   return reinterpret_cast<PyArrayDescr_Proxy*>(ptr);
}

inline const PyArrayDescr_Proxy* array_descriptor_proxy(const PyObject* ptr) {
   return reinterpret_cast<const PyArrayDescr_Proxy*>(ptr);
}

inline bool check_flags(const void* ptr, int flag) {
    return (flag == (array_proxy(ptr)->flags & flag));
}

NAMESPACE_END(detail)

class dtype : public object {
public:
    PYBIND11_OBJECT_DEFAULT(dtype, object, detail::npy_api::get().PyArrayDescr_Check_);

    explicit dtype(const buffer_info &info) {
        dtype descr(_dtype_from_pep3118()(PYBIND11_STR_TYPE(info.format)));
        // If info.itemsize == 0, use the value calculated from the format string
        m_ptr = descr.strip_padding(info.itemsize ? info.itemsize : descr.itemsize()).release().ptr();
    }

    explicit dtype(const std::string &format) {
        m_ptr = from_args(pybind11::str(format)).release().ptr();
    }

    dtype(const char *format) : dtype(std::string(format)) { }

    dtype(list names, list formats, list offsets, size_t itemsize) {
        dict args;
        args["names"] = names;
        args["formats"] = formats;
        args["offsets"] = offsets;
        args["itemsize"] = pybind11::int_(itemsize);
        m_ptr = from_args(args).release().ptr();
    }

    /// This is essentially the same as calling numpy.dtype(args) in Python.
    static dtype from_args(object args) {
        PyObject *ptr = nullptr;
        if (!detail::npy_api::get().PyArray_DescrConverter_(args.release().ptr(), &ptr) || !ptr)
            throw error_already_set();
        return reinterpret_steal<dtype>(ptr);
    }

    /// Return dtype associated with a C++ type.
    template <typename T> static dtype of() {
        return detail::npy_format_descriptor<typename std::remove_cv<T>::type>::dtype();
    }

    /// Size of the data type in bytes.
    size_t itemsize() const {
        return (size_t) detail::array_descriptor_proxy(m_ptr)->elsize;
    }

    /// Returns true for structured data types.
    bool has_fields() const {
        return detail::array_descriptor_proxy(m_ptr)->names != nullptr;
    }

    /// Single-character type code.
    char kind() const {
        return detail::array_descriptor_proxy(m_ptr)->kind;
    }

private:
    static object _dtype_from_pep3118() {
        static PyObject *obj = module::import("numpy.core._internal")
            .attr("_dtype_from_pep3118").cast<object>().release().ptr();
        return reinterpret_borrow<object>(obj);
    }

    dtype strip_padding(size_t itemsize) {
        // Recursively strip all void fields with empty names that are generated for
        // padding fields (as of NumPy v1.11).
        if (!has_fields())
            return *this;

        struct field_descr { PYBIND11_STR_TYPE name; object format; pybind11::int_ offset; };
        std::vector<field_descr> field_descriptors;

        for (auto field : attr("fields").attr("items")()) {
            auto spec = field.cast<tuple>();
            auto name = spec[0].cast<pybind11::str>();
            auto format = spec[1].cast<tuple>()[0].cast<dtype>();
            auto offset = spec[1].cast<tuple>()[1].cast<pybind11::int_>();
            if (!len(name) && format.kind() == 'V')
                continue;
            field_descriptors.push_back({(PYBIND11_STR_TYPE) name, format.strip_padding(format.itemsize()), offset});
        }

        std::sort(field_descriptors.begin(), field_descriptors.end(),
                  [](const field_descr& a, const field_descr& b) {
                      return a.offset.cast<int>() < b.offset.cast<int>();
                  });

        list names, formats, offsets;
        for (auto& descr : field_descriptors) {
            names.append(descr.name);
            formats.append(descr.format);
            offsets.append(descr.offset);
        }
        return dtype(names, formats, offsets, itemsize);
    }
};

class array : public buffer {
public:
    PYBIND11_OBJECT_CVT(array, buffer, detail::npy_api::get().PyArray_Check_, raw_array)

    enum {
        c_style = detail::npy_api::NPY_C_CONTIGUOUS_,
        f_style = detail::npy_api::NPY_F_CONTIGUOUS_,
        forcecast = detail::npy_api::NPY_ARRAY_FORCECAST_
    };

    array() : array(0, static_cast<const double *>(nullptr)) {}

    array(const pybind11::dtype &dt, const std::vector<size_t> &shape,
          const std::vector<size_t> &strides, const void *ptr = nullptr,
          handle base = handle()) {
        auto& api = detail::npy_api::get();
        auto ndim = shape.size();
        if (shape.size() != strides.size())
            pybind11_fail("NumPy: shape ndim doesn't match strides ndim");
        auto descr = dt;

        int flags = 0;
        if (base && ptr) {
            if (isinstance<array>(base))
                /* Copy flags from base (except baseship bit) */
                flags = reinterpret_borrow<array>(base).flags() & ~detail::npy_api::NPY_ARRAY_OWNDATA_;
            else
                /* Writable by default, easy to downgrade later on if needed */
                flags = detail::npy_api::NPY_ARRAY_WRITEABLE_;
        }

        auto tmp = reinterpret_steal<object>(api.PyArray_NewFromDescr_(
            api.PyArray_Type_, descr.release().ptr(), (int) ndim, (Py_intptr_t *) shape.data(),
            (Py_intptr_t *) strides.data(), const_cast<void *>(ptr), flags, nullptr));
        if (!tmp)
            pybind11_fail("NumPy: unable to create array!");
        if (ptr) {
            if (base) {
                detail::array_proxy(tmp.ptr())->base = base.inc_ref().ptr();
            } else {
                tmp = reinterpret_steal<object>(api.PyArray_NewCopy_(tmp.ptr(), -1 /* any order */));
            }
        }
        m_ptr = tmp.release().ptr();
    }

    array(const pybind11::dtype &dt, const std::vector<size_t> &shape,
          const void *ptr = nullptr, handle base = handle())
        : array(dt, shape, default_strides(shape, dt.itemsize()), ptr, base) { }

    array(const pybind11::dtype &dt, size_t count, const void *ptr = nullptr,
          handle base = handle())
        : array(dt, std::vector<size_t>{ count }, ptr, base) { }

    template<typename T> array(const std::vector<size_t>& shape,
                               const std::vector<size_t>& strides,
                               const T* ptr, handle base = handle())
    : array(pybind11::dtype::of<T>(), shape, strides, (void *) ptr, base) { }

    template <typename T>
    array(const std::vector<size_t> &shape, const T *ptr,
          handle base = handle())
        : array(shape, default_strides(shape, sizeof(T)), ptr, base) { }

    template <typename T>
    array(size_t count, const T *ptr, handle base = handle())
        : array(std::vector<size_t>{ count }, ptr, base) { }

    explicit array(const buffer_info &info)
    : array(pybind11::dtype(info), info.shape, info.strides, info.ptr) { }

    /// Array descriptor (dtype)
    pybind11::dtype dtype() const {
        return reinterpret_borrow<pybind11::dtype>(detail::array_proxy(m_ptr)->descr);
    }

    /// Total number of elements
    size_t size() const {
        return std::accumulate(shape(), shape() + ndim(), (size_t) 1, std::multiplies<size_t>());
    }

    /// Byte size of a single element
    size_t itemsize() const {
        return (size_t) detail::array_descriptor_proxy(detail::array_proxy(m_ptr)->descr)->elsize;
    }

    /// Total number of bytes
    size_t nbytes() const {
        return size() * itemsize();
    }

    /// Number of dimensions
    size_t ndim() const {
        return (size_t) detail::array_proxy(m_ptr)->nd;
    }

    /// Base object
    object base() const {
        return reinterpret_borrow<object>(detail::array_proxy(m_ptr)->base);
    }

    /// Dimensions of the array
    const size_t* shape() const {
        return reinterpret_cast<const size_t *>(detail::array_proxy(m_ptr)->dimensions);
    }

    /// Dimension along a given axis
    size_t shape(size_t dim) const {
        if (dim >= ndim())
            fail_dim_check(dim, "invalid axis");
        return shape()[dim];
    }

    /// Strides of the array
    const size_t* strides() const {
        return reinterpret_cast<const size_t *>(detail::array_proxy(m_ptr)->strides);
    }

    /// Stride along a given axis
    size_t strides(size_t dim) const {
        if (dim >= ndim())
            fail_dim_check(dim, "invalid axis");
        return strides()[dim];
    }

    /// Return the NumPy array flags
    int flags() const {
        return detail::array_proxy(m_ptr)->flags;
    }

    /// If set, the array is writeable (otherwise the buffer is read-only)
    bool writeable() const {
        return detail::check_flags(m_ptr, detail::npy_api::NPY_ARRAY_WRITEABLE_);
    }

    /// If set, the array owns the data (will be freed when the array is deleted)
    bool owndata() const {
        return detail::check_flags(m_ptr, detail::npy_api::NPY_ARRAY_OWNDATA_);
    }

    /// Pointer to the contained data. If index is not provided, points to the
    /// beginning of the buffer. May throw if the index would lead to out of bounds access.
    template<typename... Ix> const void* data(Ix... index) const {
        return static_cast<const void *>(detail::array_proxy(m_ptr)->data + offset_at(index...));
    }

    /// Mutable pointer to the contained data. If index is not provided, points to the
    /// beginning of the buffer. May throw if the index would lead to out of bounds access.
    /// May throw if the array is not writeable.
    template<typename... Ix> void* mutable_data(Ix... index) {
        check_writeable();
        return static_cast<void *>(detail::array_proxy(m_ptr)->data + offset_at(index...));
    }

    /// Byte offset from beginning of the array to a given index (full or partial).
    /// May throw if the index would lead to out of bounds access.
    template<typename... Ix> size_t offset_at(Ix... index) const {
        if (sizeof...(index) > ndim())
            fail_dim_check(sizeof...(index), "too many indices for an array");
        return byte_offset(size_t(index)...);
    }

    size_t offset_at() const { return 0; }

    /// Item count from beginning of the array to a given index (full or partial).
    /// May throw if the index would lead to out of bounds access.
    template<typename... Ix> size_t index_at(Ix... index) const {
        return offset_at(index...) / itemsize();
    }

    /// Return a new view with all of the dimensions of length 1 removed
    array squeeze() {
        auto& api = detail::npy_api::get();
        return reinterpret_steal<array>(api.PyArray_Squeeze_(m_ptr));
    }

    /// Ensure that the argument is a NumPy array
    /// In case of an error, nullptr is returned and the Python error is cleared.
    static array ensure(handle h, int ExtraFlags = 0) {
        auto result = reinterpret_steal<array>(raw_array(h.ptr(), ExtraFlags));
        if (!result)
            PyErr_Clear();
        return result;
    }

protected:
    template<typename, typename> friend struct detail::npy_format_descriptor;

    void fail_dim_check(size_t dim, const std::string& msg) const {
        throw index_error(msg + ": " + std::to_string(dim) +
                          " (ndim = " + std::to_string(ndim()) + ")");
    }

    template<typename... Ix> size_t byte_offset(Ix... index) const {
        check_dimensions(index...);
        return byte_offset_unsafe(index...);
    }

    template<size_t dim = 0, typename... Ix> size_t byte_offset_unsafe(size_t i, Ix... index) const {
        return i * strides()[dim] + byte_offset_unsafe<dim + 1>(index...);
    }

    template<size_t dim = 0> size_t byte_offset_unsafe() const { return 0; }

    void check_writeable() const {
        if (!writeable())
            throw std::runtime_error("array is not writeable");
    }

    static std::vector<size_t> default_strides(const std::vector<size_t>& shape, size_t itemsize) {
        auto ndim = shape.size();
        std::vector<size_t> strides(ndim);
        if (ndim) {
            std::fill(strides.begin(), strides.end(), itemsize);
            for (size_t i = 0; i < ndim - 1; i++)
                for (size_t j = 0; j < ndim - 1 - i; j++)
                    strides[j] *= shape[ndim - 1 - i];
        }
        return strides;
    }

    template<typename... Ix> void check_dimensions(Ix... index) const {
        check_dimensions_impl(size_t(0), shape(), size_t(index)...);
    }

    void check_dimensions_impl(size_t, const size_t*) const { }

    template<typename... Ix> void check_dimensions_impl(size_t axis, const size_t* shape, size_t i, Ix... index) const {
        if (i >= *shape) {
            throw index_error(std::string("index ") + std::to_string(i) +
                              " is out of bounds for axis " + std::to_string(axis) +
                              " with size " + std::to_string(*shape));
        }
        check_dimensions_impl(axis + 1, shape + 1, index...);
    }

    /// Create array from any object -- always returns a new reference
    static PyObject *raw_array(PyObject *ptr, int ExtraFlags = 0) {
        if (ptr == nullptr)
            return nullptr;
        return detail::npy_api::get().PyArray_FromAny_(
            ptr, nullptr, 0, 0, detail::npy_api::NPY_ENSURE_ARRAY_ | ExtraFlags, nullptr);
    }
};

template <typename T, int ExtraFlags = array::forcecast> class array_t : public array {
public:
    array_t() : array(0, static_cast<const T *>(nullptr)) {}
    array_t(handle h, borrowed_t) : array(h, borrowed) { }
    array_t(handle h, stolen_t) : array(h, stolen) { }

    PYBIND11_DEPRECATED("Use array_t<T>::ensure() instead")
    array_t(handle h, bool is_borrowed) : array(raw_array_t(h.ptr()), stolen) {
        if (!m_ptr) PyErr_Clear();
        if (!is_borrowed) Py_XDECREF(h.ptr());
    }

    array_t(const object &o) : array(raw_array_t(o.ptr()), stolen) {
        if (!m_ptr) throw error_already_set();
    }

    explicit array_t(const buffer_info& info) : array(info) { }

    array_t(const std::vector<size_t> &shape,
            const std::vector<size_t> &strides, const T *ptr = nullptr,
            handle base = handle())
        : array(shape, strides, ptr, base) { }

    explicit array_t(const std::vector<size_t> &shape, const T *ptr = nullptr,
            handle base = handle())
        : array(shape, ptr, base) { }

    explicit array_t(size_t count, const T *ptr = nullptr, handle base = handle())
        : array(count, ptr, base) { }

    constexpr size_t itemsize() const {
        return sizeof(T);
    }

    template<typename... Ix> size_t index_at(Ix... index) const {
        return offset_at(index...) / itemsize();
    }

    template<typename... Ix> const T* data(Ix... index) const {
        return static_cast<const T*>(array::data(index...));
    }

    template<typename... Ix> T* mutable_data(Ix... index) {
        return static_cast<T*>(array::mutable_data(index...));
    }

    // Reference to element at a given index
    template<typename... Ix> const T& at(Ix... index) const {
        if (sizeof...(index) != ndim())
            fail_dim_check(sizeof...(index), "index dimension mismatch");
        return *(static_cast<const T*>(array::data()) + byte_offset(size_t(index)...) / itemsize());
    }

    // Mutable reference to element at a given index
    template<typename... Ix> T& mutable_at(Ix... index) {
        if (sizeof...(index) != ndim())
            fail_dim_check(sizeof...(index), "index dimension mismatch");
        return *(static_cast<T*>(array::mutable_data()) + byte_offset(size_t(index)...) / itemsize());
    }

    /// Ensure that the argument is a NumPy array of the correct dtype.
    /// In case of an error, nullptr is returned and the Python error is cleared.
    static array_t ensure(handle h) {
        auto result = reinterpret_steal<array_t>(raw_array_t(h.ptr()));
        if (!result)
            PyErr_Clear();
        return result;
    }

    static bool _check(handle h) {
        const auto &api = detail::npy_api::get();
        return api.PyArray_Check_(h.ptr())
               && api.PyArray_EquivTypes_(detail::array_proxy(h.ptr())->descr, dtype::of<T>().ptr());
    }

protected:
    /// Create array from any object -- always returns a new reference
    static PyObject *raw_array_t(PyObject *ptr) {
        if (ptr == nullptr)
            return nullptr;
        return detail::npy_api::get().PyArray_FromAny_(
            ptr, dtype::of<T>().release().ptr(), 0, 0,
            detail::npy_api::NPY_ENSURE_ARRAY_ | ExtraFlags, nullptr);
    }
};

template <typename T>
struct format_descriptor<T, detail::enable_if_t<detail::is_pod_struct<T>::value>> {
    static std::string format() {
        return detail::npy_format_descriptor<typename std::remove_cv<T>::type>::format();
    }
};

template <size_t N> struct format_descriptor<char[N]> {
    static std::string format() { return std::to_string(N) + "s"; }
};
template <size_t N> struct format_descriptor<std::array<char, N>> {
    static std::string format() { return std::to_string(N) + "s"; }
};

template <typename T>
struct format_descriptor<T, detail::enable_if_t<std::is_enum<T>::value>> {
    static std::string format() {
        return format_descriptor<
            typename std::remove_cv<typename std::underlying_type<T>::type>::type>::format();
    }
};

NAMESPACE_BEGIN(detail)
template <typename T, int ExtraFlags>
struct pyobject_caster<array_t<T, ExtraFlags>> {
    using type = array_t<T, ExtraFlags>;

    bool load(handle src, bool /* convert */) {
        value = type::ensure(src);
        return static_cast<bool>(value);
    }

    static handle cast(const handle &src, return_value_policy /* policy */, handle /* parent */) {
        return src.inc_ref();
    }
    PYBIND11_TYPE_CASTER(type, handle_type_name<type>::name());
};

template <typename T> struct is_std_array : std::false_type { };
template <typename T, size_t N> struct is_std_array<std::array<T, N>> : std::true_type { };

template <typename T>
struct is_pod_struct {
    enum { value = std::is_pod<T>::value && // offsetof only works correctly for POD types
           !std::is_reference<T>::value &&
           !std::is_array<T>::value &&
           !is_std_array<T>::value &&
           !std::is_integral<T>::value &&
           !std::is_enum<T>::value &&
           !std::is_same<typename std::remove_cv<T>::type, float>::value &&
           !std::is_same<typename std::remove_cv<T>::type, double>::value &&
           !std::is_same<typename std::remove_cv<T>::type, bool>::value &&
           !std::is_same<typename std::remove_cv<T>::type, std::complex<float>>::value &&
           !std::is_same<typename std::remove_cv<T>::type, std::complex<double>>::value };
};

template <typename T> struct npy_format_descriptor<T, enable_if_t<std::is_integral<T>::value>> {
private:
    constexpr static const int values[8] = {
        npy_api::NPY_BYTE_, npy_api::NPY_UBYTE_, npy_api::NPY_SHORT_,    npy_api::NPY_USHORT_,
        npy_api::NPY_INT_,  npy_api::NPY_UINT_,  npy_api::NPY_LONGLONG_, npy_api::NPY_ULONGLONG_ };
public:
    enum { value = values[detail::log2(sizeof(T)) * 2 + (std::is_unsigned<T>::value ? 1 : 0)] };
    static pybind11::dtype dtype() {
        if (auto ptr = npy_api::get().PyArray_DescrFromType_(value))
            return reinterpret_borrow<pybind11::dtype>(ptr);
        pybind11_fail("Unsupported buffer format!");
    }
    template <typename T2 = T, enable_if_t<std::is_signed<T2>::value, int> = 0>
    static PYBIND11_DESCR name() { return _("int") + _<sizeof(T)*8>(); }
    template <typename T2 = T, enable_if_t<!std::is_signed<T2>::value, int> = 0>
    static PYBIND11_DESCR name() { return _("uint") + _<sizeof(T)*8>(); }
};
template <typename T> constexpr const int npy_format_descriptor<
    T, enable_if_t<std::is_integral<T>::value>>::values[8];

#define DECL_FMT(Type, NumPyName, Name) template<> struct npy_format_descriptor<Type> { \
    enum { value = npy_api::NumPyName }; \
    static pybind11::dtype dtype() { \
        if (auto ptr = npy_api::get().PyArray_DescrFromType_(value)) \
            return reinterpret_borrow<pybind11::dtype>(ptr); \
        pybind11_fail("Unsupported buffer format!"); \
    } \
    static PYBIND11_DESCR name() { return _(Name); } }
DECL_FMT(float, NPY_FLOAT_, "float32");
DECL_FMT(double, NPY_DOUBLE_, "float64");
DECL_FMT(bool, NPY_BOOL_, "bool");
DECL_FMT(std::complex<float>, NPY_CFLOAT_, "complex64");
DECL_FMT(std::complex<double>, NPY_CDOUBLE_, "complex128");
#undef DECL_FMT

#define DECL_CHAR_FMT \
    static PYBIND11_DESCR name() { return _("S") + _<N>(); } \
    static pybind11::dtype dtype() { return pybind11::dtype(std::string("S") + std::to_string(N)); }
template <size_t N> struct npy_format_descriptor<char[N]> { DECL_CHAR_FMT };
template <size_t N> struct npy_format_descriptor<std::array<char, N>> { DECL_CHAR_FMT };
#undef DECL_CHAR_FMT

template<typename T> struct npy_format_descriptor<T, enable_if_t<std::is_enum<T>::value>> {
private:
    using base_descr = npy_format_descriptor<typename std::underlying_type<T>::type>;
public:
    static PYBIND11_DESCR name() { return base_descr::name(); }
    static pybind11::dtype dtype() { return base_descr::dtype(); }
};

struct field_descriptor {
    const char *name;
    size_t offset;
    size_t size;
    size_t alignment;
    std::string format;
    dtype descr;
};

inline PYBIND11_NOINLINE void register_structured_dtype(
    const std::initializer_list<field_descriptor>& fields,
    const std::type_info& tinfo, size_t itemsize,
    bool (*direct_converter)(PyObject *, void *&)) {

    auto& numpy_internals = get_numpy_internals();
    if (numpy_internals.get_type_info(tinfo, false))
        pybind11_fail("NumPy: dtype is already registered");

    list names, formats, offsets;
    for (auto field : fields) {
        if (!field.descr)
            pybind11_fail(std::string("NumPy: unsupported field dtype: `") +
                            field.name + "` @ " + tinfo.name());
        names.append(PYBIND11_STR_TYPE(field.name));
        formats.append(field.descr);
        offsets.append(pybind11::int_(field.offset));
    }
    auto dtype_ptr = pybind11::dtype(names, formats, offsets, itemsize).release().ptr();

    // There is an existing bug in NumPy (as of v1.11): trailing bytes are
    // not encoded explicitly into the format string. This will supposedly
    // get fixed in v1.12; for further details, see these:
    // - https://github.com/numpy/numpy/issues/7797
    // - https://github.com/numpy/numpy/pull/7798
    // Because of this, we won't use numpy's logic to generate buffer format
    // strings and will just do it ourselves.
    std::vector<field_descriptor> ordered_fields(fields);
    std::sort(ordered_fields.begin(), ordered_fields.end(),
        [](const field_descriptor &a, const field_descriptor &b) { return a.offset < b.offset; });
    size_t offset = 0;
    std::ostringstream oss;
    oss << "T{";
    for (auto& field : ordered_fields) {
        if (field.offset > offset)
            oss << (field.offset - offset) << 'x';
        // mark unaligned fields with '='
        if (field.offset % field.alignment)
            oss << '=';
        oss << field.format << ':' << field.name << ':';
        offset = field.offset + field.size;
    }
    if (itemsize > offset)
        oss << (itemsize - offset) << 'x';
    oss << '}';
    auto format_str = oss.str();

    // Sanity check: verify that NumPy properly parses our buffer format string
    auto& api = npy_api::get();
    auto arr =  array(buffer_info(nullptr, itemsize, format_str, 1));
    if (!api.PyArray_EquivTypes_(dtype_ptr, arr.dtype().ptr()))
        pybind11_fail("NumPy: invalid buffer descriptor!");

    auto tindex = std::type_index(tinfo);
    numpy_internals.registered_dtypes[tindex] = { dtype_ptr, format_str };
    get_internals().direct_conversions[tindex].push_back(direct_converter);
}

template <typename T>
struct npy_format_descriptor<T, enable_if_t<is_pod_struct<T>::value>> {
    static PYBIND11_DESCR name() { return _("struct"); }

    static pybind11::dtype dtype() {
        return reinterpret_borrow<pybind11::dtype>(dtype_ptr());
    }

    static std::string format() {
        static auto format_str = get_numpy_internals().get_type_info<T>(true)->format_str;
        return format_str;
    }

    static void register_dtype(const std::initializer_list<field_descriptor>& fields) {
        register_structured_dtype(fields, typeid(typename std::remove_cv<T>::type),
                                  sizeof(T), &direct_converter);
    }

private:
    static PyObject* dtype_ptr() {
        static PyObject* ptr = get_numpy_internals().get_type_info<T>(true)->dtype_ptr;
        return ptr;
    }

    static bool direct_converter(PyObject *obj, void*& value) {
        auto& api = npy_api::get();
        if (!PyObject_TypeCheck(obj, api.PyVoidArrType_Type_))
            return false;
        if (auto descr = reinterpret_steal<object>(api.PyArray_DescrFromScalar_(obj))) {
            if (api.PyArray_EquivTypes_(dtype_ptr(), descr.ptr())) {
                value = ((PyVoidScalarObject_Proxy *) obj)->obval;
                return true;
            }
        }
        return false;
    }
};

#define PYBIND11_FIELD_DESCRIPTOR_EX(T, Field, Name)                                          \
    ::pybind11::detail::field_descriptor {                                                    \
        Name, offsetof(T, Field), sizeof(decltype(std::declval<T>().Field)),                  \
        alignof(decltype(std::declval<T>().Field)),                                           \
        ::pybind11::format_descriptor<decltype(std::declval<T>().Field)>::format(),           \
        ::pybind11::detail::npy_format_descriptor<decltype(std::declval<T>().Field)>::dtype() \
    }

// Extract name, offset and format descriptor for a struct field
#define PYBIND11_FIELD_DESCRIPTOR(T, Field) PYBIND11_FIELD_DESCRIPTOR_EX(T, Field, #Field)

// The main idea of this macro is borrowed from https://github.com/swansontec/map-macro
// (C) William Swanson, Paul Fultz
#define PYBIND11_EVAL0(...) __VA_ARGS__
#define PYBIND11_EVAL1(...) PYBIND11_EVAL0 (PYBIND11_EVAL0 (PYBIND11_EVAL0 (__VA_ARGS__)))
#define PYBIND11_EVAL2(...) PYBIND11_EVAL1 (PYBIND11_EVAL1 (PYBIND11_EVAL1 (__VA_ARGS__)))
#define PYBIND11_EVAL3(...) PYBIND11_EVAL2 (PYBIND11_EVAL2 (PYBIND11_EVAL2 (__VA_ARGS__)))
#define PYBIND11_EVAL4(...) PYBIND11_EVAL3 (PYBIND11_EVAL3 (PYBIND11_EVAL3 (__VA_ARGS__)))
#define PYBIND11_EVAL(...)  PYBIND11_EVAL4 (PYBIND11_EVAL4 (PYBIND11_EVAL4 (__VA_ARGS__)))
#define PYBIND11_MAP_END(...)
#define PYBIND11_MAP_OUT
#define PYBIND11_MAP_COMMA ,
#define PYBIND11_MAP_GET_END() 0, PYBIND11_MAP_END
#define PYBIND11_MAP_NEXT0(test, next, ...) next PYBIND11_MAP_OUT
#define PYBIND11_MAP_NEXT1(test, next) PYBIND11_MAP_NEXT0 (test, next, 0)
#define PYBIND11_MAP_NEXT(test, next)  PYBIND11_MAP_NEXT1 (PYBIND11_MAP_GET_END test, next)
#ifdef _MSC_VER // MSVC is not as eager to expand macros, hence this workaround
#define PYBIND11_MAP_LIST_NEXT1(test, next) \
    PYBIND11_EVAL0 (PYBIND11_MAP_NEXT0 (test, PYBIND11_MAP_COMMA next, 0))
#else
#define PYBIND11_MAP_LIST_NEXT1(test, next) \
    PYBIND11_MAP_NEXT0 (test, PYBIND11_MAP_COMMA next, 0)
#endif
#define PYBIND11_MAP_LIST_NEXT(test, next) \
    PYBIND11_MAP_LIST_NEXT1 (PYBIND11_MAP_GET_END test, next)
#define PYBIND11_MAP_LIST0(f, t, x, peek, ...) \
    f(t, x) PYBIND11_MAP_LIST_NEXT (peek, PYBIND11_MAP_LIST1) (f, t, peek, __VA_ARGS__)
#define PYBIND11_MAP_LIST1(f, t, x, peek, ...) \
    f(t, x) PYBIND11_MAP_LIST_NEXT (peek, PYBIND11_MAP_LIST0) (f, t, peek, __VA_ARGS__)
// PYBIND11_MAP_LIST(f, t, a1, a2, ...) expands to f(t, a1), f(t, a2), ...
#define PYBIND11_MAP_LIST(f, t, ...) \
    PYBIND11_EVAL (PYBIND11_MAP_LIST1 (f, t, __VA_ARGS__, (), 0))

#define PYBIND11_NUMPY_DTYPE(Type, ...) \
    ::pybind11::detail::npy_format_descriptor<Type>::register_dtype \
        ({PYBIND11_MAP_LIST (PYBIND11_FIELD_DESCRIPTOR, Type, __VA_ARGS__)})

#ifdef _MSC_VER
#define PYBIND11_MAP2_LIST_NEXT1(test, next) \
    PYBIND11_EVAL0 (PYBIND11_MAP_NEXT0 (test, PYBIND11_MAP_COMMA next, 0))
#else
#define PYBIND11_MAP2_LIST_NEXT1(test, next) \
    PYBIND11_MAP_NEXT0 (test, PYBIND11_MAP_COMMA next, 0)
#endif
#define PYBIND11_MAP2_LIST_NEXT(test, next) \
    PYBIND11_MAP2_LIST_NEXT1 (PYBIND11_MAP_GET_END test, next)
#define PYBIND11_MAP2_LIST0(f, t, x1, x2, peek, ...) \
    f(t, x1, x2) PYBIND11_MAP2_LIST_NEXT (peek, PYBIND11_MAP2_LIST1) (f, t, peek, __VA_ARGS__)
#define PYBIND11_MAP2_LIST1(f, t, x1, x2, peek, ...) \
    f(t, x1, x2) PYBIND11_MAP2_LIST_NEXT (peek, PYBIND11_MAP2_LIST0) (f, t, peek, __VA_ARGS__)
// PYBIND11_MAP2_LIST(f, t, a1, a2, ...) expands to f(t, a1, a2), f(t, a3, a4), ...
#define PYBIND11_MAP2_LIST(f, t, ...) \
    PYBIND11_EVAL (PYBIND11_MAP2_LIST1 (f, t, __VA_ARGS__, (), 0))

#define PYBIND11_NUMPY_DTYPE_EX(Type, ...) \
    ::pybind11::detail::npy_format_descriptor<Type>::register_dtype \
        ({PYBIND11_MAP2_LIST (PYBIND11_FIELD_DESCRIPTOR_EX, Type, __VA_ARGS__)})

template  <class T>
using array_iterator = typename std::add_pointer<T>::type;

template <class T>
array_iterator<T> array_begin(const buffer_info& buffer) {
    return array_iterator<T>(reinterpret_cast<T*>(buffer.ptr));
}

template <class T>
array_iterator<T> array_end(const buffer_info& buffer) {
    return array_iterator<T>(reinterpret_cast<T*>(buffer.ptr) + buffer.size);
}

class common_iterator {
public:
    using container_type = std::vector<size_t>;
    using value_type = container_type::value_type;
    using size_type = container_type::size_type;

    common_iterator() : p_ptr(0), m_strides() {}

    common_iterator(void* ptr, const container_type& strides, const std::vector<size_t>& shape)
        : p_ptr(reinterpret_cast<char*>(ptr)), m_strides(strides.size()) {
        m_strides.back() = static_cast<value_type>(strides.back());
        for (size_type i = m_strides.size() - 1; i != 0; --i) {
            size_type j = i - 1;
            value_type s = static_cast<value_type>(shape[i]);
            m_strides[j] = strides[j] + m_strides[i] - strides[i] * s;
        }
    }

    void increment(size_type dim) {
        p_ptr += m_strides[dim];
    }

    void* data() const {
        return p_ptr;
    }

private:
    char* p_ptr;
    container_type m_strides;
};

template <size_t N> class multi_array_iterator {
public:
    using container_type = std::vector<size_t>;

    multi_array_iterator(const std::array<buffer_info, N> &buffers,
                         const std::vector<size_t> &shape)
        : m_shape(shape.size()), m_index(shape.size(), 0),
          m_common_iterator() {

        // Manual copy to avoid conversion warning if using std::copy
        for (size_t i = 0; i < shape.size(); ++i)
            m_shape[i] = static_cast<container_type::value_type>(shape[i]);

        container_type strides(shape.size());
        for (size_t i = 0; i < N; ++i)
            init_common_iterator(buffers[i], shape, m_common_iterator[i], strides);
    }

    multi_array_iterator& operator++() {
        for (size_t j = m_index.size(); j != 0; --j) {
            size_t i = j - 1;
            if (++m_index[i] != m_shape[i]) {
                increment_common_iterator(i);
                break;
            } else {
                m_index[i] = 0;
            }
        }
        return *this;
    }

    template <size_t K, class T> const T& data() const {
        return *reinterpret_cast<T*>(m_common_iterator[K].data());
    }

private:

    using common_iter = common_iterator;

    void init_common_iterator(const buffer_info &buffer,
                              const std::vector<size_t> &shape,
                              common_iter &iterator, container_type &strides) {
        auto buffer_shape_iter = buffer.shape.rbegin();
        auto buffer_strides_iter = buffer.strides.rbegin();
        auto shape_iter = shape.rbegin();
        auto strides_iter = strides.rbegin();

        while (buffer_shape_iter != buffer.shape.rend()) {
            if (*shape_iter == *buffer_shape_iter)
                *strides_iter = static_cast<size_t>(*buffer_strides_iter);
            else
                *strides_iter = 0;

            ++buffer_shape_iter;
            ++buffer_strides_iter;
            ++shape_iter;
            ++strides_iter;
        }

        std::fill(strides_iter, strides.rend(), 0);
        iterator = common_iter(buffer.ptr, strides, shape);
    }

    void increment_common_iterator(size_t dim) {
        for (auto &iter : m_common_iterator)
            iter.increment(dim);
    }

    container_type m_shape;
    container_type m_index;
    std::array<common_iter, N> m_common_iterator;
};

template <size_t N>
bool broadcast(const std::array<buffer_info, N>& buffers, size_t& ndim, std::vector<size_t>& shape) {
    ndim = std::accumulate(buffers.begin(), buffers.end(), size_t(0), [](size_t res, const buffer_info& buf) {
        return std::max(res, buf.ndim);
    });

    shape = std::vector<size_t>(ndim, 1);
    bool trivial_broadcast = true;
    for (size_t i = 0; i < N; ++i) {
        auto res_iter = shape.rbegin();
        bool i_trivial_broadcast = (buffers[i].size == 1) || (buffers[i].ndim == ndim);
        for (auto shape_iter = buffers[i].shape.rbegin();
             shape_iter != buffers[i].shape.rend(); ++shape_iter, ++res_iter) {

            if (*res_iter == 1)
                *res_iter = *shape_iter;
            else if ((*shape_iter != 1) && (*res_iter != *shape_iter))
                pybind11_fail("pybind11::vectorize: incompatible size/dimension of inputs!");

            i_trivial_broadcast = i_trivial_broadcast && (*res_iter == *shape_iter);
        }
        trivial_broadcast = trivial_broadcast && i_trivial_broadcast;
    }
    return trivial_broadcast;
}

template <typename Func, typename Return, typename... Args>
struct vectorize_helper {
    typename std::remove_reference<Func>::type f;

    template <typename T>
    explicit vectorize_helper(T&&f) : f(std::forward<T>(f)) { }

    object operator()(array_t<Args, array::c_style | array::forcecast>... args) {
        return run(args..., make_index_sequence<sizeof...(Args)>());
    }

    template <size_t ... Index> object run(array_t<Args, array::c_style | array::forcecast>&... args, index_sequence<Index...> index) {
        /* Request buffers from all parameters */
        const size_t N = sizeof...(Args);

        std::array<buffer_info, N> buffers {{ args.request()... }};

        /* Determine dimensions parameters of output array */
        size_t ndim = 0;
        std::vector<size_t> shape(0);
        bool trivial_broadcast = broadcast(buffers, ndim, shape);

        size_t size = 1;
        std::vector<size_t> strides(ndim);
        if (ndim > 0) {
            strides[ndim-1] = sizeof(Return);
            for (size_t i = ndim - 1; i > 0; --i) {
                strides[i - 1] = strides[i] * shape[i];
                size *= shape[i];
            }
            size *= shape[0];
        }

        if (size == 1)
            return cast(f(*((Args *) buffers[Index].ptr)...));

        array_t<Return> result(shape, strides);
        auto buf = result.request();
        auto output = (Return *) buf.ptr;

        if (trivial_broadcast) {
            /* Call the function */
            for (size_t i = 0; i < size; ++i) {
                output[i] = f((buffers[Index].size == 1
                               ? *((Args *) buffers[Index].ptr)
                               : ((Args *) buffers[Index].ptr)[i])...);
            }
        } else {
            apply_broadcast<N, Index...>(buffers, buf, index);
        }

        return result;
    }

    template <size_t N, size_t... Index>
    void apply_broadcast(const std::array<buffer_info, N> &buffers,
                         buffer_info &output, index_sequence<Index...>) {
        using input_iterator = multi_array_iterator<N>;
        using output_iterator = array_iterator<Return>;

        input_iterator input_iter(buffers, output.shape);
        output_iterator output_end = array_end<Return>(output);

        for (output_iterator iter = array_begin<Return>(output);
             iter != output_end; ++iter, ++input_iter) {
            *iter = f((input_iter.template data<Index, Args>())...);
        }
    }
};

template <typename T, int Flags> struct handle_type_name<array_t<T, Flags>> {
    static PYBIND11_DESCR name() { return _("numpy.ndarray[") + make_caster<T>::name() + _("]"); }
};

NAMESPACE_END(detail)

template <typename Func, typename Return, typename... Args /*,*/ PYBIND11_NOEXCEPT_TPL_ARG>
detail::vectorize_helper<Func, Return, Args...>
vectorize(const Func &f, Return (*) (Args ...) PYBIND11_NOEXCEPT_SPECIFIER) {
    return detail::vectorize_helper<Func, Return, Args...>(f);
}

template <typename Return, typename... Args /*,*/ PYBIND11_NOEXCEPT_TPL_ARG>
detail::vectorize_helper<Return (*) (Args ...) PYBIND11_NOEXCEPT_SPECIFIER, Return, Args...>
vectorize(Return (*f) (Args ...) PYBIND11_NOEXCEPT_SPECIFIER) {
    return vectorize<Return (*) (Args ...), Return, Args...>(f, f);
}

template <typename Func>
auto vectorize(Func &&f) -> decltype(
        vectorize(std::forward<Func>(f), (typename detail::remove_class<decltype(&std::remove_reference<Func>::type::operator())>::type *) nullptr)) {
    return vectorize(std::forward<Func>(f), (typename detail::remove_class<decltype(
                   &std::remove_reference<Func>::type::operator())>::type *) nullptr);
}

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
