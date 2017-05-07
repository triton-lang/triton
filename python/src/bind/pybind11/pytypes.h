/*
    pybind11/typeid.h: Convenience wrapper classes for basic Python types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"
#include <utility>
#include <type_traits>

NAMESPACE_BEGIN(pybind11)

/* A few forward declarations */
class handle; class object;
class str; class iterator;
struct arg; struct arg_v;

NAMESPACE_BEGIN(detail)
class args_proxy;
inline bool isinstance_generic(handle obj, const std::type_info &tp);

// Accessor forward declarations
template <typename Policy> class accessor;
namespace accessor_policies {
    struct obj_attr;
    struct str_attr;
    struct generic_item;
    struct sequence_item;
    struct list_item;
    struct tuple_item;
}
using obj_attr_accessor = accessor<accessor_policies::obj_attr>;
using str_attr_accessor = accessor<accessor_policies::str_attr>;
using item_accessor = accessor<accessor_policies::generic_item>;
using sequence_accessor = accessor<accessor_policies::sequence_item>;
using list_accessor = accessor<accessor_policies::list_item>;
using tuple_accessor = accessor<accessor_policies::tuple_item>;

/// Tag and check to identify a class which implements the Python object API
class pyobject_tag { };
template <typename T> using is_pyobject = std::is_base_of<pyobject_tag, typename std::remove_reference<T>::type>;

/// Mixin which adds common functions to handle, object and various accessors.
/// The only requirement for `Derived` is to implement `PyObject *Derived::ptr() const`.
template <typename Derived>
class object_api : public pyobject_tag {
    const Derived &derived() const { return static_cast<const Derived &>(*this); }

public:
    iterator begin() const;
    iterator end() const;
    item_accessor operator[](handle key) const;
    item_accessor operator[](const char *key) const;
    obj_attr_accessor attr(handle key) const;
    str_attr_accessor attr(const char *key) const;
    args_proxy operator*() const;
    template <typename T> bool contains(T &&key) const;

    template <return_value_policy policy = return_value_policy::automatic_reference, typename... Args>
    object operator()(Args &&...args) const;
    template <return_value_policy policy = return_value_policy::automatic_reference, typename... Args>
    PYBIND11_DEPRECATED("call(...) was deprecated in favor of operator()(...)")
        object call(Args&&... args) const;

    bool is_none() const { return derived().ptr() == Py_None; }
    PYBIND11_DEPRECATED("Instead of obj.str(), use py::str(obj)")
    pybind11::str str() const;

    int ref_count() const { return static_cast<int>(Py_REFCNT(derived().ptr())); }
    handle get_type() const;
};

NAMESPACE_END(detail)

/// Holds a reference to a Python object (no reference counting)
class handle : public detail::object_api<handle> {
public:
    handle() = default;
    handle(PyObject *ptr) : m_ptr(ptr) { } // Allow implicit conversion from PyObject*

    PyObject *ptr() const { return m_ptr; }
    PyObject *&ptr() { return m_ptr; }
    const handle& inc_ref() const { Py_XINCREF(m_ptr); return *this; }
    const handle& dec_ref() const { Py_XDECREF(m_ptr); return *this; }

    template <typename T> T cast() const;
    explicit operator bool() const { return m_ptr != nullptr; }
    bool operator==(const handle &h) const { return m_ptr == h.m_ptr; }
    bool operator!=(const handle &h) const { return m_ptr != h.m_ptr; }
    PYBIND11_DEPRECATED("Use handle::operator bool() instead")
    bool check() const { return m_ptr != nullptr; }
protected:
    PyObject *m_ptr = nullptr;
};

/// Holds a reference to a Python object (with reference counting)
class object : public handle {
public:
    object() = default;
    PYBIND11_DEPRECATED("Use reinterpret_borrow<object>() or reinterpret_steal<object>()")
    object(handle h, bool is_borrowed) : handle(h) { if (is_borrowed) inc_ref(); }
    object(const object &o) : handle(o) { inc_ref(); }
    object(object &&other) noexcept { m_ptr = other.m_ptr; other.m_ptr = nullptr; }
    ~object() { dec_ref(); }

    handle release() {
      PyObject *tmp = m_ptr;
      m_ptr = nullptr;
      return handle(tmp);
    }

    object& operator=(const object &other) {
        other.inc_ref();
        dec_ref();
        m_ptr = other.m_ptr;
        return *this;
    }

    object& operator=(object &&other) noexcept {
        if (this != &other) {
            handle temp(m_ptr);
            m_ptr = other.m_ptr;
            other.m_ptr = nullptr;
            temp.dec_ref();
        }
        return *this;
    }

    // Calling cast() on an object lvalue just copies (via handle::cast)
    template <typename T> T cast() const &;
    // Calling on an object rvalue does a move, if needed and/or possible
    template <typename T> T cast() &&;

protected:
    // Tags for choosing constructors from raw PyObject *
    struct borrowed_t { }; static constexpr borrowed_t borrowed{};
    struct stolen_t { }; static constexpr stolen_t stolen{};

    template <typename T> friend T reinterpret_borrow(handle);
    template <typename T> friend T reinterpret_steal(handle);

public:
    // Only accessible from derived classes and the reinterpret_* functions
    object(handle h, borrowed_t) : handle(h) { inc_ref(); }
    object(handle h, stolen_t) : handle(h) { }
};

/** The following functions don't do any kind of conversion, they simply declare
    that a PyObject is a certain type and borrow or steal the reference. */
template <typename T> T reinterpret_borrow(handle h) { return {h, object::borrowed}; }
template <typename T> T reinterpret_steal(handle h) { return {h, object::stolen}; }

/// Check if `obj` is an instance of type `T`
template <typename T, detail::enable_if_t<std::is_base_of<object, T>::value, int> = 0>
bool isinstance(handle obj) { return T::_check(obj); }

template <typename T, detail::enable_if_t<!std::is_base_of<object, T>::value, int> = 0>
bool isinstance(handle obj) { return detail::isinstance_generic(obj, typeid(T)); }

template <> inline bool isinstance<handle>(handle obj) = delete;
template <> inline bool isinstance<object>(handle obj) { return obj.ptr() != nullptr; }

inline bool isinstance(handle obj, handle type) {
    const auto result = PyObject_IsInstance(obj.ptr(), type.ptr());
    if (result == -1)
        throw error_already_set();
    return result != 0;
}

inline bool hasattr(handle obj, handle name) {
    return PyObject_HasAttr(obj.ptr(), name.ptr()) == 1;
}

inline bool hasattr(handle obj, const char *name) {
    return PyObject_HasAttrString(obj.ptr(), name) == 1;
}

inline object getattr(handle obj, handle name) {
    PyObject *result = PyObject_GetAttr(obj.ptr(), name.ptr());
    if (!result) { throw error_already_set(); }
    return reinterpret_steal<object>(result);
}

inline object getattr(handle obj, const char *name) {
    PyObject *result = PyObject_GetAttrString(obj.ptr(), name);
    if (!result) { throw error_already_set(); }
    return reinterpret_steal<object>(result);
}

inline object getattr(handle obj, handle name, handle default_) {
    if (PyObject *result = PyObject_GetAttr(obj.ptr(), name.ptr())) {
        return reinterpret_steal<object>(result);
    } else {
        PyErr_Clear();
        return reinterpret_borrow<object>(default_);
    }
}

inline object getattr(handle obj, const char *name, handle default_) {
    if (PyObject *result = PyObject_GetAttrString(obj.ptr(), name)) {
        return reinterpret_steal<object>(result);
    } else {
        PyErr_Clear();
        return reinterpret_borrow<object>(default_);
    }
}

inline void setattr(handle obj, handle name, handle value) {
    if (PyObject_SetAttr(obj.ptr(), name.ptr(), value.ptr()) != 0) { throw error_already_set(); }
}

inline void setattr(handle obj, const char *name, handle value) {
    if (PyObject_SetAttrString(obj.ptr(), name, value.ptr()) != 0) { throw error_already_set(); }
}

NAMESPACE_BEGIN(detail)
inline handle get_function(handle value) {
    if (value) {
#if PY_MAJOR_VERSION >= 3
        if (PyInstanceMethod_Check(value.ptr()))
            value = PyInstanceMethod_GET_FUNCTION(value.ptr());
#endif
        if (PyMethod_Check(value.ptr()))
            value = PyMethod_GET_FUNCTION(value.ptr());
    }
    return value;
}

// Helper aliases/functions to support implicit casting of values given to python accessors/methods.
// When given a pyobject, this simply returns the pyobject as-is; for other C++ type, the value goes
// through pybind11::cast(obj) to convert it to an `object`.
template <typename T, enable_if_t<is_pyobject<T>::value, int> = 0>
auto object_or_cast(T &&o) -> decltype(std::forward<T>(o)) { return std::forward<T>(o); }
// The following casting version is implemented in cast.h:
template <typename T, enable_if_t<!is_pyobject<T>::value, int> = 0>
object object_or_cast(T &&o);
// Match a PyObject*, which we want to convert directly to handle via its converting constructor
inline handle object_or_cast(PyObject *ptr) { return ptr; }


template <typename Policy>
class accessor : public object_api<accessor<Policy>> {
    using key_type = typename Policy::key_type;

public:
    accessor(handle obj, key_type key) : obj(obj), key(std::move(key)) { }

    // accessor overload required to override default assignment operator (templates are not allowed
    // to replace default compiler-generated assignments).
    void operator=(const accessor &a) && { std::move(*this).operator=(handle(a)); }
    void operator=(const accessor &a) & { operator=(handle(a)); }

    template <typename T> void operator=(T &&value) && {
        Policy::set(obj, key, object_or_cast(std::forward<T>(value)));
    }
    template <typename T> void operator=(T &&value) & {
        get_cache() = reinterpret_borrow<object>(object_or_cast(std::forward<T>(value)));
    }

    template <typename T = Policy>
    PYBIND11_DEPRECATED("Use of obj.attr(...) as bool is deprecated in favor of pybind11::hasattr(obj, ...)")
    explicit operator enable_if_t<std::is_same<T, accessor_policies::str_attr>::value ||
            std::is_same<T, accessor_policies::obj_attr>::value, bool>() const {
        return hasattr(obj, key);
    }
    template <typename T = Policy>
    PYBIND11_DEPRECATED("Use of obj[key] as bool is deprecated in favor of obj.contains(key)")
    explicit operator enable_if_t<std::is_same<T, accessor_policies::generic_item>::value, bool>() const {
        return obj.contains(key);
    }

    operator object() const { return get_cache(); }
    PyObject *ptr() const { return get_cache().ptr(); }
    template <typename T> T cast() const { return get_cache().template cast<T>(); }

private:
    object &get_cache() const {
        if (!cache) { cache = Policy::get(obj, key); }
        return cache;
    }

private:
    handle obj;
    key_type key;
    mutable object cache;
};

NAMESPACE_BEGIN(accessor_policies)
struct obj_attr {
    using key_type = object;
    static object get(handle obj, handle key) { return getattr(obj, key); }
    static void set(handle obj, handle key, handle val) { setattr(obj, key, val); }
};

struct str_attr {
    using key_type = const char *;
    static object get(handle obj, const char *key) { return getattr(obj, key); }
    static void set(handle obj, const char *key, handle val) { setattr(obj, key, val); }
};

struct generic_item {
    using key_type = object;

    static object get(handle obj, handle key) {
        PyObject *result = PyObject_GetItem(obj.ptr(), key.ptr());
        if (!result) { throw error_already_set(); }
        return reinterpret_steal<object>(result);
    }

    static void set(handle obj, handle key, handle val) {
        if (PyObject_SetItem(obj.ptr(), key.ptr(), val.ptr()) != 0) { throw error_already_set(); }
    }
};

struct sequence_item {
    using key_type = size_t;

    static object get(handle obj, size_t index) {
        PyObject *result = PySequence_GetItem(obj.ptr(), static_cast<ssize_t>(index));
        if (!result) { throw error_already_set(); }
        return reinterpret_borrow<object>(result);
    }

    static void set(handle obj, size_t index, handle val) {
        // PySequence_SetItem does not steal a reference to 'val'
        if (PySequence_SetItem(obj.ptr(), static_cast<ssize_t>(index), val.ptr()) != 0) {
            throw error_already_set();
        }
    }
};

struct list_item {
    using key_type = size_t;

    static object get(handle obj, size_t index) {
        PyObject *result = PyList_GetItem(obj.ptr(), static_cast<ssize_t>(index));
        if (!result) { throw error_already_set(); }
        return reinterpret_borrow<object>(result);
    }

    static void set(handle obj, size_t index, handle val) {
        // PyList_SetItem steals a reference to 'val'
        if (PyList_SetItem(obj.ptr(), static_cast<ssize_t>(index), val.inc_ref().ptr()) != 0) {
            throw error_already_set();
        }
    }
};

struct tuple_item {
    using key_type = size_t;

    static object get(handle obj, size_t index) {
        PyObject *result = PyTuple_GetItem(obj.ptr(), static_cast<ssize_t>(index));
        if (!result) { throw error_already_set(); }
        return reinterpret_borrow<object>(result);
    }

    static void set(handle obj, size_t index, handle val) {
        // PyTuple_SetItem steals a reference to 'val'
        if (PyTuple_SetItem(obj.ptr(), static_cast<ssize_t>(index), val.inc_ref().ptr()) != 0) {
            throw error_already_set();
        }
    }
};
NAMESPACE_END(accessor_policies)

struct dict_iterator {
public:
    explicit dict_iterator(handle dict = handle(), ssize_t pos = -1) : dict(dict), pos(pos) { }
    dict_iterator& operator++() {
        if (!PyDict_Next(dict.ptr(), &pos, &key.ptr(), &value.ptr()))
            pos = -1;
        return *this;
    }
    std::pair<handle, handle> operator*() const {
        return std::make_pair(key, value);
    }
    bool operator==(const dict_iterator &it) const { return it.pos == pos; }
    bool operator!=(const dict_iterator &it) const { return it.pos != pos; }
private:
    handle dict, key, value;
    ssize_t pos = 0;
};

inline bool PyIterable_Check(PyObject *obj) {
    PyObject *iter = PyObject_GetIter(obj);
    if (iter) {
        Py_DECREF(iter);
        return true;
    } else {
        PyErr_Clear();
        return false;
    }
}

inline bool PyNone_Check(PyObject *o) { return o == Py_None; }

inline bool PyUnicode_Check_Permissive(PyObject *o) { return PyUnicode_Check(o) || PYBIND11_BYTES_CHECK(o); }

class kwargs_proxy : public handle {
public:
    explicit kwargs_proxy(handle h) : handle(h) { }
};

class args_proxy : public handle {
public:
    explicit args_proxy(handle h) : handle(h) { }
    kwargs_proxy operator*() const { return kwargs_proxy(*this); }
};

/// Python argument categories (using PEP 448 terms)
template <typename T> using is_keyword = std::is_base_of<arg, T>;
template <typename T> using is_s_unpacking = std::is_same<args_proxy, T>; // * unpacking
template <typename T> using is_ds_unpacking = std::is_same<kwargs_proxy, T>; // ** unpacking
template <typename T> using is_positional = none_of<
    is_keyword<T>, is_s_unpacking<T>, is_ds_unpacking<T>
>;
template <typename T> using is_keyword_or_ds = any_of<is_keyword<T>, is_ds_unpacking<T>>;

// Call argument collector forward declarations
template <return_value_policy policy = return_value_policy::automatic_reference>
class simple_collector;
template <return_value_policy policy = return_value_policy::automatic_reference>
class unpacking_collector;

NAMESPACE_END(detail)

// TODO: After the deprecated constructors are removed, this macro can be simplified by
//       inheriting ctors: `using Parent::Parent`. It's not an option right now because
//       the `using` statement triggers the parent deprecation warning even if the ctor
//       isn't even used.
#define PYBIND11_OBJECT_COMMON(Name, Parent, CheckFun) \
    public: \
        PYBIND11_DEPRECATED("Use reinterpret_borrow<"#Name">() or reinterpret_steal<"#Name">()") \
        Name(handle h, bool is_borrowed) : Parent(is_borrowed ? Parent(h, borrowed) : Parent(h, stolen)) { } \
        Name(handle h, borrowed_t) : Parent(h, borrowed) { } \
        Name(handle h, stolen_t) : Parent(h, stolen) { } \
        PYBIND11_DEPRECATED("Use py::isinstance<py::python_type>(obj) instead") \
        bool check() const { return m_ptr != nullptr && (bool) CheckFun(m_ptr); } \
        static bool _check(handle h) { return h.ptr() != nullptr && CheckFun(h.ptr()); }

#define PYBIND11_OBJECT_CVT(Name, Parent, CheckFun, ConvertFun) \
    PYBIND11_OBJECT_COMMON(Name, Parent, CheckFun) \
    /* This is deliberately not 'explicit' to allow implicit conversion from object: */ \
    Name(const object &o) : Parent(ConvertFun(o.ptr()), stolen) { if (!m_ptr) throw error_already_set(); }

#define PYBIND11_OBJECT(Name, Parent, CheckFun) \
    PYBIND11_OBJECT_COMMON(Name, Parent, CheckFun) \
    /* This is deliberately not 'explicit' to allow implicit conversion from object: */ \
    Name(const object &o) : Parent(o) { } \
    Name(object &&o) : Parent(std::move(o)) { }

#define PYBIND11_OBJECT_DEFAULT(Name, Parent, CheckFun) \
    PYBIND11_OBJECT(Name, Parent, CheckFun) \
    Name() : Parent() { }

class iterator : public object {
public:
    /** Caveat: copying an iterator does not (and cannot) clone the internal
        state of the Python iterable */
    PYBIND11_OBJECT_DEFAULT(iterator, object, PyIter_Check)

    iterator& operator++() {
        if (m_ptr)
            advance();
        return *this;
    }

    /** Caveat: this postincrement operator does not (and cannot) clone the
        internal state of the Python iterable. It should only be used to
        retrieve the current iterate using <tt>operator*()</tt> */
    iterator operator++(int) {
        iterator rv(*this);
        rv.value = value;
        if (m_ptr)
            advance();
        return rv;
    }

    bool operator==(const iterator &it) const { return *it == **this; }
    bool operator!=(const iterator &it) const { return *it != **this; }

    handle operator*() const {
        if (!ready && m_ptr) {
            auto& self = const_cast<iterator &>(*this);
            self.advance();
            self.ready = true;
        }
        return value;
    }

private:
    void advance() { value = reinterpret_steal<object>(PyIter_Next(m_ptr)); }

private:
    object value = {};
    bool ready = false;
};

class iterable : public object {
public:
    PYBIND11_OBJECT_DEFAULT(iterable, object, detail::PyIterable_Check)
};

class bytes;

class str : public object {
public:
    PYBIND11_OBJECT_CVT(str, object, detail::PyUnicode_Check_Permissive, raw_str)

    str(const char *c, size_t n)
        : object(PyUnicode_FromStringAndSize(c, (ssize_t) n), stolen) {
        if (!m_ptr) pybind11_fail("Could not allocate string object!");
    }

    // 'explicit' is explicitly omitted from the following constructors to allow implicit conversion to py::str from C++ string-like objects
    str(const char *c = "")
        : object(PyUnicode_FromString(c), stolen) {
        if (!m_ptr) pybind11_fail("Could not allocate string object!");
    }

    str(const std::string &s) : str(s.data(), s.size()) { }

    explicit str(const bytes &b);

    explicit str(handle h) : object(raw_str(h.ptr()), stolen) { }

    operator std::string() const {
        object temp = *this;
        if (PyUnicode_Check(m_ptr)) {
            temp = reinterpret_steal<object>(PyUnicode_AsUTF8String(m_ptr));
            if (!temp)
                pybind11_fail("Unable to extract string contents! (encoding issue)");
        }
        char *buffer;
        ssize_t length;
        if (PYBIND11_BYTES_AS_STRING_AND_SIZE(temp.ptr(), &buffer, &length))
            pybind11_fail("Unable to extract string contents! (invalid type)");
        return std::string(buffer, (size_t) length);
    }

    template <typename... Args>
    str format(Args &&...args) const {
        return attr("format")(std::forward<Args>(args)...);
    }

private:
    /// Return string representation -- always returns a new reference, even if already a str
    static PyObject *raw_str(PyObject *op) {
        PyObject *str_value = PyObject_Str(op);
#if PY_MAJOR_VERSION < 3
        if (!str_value) throw error_already_set();
        PyObject *unicode = PyUnicode_FromEncodedObject(str_value, "utf-8", nullptr);
        Py_XDECREF(str_value); str_value = unicode;
#endif
        return str_value;
    }
};

inline namespace literals {
/// String literal version of str
inline str operator"" _s(const char *s, size_t size) { return {s, size}; }
}

class bytes : public object {
public:
    PYBIND11_OBJECT(bytes, object, PYBIND11_BYTES_CHECK)

    // Allow implicit conversion:
    bytes(const char *c = "")
        : object(PYBIND11_BYTES_FROM_STRING(c), stolen) {
        if (!m_ptr) pybind11_fail("Could not allocate bytes object!");
    }

    bytes(const char *c, size_t n)
        : object(PYBIND11_BYTES_FROM_STRING_AND_SIZE(c, (ssize_t) n), stolen) {
        if (!m_ptr) pybind11_fail("Could not allocate bytes object!");
    }

    // Allow implicit conversion:
    bytes(const std::string &s) : bytes(s.data(), s.size()) { }

    explicit bytes(const pybind11::str &s);

    operator std::string() const {
        char *buffer;
        ssize_t length;
        if (PYBIND11_BYTES_AS_STRING_AND_SIZE(m_ptr, &buffer, &length))
            pybind11_fail("Unable to extract bytes contents!");
        return std::string(buffer, (size_t) length);
    }
};

inline bytes::bytes(const pybind11::str &s) {
    object temp = s;
    if (PyUnicode_Check(s.ptr())) {
        temp = reinterpret_steal<object>(PyUnicode_AsUTF8String(s.ptr()));
        if (!temp)
            pybind11_fail("Unable to extract string contents! (encoding issue)");
    }
    char *buffer;
    ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(temp.ptr(), &buffer, &length))
        pybind11_fail("Unable to extract string contents! (invalid type)");
    auto obj = reinterpret_steal<object>(PYBIND11_BYTES_FROM_STRING_AND_SIZE(buffer, length));
    if (!obj)
        pybind11_fail("Could not allocate bytes object!");
    m_ptr = obj.release().ptr();
}

inline str::str(const bytes& b) {
    char *buffer;
    ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(b.ptr(), &buffer, &length))
        pybind11_fail("Unable to extract bytes contents!");
    auto obj = reinterpret_steal<object>(PyUnicode_FromStringAndSize(buffer, (ssize_t) length));
    if (!obj)
        pybind11_fail("Could not allocate string object!");
    m_ptr = obj.release().ptr();
}

class none : public object {
public:
    PYBIND11_OBJECT(none, object, detail::PyNone_Check)
    none() : object(Py_None, borrowed) { }
};

class bool_ : public object {
public:
    PYBIND11_OBJECT_CVT(bool_, object, PyBool_Check, raw_bool)
    bool_() : object(Py_False, borrowed) { }
    // Allow implicit conversion from and to `bool`:
    bool_(bool value) : object(value ? Py_True : Py_False, borrowed) { }
    operator bool() const { return m_ptr && PyLong_AsLong(m_ptr) != 0; }

private:
    /// Return the truth value of an object -- always returns a new reference
    static PyObject *raw_bool(PyObject *op) {
        const auto value = PyObject_IsTrue(op);
        if (value == -1) return nullptr;
        return handle(value ? Py_True : Py_False).inc_ref().ptr();
    }
};

class int_ : public object {
public:
    PYBIND11_OBJECT_CVT(int_, object, PYBIND11_LONG_CHECK, PyNumber_Long)
    int_() : object(PyLong_FromLong(0), stolen) { }
    // Allow implicit conversion from C++ integral types:
    template <typename T,
              detail::enable_if_t<std::is_integral<T>::value, int> = 0>
    int_(T value) {
        if (sizeof(T) <= sizeof(long)) {
            if (std::is_signed<T>::value)
                m_ptr = PyLong_FromLong((long) value);
            else
                m_ptr = PyLong_FromUnsignedLong((unsigned long) value);
        } else {
            if (std::is_signed<T>::value)
                m_ptr = PyLong_FromLongLong((long long) value);
            else
                m_ptr = PyLong_FromUnsignedLongLong((unsigned long long) value);
        }
        if (!m_ptr) pybind11_fail("Could not allocate int object!");
    }

    template <typename T,
              detail::enable_if_t<std::is_integral<T>::value, int> = 0>
    operator T() const {
        if (sizeof(T) <= sizeof(long)) {
            if (std::is_signed<T>::value)
                return (T) PyLong_AsLong(m_ptr);
            else
                return (T) PyLong_AsUnsignedLong(m_ptr);
        } else {
            if (std::is_signed<T>::value)
                return (T) PYBIND11_LONG_AS_LONGLONG(m_ptr);
            else
                return (T) PYBIND11_LONG_AS_UNSIGNED_LONGLONG(m_ptr);
        }
    }
};

class float_ : public object {
public:
    PYBIND11_OBJECT_CVT(float_, object, PyFloat_Check, PyNumber_Float)
    // Allow implicit conversion from float/double:
    float_(float value) : object(PyFloat_FromDouble((double) value), stolen) {
        if (!m_ptr) pybind11_fail("Could not allocate float object!");
    }
    float_(double value = .0) : object(PyFloat_FromDouble((double) value), stolen) {
        if (!m_ptr) pybind11_fail("Could not allocate float object!");
    }
    operator float() const { return (float) PyFloat_AsDouble(m_ptr); }
    operator double() const { return (double) PyFloat_AsDouble(m_ptr); }
};

class weakref : public object {
public:
    PYBIND11_OBJECT_DEFAULT(weakref, object, PyWeakref_Check)
    explicit weakref(handle obj, handle callback = {})
        : object(PyWeakref_NewRef(obj.ptr(), callback.ptr()), stolen) {
        if (!m_ptr) pybind11_fail("Could not allocate weak reference!");
    }
};

class slice : public object {
public:
    PYBIND11_OBJECT_DEFAULT(slice, object, PySlice_Check)
    slice(ssize_t start_, ssize_t stop_, ssize_t step_) {
        int_ start(start_), stop(stop_), step(step_);
        m_ptr = PySlice_New(start.ptr(), stop.ptr(), step.ptr());
        if (!m_ptr) pybind11_fail("Could not allocate slice object!");
    }
    bool compute(size_t length, size_t *start, size_t *stop, size_t *step,
                 size_t *slicelength) const {
        return PySlice_GetIndicesEx((PYBIND11_SLICE_OBJECT *) m_ptr,
                                    (ssize_t) length, (ssize_t *) start,
                                    (ssize_t *) stop, (ssize_t *) step,
                                    (ssize_t *) slicelength) == 0;
    }
};

class capsule : public object {
public:
    PYBIND11_OBJECT_DEFAULT(capsule, object, PyCapsule_CheckExact)
    PYBIND11_DEPRECATED("Use reinterpret_borrow<capsule>() or reinterpret_steal<capsule>()")
    capsule(PyObject *ptr, bool is_borrowed) : object(is_borrowed ? object(ptr, borrowed) : object(ptr, stolen)) { }
    explicit capsule(const void *value, void (*destruct)(PyObject *) = nullptr)
        : object(PyCapsule_New(const_cast<void*>(value), nullptr, destruct), stolen) {
        if (!m_ptr) pybind11_fail("Could not allocate capsule object!");
    }
    template <typename T> operator T *() const {
        T * result = static_cast<T *>(PyCapsule_GetPointer(m_ptr, nullptr));
        if (!result) pybind11_fail("Unable to extract capsule contents!");
        return result;
    }
};

class tuple : public object {
public:
    PYBIND11_OBJECT_CVT(tuple, object, PyTuple_Check, PySequence_Tuple)
    explicit tuple(size_t size = 0) : object(PyTuple_New((ssize_t) size), stolen) {
        if (!m_ptr) pybind11_fail("Could not allocate tuple object!");
    }
    size_t size() const { return (size_t) PyTuple_Size(m_ptr); }
    detail::tuple_accessor operator[](size_t index) const { return {*this, index}; }
};

class dict : public object {
public:
    PYBIND11_OBJECT_CVT(dict, object, PyDict_Check, raw_dict)
    dict() : object(PyDict_New(), stolen) {
        if (!m_ptr) pybind11_fail("Could not allocate dict object!");
    }
    template <typename... Args,
              typename = detail::enable_if_t<detail::all_of<detail::is_keyword_or_ds<Args>...>::value>,
              // MSVC workaround: it can't compile an out-of-line definition, so defer the collector
              typename collector = detail::deferred_t<detail::unpacking_collector<>, Args...>>
    explicit dict(Args &&...args) : dict(collector(std::forward<Args>(args)...).kwargs()) { }

    size_t size() const { return (size_t) PyDict_Size(m_ptr); }
    detail::dict_iterator begin() const { return (++detail::dict_iterator(*this, 0)); }
    detail::dict_iterator end() const { return detail::dict_iterator(); }
    void clear() const { PyDict_Clear(ptr()); }
    bool contains(handle key) const { return PyDict_Contains(ptr(), key.ptr()) == 1; }
    bool contains(const char *key) const { return PyDict_Contains(ptr(), pybind11::str(key).ptr()) == 1; }

private:
    /// Call the `dict` Python type -- always returns a new reference
    static PyObject *raw_dict(PyObject *op) {
        if (PyDict_Check(op))
            return handle(op).inc_ref().ptr();
        return PyObject_CallFunctionObjArgs((PyObject *) &PyDict_Type, op, nullptr);
    }
};

class sequence : public object {
public:
    PYBIND11_OBJECT(sequence, object, PySequence_Check)
    size_t size() const { return (size_t) PySequence_Size(m_ptr); }
    detail::sequence_accessor operator[](size_t index) const { return {*this, index}; }
};

class list : public object {
public:
    PYBIND11_OBJECT_CVT(list, object, PyList_Check, PySequence_List)
    explicit list(size_t size = 0) : object(PyList_New((ssize_t) size), stolen) {
        if (!m_ptr) pybind11_fail("Could not allocate list object!");
    }
    size_t size() const { return (size_t) PyList_Size(m_ptr); }
    detail::list_accessor operator[](size_t index) const { return {*this, index}; }
    template <typename T> void append(T &&val) const {
        PyList_Append(m_ptr, detail::object_or_cast(std::forward<T>(val)).ptr());
    }
};

class args : public tuple { PYBIND11_OBJECT_DEFAULT(args, tuple, PyTuple_Check) };
class kwargs : public dict { PYBIND11_OBJECT_DEFAULT(kwargs, dict, PyDict_Check)  };

class set : public object {
public:
    PYBIND11_OBJECT_CVT(set, object, PySet_Check, PySet_New)
    set() : object(PySet_New(nullptr), stolen) {
        if (!m_ptr) pybind11_fail("Could not allocate set object!");
    }
    size_t size() const { return (size_t) PySet_Size(m_ptr); }
    template <typename T> bool add(T &&val) const {
        return PySet_Add(m_ptr, detail::object_or_cast(std::forward<T>(val)).ptr()) == 0;
    }
    void clear() const { PySet_Clear(m_ptr); }
};

class function : public object {
public:
    PYBIND11_OBJECT_DEFAULT(function, object, PyCallable_Check)
    bool is_cpp_function() const {
        handle fun = detail::get_function(m_ptr);
        return fun && PyCFunction_Check(fun.ptr());
    }
};

class buffer : public object {
public:
    PYBIND11_OBJECT_DEFAULT(buffer, object, PyObject_CheckBuffer)

    buffer_info request(bool writable = false) {
        int flags = PyBUF_STRIDES | PyBUF_FORMAT;
        if (writable) flags |= PyBUF_WRITABLE;
        Py_buffer *view = new Py_buffer();
        if (PyObject_GetBuffer(m_ptr, view, flags) != 0)
            throw error_already_set();
        return buffer_info(view);
    }
};

class memoryview : public object {
public:
    explicit memoryview(const buffer_info& info) {
        static Py_buffer buf { };
        // Py_buffer uses signed sizes, strides and shape!..
        static std::vector<Py_ssize_t> py_strides { };
        static std::vector<Py_ssize_t> py_shape { };
        buf.buf = info.ptr;
        buf.itemsize = (Py_ssize_t) info.itemsize;
        buf.format = const_cast<char *>(info.format.c_str());
        buf.ndim = (int) info.ndim;
        buf.len = (Py_ssize_t) info.size;
        py_strides.clear();
        py_shape.clear();
        for (size_t i = 0; i < info.ndim; ++i) {
            py_strides.push_back((Py_ssize_t) info.strides[i]);
            py_shape.push_back((Py_ssize_t) info.shape[i]);
        }
        buf.strides = py_strides.data();
        buf.shape = py_shape.data();
        buf.suboffsets = nullptr;
        buf.readonly = false;
        buf.internal = nullptr;

        m_ptr = PyMemoryView_FromBuffer(&buf);
        if (!m_ptr)
            pybind11_fail("Unable to create memoryview from buffer descriptor");
    }

    PYBIND11_OBJECT_CVT(memoryview, object, PyMemoryView_Check, PyMemoryView_FromObject)
};

inline size_t len(handle h) {
    ssize_t result = PyObject_Length(h.ptr());
    if (result < 0)
        pybind11_fail("Unable to compute length of object");
    return (size_t) result;
}

inline str repr(handle h) {
    PyObject *str_value = PyObject_Repr(h.ptr());
    if (!str_value) throw error_already_set();
#if PY_MAJOR_VERSION < 3
    PyObject *unicode = PyUnicode_FromEncodedObject(str_value, "utf-8", nullptr);
    Py_XDECREF(str_value); str_value = unicode;
    if (!str_value) throw error_already_set();
#endif
    return reinterpret_steal<str>(str_value);
}

NAMESPACE_BEGIN(detail)
template <typename D> iterator object_api<D>::begin() const {
    return reinterpret_steal<iterator>(PyObject_GetIter(derived().ptr()));
}
template <typename D> iterator object_api<D>::end() const {
    return {};
}
template <typename D> item_accessor object_api<D>::operator[](handle key) const {
    return {derived(), reinterpret_borrow<object>(key)};
}
template <typename D> item_accessor object_api<D>::operator[](const char *key) const {
    return {derived(), pybind11::str(key)};
}
template <typename D> obj_attr_accessor object_api<D>::attr(handle key) const {
    return {derived(), reinterpret_borrow<object>(key)};
}
template <typename D> str_attr_accessor object_api<D>::attr(const char *key) const {
    return {derived(), key};
}
template <typename D> args_proxy object_api<D>::operator*() const {
    return args_proxy(derived().ptr());
}
template <typename D> template <typename T> bool object_api<D>::contains(T &&key) const {
    return attr("__contains__")(std::forward<T>(key)).template cast<bool>();
}

template <typename D>
pybind11::str object_api<D>::str() const { return pybind11::str(derived()); }

template <typename D>
handle object_api<D>::get_type() const { return (PyObject *) Py_TYPE(derived().ptr()); }

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
