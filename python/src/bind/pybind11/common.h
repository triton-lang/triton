/*
    pybind11/common.h -- Basic macros

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#if !defined(NAMESPACE_BEGIN)
#  define NAMESPACE_BEGIN(name) namespace name {
#endif
#if !defined(NAMESPACE_END)
#  define NAMESPACE_END(name) }
#endif

// Neither MSVC nor Intel support enough of C++14 yet (in particular, as of MSVC 2015 and ICC 17
// beta, neither support extended constexpr, which we rely on in descr.h), so don't enable pybind
// CPP14 features for them.
#if !defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#  if __cplusplus >= 201402L
#    define PYBIND11_CPP14
#    if __cplusplus > 201402L /* Temporary: should be updated to >= the final C++17 value once known */
#      define PYBIND11_CPP17
#    endif
#  endif
#endif

// Compiler version assertions
#if defined(__INTEL_COMPILER)
#  if __INTEL_COMPILER < 1500
#    error pybind11 requires Intel C++ compiler v15 or newer
#  endif
#elif defined(__clang__) && !defined(__apple_build_version__)
#  if __clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ < 3)
#    error pybind11 requires clang 3.3 or newer
#  endif
#elif defined(__clang__)
// Apple changes clang version macros to its Xcode version; the first Xcode release based on
// (upstream) clang 3.3 was Xcode 5:
#  if __clang_major__ < 5
#    error pybind11 requires Xcode/clang 5.0 or newer
#  endif
#elif defined(__GNUG__)
#  if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8)
#    error pybind11 requires gcc 4.8 or newer
#  endif
#elif defined(_MSC_VER)
// Pybind hits various compiler bugs in 2015u2 and earlier, and also makes use of some stl features
// (e.g. std::negation) added in 2015u3:
#  if _MSC_FULL_VER < 190024213
#    error pybind11 requires MSVC 2015 update 3 or newer
#  endif
#endif

#if !defined(PYBIND11_EXPORT)
#  if defined(WIN32) || defined(_WIN32)
#    define PYBIND11_EXPORT __declspec(dllexport)
#  else
#    define PYBIND11_EXPORT __attribute__ ((visibility("default")))
#  endif
#endif

#if defined(_MSC_VER)
#  define PYBIND11_NOINLINE __declspec(noinline)
#else
#  define PYBIND11_NOINLINE __attribute__ ((noinline))
#endif

#if defined(PYBIND11_CPP14)
#  define PYBIND11_DEPRECATED(reason) [[deprecated(reason)]]
#elif defined(__clang__)
#  define PYBIND11_DEPRECATED(reason) __attribute__((deprecated(reason)))
#elif defined(__GNUG__)
#  define PYBIND11_DEPRECATED(reason) __attribute__((deprecated))
#elif defined(_MSC_VER)
#  define PYBIND11_DEPRECATED(reason) __declspec(deprecated)
#endif

#define PYBIND11_VERSION_MAJOR 2
#define PYBIND11_VERSION_MINOR 1
#define PYBIND11_VERSION_PATCH 0

/// Include Python header, disable linking to pythonX_d.lib on Windows in debug mode
#if defined(_MSC_VER)
#  if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 4)
#    define HAVE_ROUND 1
#  endif
#  pragma warning(push)
#  pragma warning(disable: 4510 4610 4512 4005)
#  if defined(_DEBUG)
#    define PYBIND11_DEBUG_MARKER
#    undef _DEBUG
#  endif
#endif

#include <Python.h>
#include <frameobject.h>
#include <pythread.h>

#if defined(_WIN32) && (defined(min) || defined(max))
#  error Macro clash with min and max -- define NOMINMAX when compiling your program on Windows
#endif

#if defined(isalnum)
#  undef isalnum
#  undef isalpha
#  undef islower
#  undef isspace
#  undef isupper
#  undef tolower
#  undef toupper
#endif

#if defined(_MSC_VER)
#  if defined(PYBIND11_DEBUG_MARKER)
#    define _DEBUG
#    undef PYBIND11_DEBUG_MARKER
#  endif
#  pragma warning(pop)
#endif

#include <cstddef>
#include <forward_list>
#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <typeindex>
#include <type_traits>

#if PY_MAJOR_VERSION >= 3 /// Compatibility macros for various Python versions
#define PYBIND11_INSTANCE_METHOD_NEW(ptr, class_) PyInstanceMethod_New(ptr)
#define PYBIND11_BYTES_CHECK PyBytes_Check
#define PYBIND11_BYTES_FROM_STRING PyBytes_FromString
#define PYBIND11_BYTES_FROM_STRING_AND_SIZE PyBytes_FromStringAndSize
#define PYBIND11_BYTES_AS_STRING_AND_SIZE PyBytes_AsStringAndSize
#define PYBIND11_BYTES_AS_STRING PyBytes_AsString
#define PYBIND11_BYTES_SIZE PyBytes_Size
#define PYBIND11_LONG_CHECK(o) PyLong_Check(o)
#define PYBIND11_LONG_AS_LONGLONG(o) PyLong_AsLongLong(o)
#define PYBIND11_LONG_AS_UNSIGNED_LONGLONG(o) PyLong_AsUnsignedLongLong(o)
#define PYBIND11_BYTES_NAME "bytes"
#define PYBIND11_STRING_NAME "str"
#define PYBIND11_SLICE_OBJECT PyObject
#define PYBIND11_FROM_STRING PyUnicode_FromString
#define PYBIND11_STR_TYPE ::pybind11::str
#define PYBIND11_PLUGIN_IMPL(name) \
    extern "C" PYBIND11_EXPORT PyObject *PyInit_##name()
#else
#define PYBIND11_INSTANCE_METHOD_NEW(ptr, class_) PyMethod_New(ptr, nullptr, class_)
#define PYBIND11_BYTES_CHECK PyString_Check
#define PYBIND11_BYTES_FROM_STRING PyString_FromString
#define PYBIND11_BYTES_FROM_STRING_AND_SIZE PyString_FromStringAndSize
#define PYBIND11_BYTES_AS_STRING_AND_SIZE PyString_AsStringAndSize
#define PYBIND11_BYTES_AS_STRING PyString_AsString
#define PYBIND11_BYTES_SIZE PyString_Size
#define PYBIND11_LONG_CHECK(o) (PyInt_Check(o) || PyLong_Check(o))
#define PYBIND11_LONG_AS_LONGLONG(o) (PyInt_Check(o) ? (long long) PyLong_AsLong(o) : PyLong_AsLongLong(o))
#define PYBIND11_LONG_AS_UNSIGNED_LONGLONG(o) (PyInt_Check(o) ? (unsigned long long) PyLong_AsUnsignedLong(o) : PyLong_AsUnsignedLongLong(o))
#define PYBIND11_BYTES_NAME "str"
#define PYBIND11_STRING_NAME "unicode"
#define PYBIND11_SLICE_OBJECT PySliceObject
#define PYBIND11_FROM_STRING PyString_FromString
#define PYBIND11_STR_TYPE ::pybind11::bytes
#define PYBIND11_PLUGIN_IMPL(name) \
    static PyObject *pybind11_init_wrapper();               \
    extern "C" PYBIND11_EXPORT void init##name() {          \
        (void)pybind11_init_wrapper();                      \
    }                                                       \
    PyObject *pybind11_init_wrapper()
#endif

#if PY_VERSION_HEX >= 0x03050000 && PY_VERSION_HEX < 0x03050200
extern "C" {
    struct _Py_atomic_address { void *value; };
    PyAPI_DATA(_Py_atomic_address) _PyThreadState_Current;
}
#endif

#define PYBIND11_TRY_NEXT_OVERLOAD ((PyObject *) 1) // special failure return code
#define PYBIND11_STRINGIFY(x) #x
#define PYBIND11_TOSTRING(x) PYBIND11_STRINGIFY(x)
#define PYBIND11_INTERNALS_ID "__pybind11_" \
    PYBIND11_TOSTRING(PYBIND11_VERSION_MAJOR) "_" PYBIND11_TOSTRING(PYBIND11_VERSION_MINOR) "__"

/** \rst
    This macro creates the entry point that will be invoked when the Python interpreter
    imports a plugin library. Please create a `module` in the function body and return
    the pointer to its underlying Python object at the end.

    .. code-block:: cpp

        PYBIND11_PLUGIN(example) {
            pybind11::module m("example", "pybind11 example plugin");
            /// Set up bindings here
            return m.ptr();
        }
\endrst */
#define PYBIND11_PLUGIN(name)                                                  \
    static PyObject *pybind11_init();                                          \
    PYBIND11_PLUGIN_IMPL(name) {                                               \
        int major, minor;                                                      \
        if (sscanf(Py_GetVersion(), "%i.%i", &major, &minor) != 2) {           \
            PyErr_SetString(PyExc_ImportError, "Can't parse Python version."); \
            return nullptr;                                                    \
        } else if (major != PY_MAJOR_VERSION || minor != PY_MINOR_VERSION) {   \
            PyErr_Format(PyExc_ImportError,                                    \
                         "Python version mismatch: module was compiled for "   \
                         "version %i.%i, while the interpreter is running "    \
                         "version %i.%i.", PY_MAJOR_VERSION, PY_MINOR_VERSION, \
                         major, minor);                                        \
            return nullptr;                                                    \
        }                                                                      \
        try {                                                                  \
            return pybind11_init();                                            \
        } catch (pybind11::error_already_set &e) {                             \
            e.clear();                                                         \
            PyErr_SetString(PyExc_ImportError, e.what());                      \
            return nullptr;                                                    \
        } catch (const std::exception &e) {                                    \
            PyErr_SetString(PyExc_ImportError, e.what());                      \
            return nullptr;                                                    \
        }                                                                      \
    }                                                                          \
    PyObject *pybind11_init()

NAMESPACE_BEGIN(pybind11)

using ssize_t = Py_ssize_t;
using size_t  = std::size_t;

/// Approach used to cast a previously unknown C++ instance into a Python object
enum class return_value_policy : uint8_t {
    /** This is the default return value policy, which falls back to the policy
        return_value_policy::take_ownership when the return value is a pointer.
        Otherwise, it uses return_value::move or return_value::copy for rvalue
        and lvalue references, respectively. See below for a description of what
        all of these different policies do. */
    automatic = 0,

    /** As above, but use policy return_value_policy::reference when the return
        value is a pointer. This is the default conversion policy for function
        arguments when calling Python functions manually from C++ code (i.e. via
        handle::operator()). You probably won't need to use this. */
    automatic_reference,

    /** Reference an existing object (i.e. do not create a new copy) and take
        ownership. Python will call the destructor and delete operator when the
        object’s reference count reaches zero. Undefined behavior ensues when
        the C++ side does the same.. */
    take_ownership,

    /** Create a new copy of the returned object, which will be owned by
        Python. This policy is comparably safe because the lifetimes of the two
        instances are decoupled. */
    copy,

    /** Use std::move to move the return value contents into a new instance
        that will be owned by Python. This policy is comparably safe because the
        lifetimes of the two instances (move source and destination) are
        decoupled. */
    move,

    /** Reference an existing object, but do not take ownership. The C++ side
        is responsible for managing the object’s lifetime and deallocating it
        when it is no longer used. Warning: undefined behavior will ensue when
        the C++ side deletes an object that is still referenced and used by
        Python. */
    reference,

    /** This policy only applies to methods and properties. It references the
        object without taking ownership similar to the above
        return_value_policy::reference policy. In contrast to that policy, the
        function or property’s implicit this argument (called the parent) is
        considered to be the the owner of the return value (the child).
        pybind11 then couples the lifetime of the parent to the child via a
        reference relationship that ensures that the parent cannot be garbage
        collected while Python is still using the child. More advanced
        variations of this scheme are also possible using combinations of
        return_value_policy::reference and the keep_alive call policy */
    reference_internal
};

/// Information record describing a Python buffer object
struct buffer_info {
    void *ptr = nullptr;         // Pointer to the underlying storage
    size_t itemsize = 0;         // Size of individual items in bytes
    size_t size = 0;             // Total number of entries
    std::string format;          // For homogeneous buffers, this should be set to format_descriptor<T>::format()
    size_t ndim = 0;             // Number of dimensions
    std::vector<size_t> shape;   // Shape of the tensor (1 entry per dimension)
    std::vector<size_t> strides; // Number of entries between adjacent entries (for each per dimension)

    buffer_info() { }

    buffer_info(void *ptr, size_t itemsize, const std::string &format, size_t ndim,
                const std::vector<size_t> &shape, const std::vector<size_t> &strides)
        : ptr(ptr), itemsize(itemsize), size(1), format(format),
          ndim(ndim), shape(shape), strides(strides) {
        for (size_t i = 0; i < ndim; ++i)
            size *= shape[i];
    }

    buffer_info(void *ptr, size_t itemsize, const std::string &format, size_t size)
    : buffer_info(ptr, itemsize, format, 1, std::vector<size_t> { size },
                  std::vector<size_t> { itemsize }) { }

    explicit buffer_info(Py_buffer *view, bool ownview = true)
        : ptr(view->buf), itemsize((size_t) view->itemsize), size(1), format(view->format),
          ndim((size_t) view->ndim), shape((size_t) view->ndim), strides((size_t) view->ndim), view(view), ownview(ownview) {
        for (size_t i = 0; i < (size_t) view->ndim; ++i) {
            shape[i] = (size_t) view->shape[i];
            strides[i] = (size_t) view->strides[i];
            size *= shape[i];
        }
    }

    buffer_info(const buffer_info &) = delete;
    buffer_info& operator=(const buffer_info &) = delete;

    buffer_info(buffer_info &&other) {
        (*this) = std::move(other);
    }

    buffer_info& operator=(buffer_info &&rhs) {
        ptr = rhs.ptr;
        itemsize = rhs.itemsize;
        size = rhs.size;
        format = std::move(rhs.format);
        ndim = rhs.ndim;
        shape = std::move(rhs.shape);
        strides = std::move(rhs.strides);
        std::swap(view, rhs.view);
        std::swap(ownview, rhs.ownview);
        return *this;
    }

    ~buffer_info() {
        if (view && ownview) { PyBuffer_Release(view); delete view; }
    }

private:
    Py_buffer *view = nullptr;
    bool ownview = false;
};

NAMESPACE_BEGIN(detail)

inline static constexpr int log2(size_t n, int k = 0) { return (n <= 1) ? k : log2(n >> 1, k + 1); }

inline std::string error_string();

/// Core part of the 'instance' type which POD (needed to be able to use 'offsetof')
template <typename type> struct instance_essentials {
    PyObject_HEAD
    type *value;
    PyObject *weakrefs;
    bool owned : 1;
    bool holder_constructed : 1;
};

/// PyObject wrapper around generic types, includes a special holder type that is responsible for lifetime management
template <typename type, typename holder_type = std::unique_ptr<type>> struct instance : instance_essentials<type> {
    holder_type holder;
};

struct overload_hash {
    inline size_t operator()(const std::pair<const PyObject *, const char *>& v) const {
        size_t value = std::hash<const void *>()(v.first);
        value ^= std::hash<const void *>()(v.second)  + 0x9e3779b9 + (value<<6) + (value>>2);
        return value;
    }
};

/// Internal data structure used to track registered instances and types
struct internals {
    std::unordered_map<std::type_index, void*> registered_types_cpp;   // std::type_index -> type_info
    std::unordered_map<const void *, void*> registered_types_py;       // PyTypeObject* -> type_info
    std::unordered_multimap<const void *, void*> registered_instances; // void * -> PyObject*
    std::unordered_set<std::pair<const PyObject *, const char *>, overload_hash> inactive_overload_cache;
    std::unordered_map<std::type_index, std::vector<bool (*)(PyObject *, void *&)>> direct_conversions;
    std::forward_list<void (*) (std::exception_ptr)> registered_exception_translators;
    std::unordered_map<std::string, void *> shared_data; // Custom data to be shared across extensions
    PyTypeObject *static_property_type;
    PyTypeObject *default_metaclass;
    std::unordered_map<size_t, PyObject *> bases; // one base type per `instance_size` (very few)
#if defined(WITH_THREAD)
    decltype(PyThread_create_key()) tstate = 0; // Usually an int but a long on Cygwin64 with Python 3.x
    PyInterpreterState *istate = nullptr;
#endif

    /// Return the appropriate base type for the given instance size
    PyObject *get_base(size_t instance_size);
};

/// Return a reference to the current 'internals' information
inline internals &get_internals();

/// from __cpp_future__ import (convenient aliases from C++14/17)
#ifdef PYBIND11_CPP14
using std::enable_if_t;
using std::conditional_t;
using std::remove_cv_t;
#else
template <bool B, typename T = void> using enable_if_t = typename std::enable_if<B, T>::type;
template <bool B, typename T, typename F> using conditional_t = typename std::conditional<B, T, F>::type;
template <typename T> using remove_cv_t = typename std::remove_cv<T>::type;
#endif

/// Index sequences
#if defined(PYBIND11_CPP14) || defined(_MSC_VER)
using std::index_sequence;
using std::make_index_sequence;
#else
template<size_t ...> struct index_sequence  { };
template<size_t N, size_t ...S> struct make_index_sequence_impl : make_index_sequence_impl <N - 1, N - 1, S...> { };
template<size_t ...S> struct make_index_sequence_impl <0, S...> { typedef index_sequence<S...> type; };
template<size_t N> using make_index_sequence = typename make_index_sequence_impl<N>::type;
#endif

/// Backports of std::bool_constant and std::negation to accomodate older compilers
template <bool B> using bool_constant = std::integral_constant<bool, B>;
template <typename T> struct negation : bool_constant<!T::value> { };

template <typename...> struct void_t_impl { using type = void; };
template <typename... Ts> using void_t = typename void_t_impl<Ts...>::type;

/// Compile-time all/any/none of that check the boolean value of all template types
#ifdef __cpp_fold_expressions
template <class... Ts> using all_of = bool_constant<(Ts::value && ...)>;
template <class... Ts> using any_of = bool_constant<(Ts::value || ...)>;
#elif !defined(_MSC_VER)
template <bool...> struct bools {};
template <class... Ts> using all_of = std::is_same<
    bools<Ts::value..., true>,
    bools<true, Ts::value...>>;
template <class... Ts> using any_of = negation<all_of<negation<Ts>...>>;
#else
// MSVC has trouble with the above, but supports std::conjunction, which we can use instead (albeit
// at a slight loss of compilation efficiency).
template <class... Ts> using all_of = std::conjunction<Ts...>;
template <class... Ts> using any_of = std::disjunction<Ts...>;
#endif
template <class... Ts> using none_of = negation<any_of<Ts...>>;

template <class T, template<class> class... Predicates> using satisfies_all_of = all_of<Predicates<T>...>;
template <class T, template<class> class... Predicates> using satisfies_any_of = any_of<Predicates<T>...>;
template <class T, template<class> class... Predicates> using satisfies_none_of = none_of<Predicates<T>...>;

/// Strip the class from a method type
template <typename T> struct remove_class { };
template <typename C, typename R, typename... A> struct remove_class<R (C::*)(A...)> { typedef R type(A...); };
template <typename C, typename R, typename... A> struct remove_class<R (C::*)(A...) const> { typedef R type(A...); };

/// Helper template to strip away type modifiers
template <typename T> struct intrinsic_type                       { typedef T type; };
template <typename T> struct intrinsic_type<const T>              { typedef typename intrinsic_type<T>::type type; };
template <typename T> struct intrinsic_type<T*>                   { typedef typename intrinsic_type<T>::type type; };
template <typename T> struct intrinsic_type<T&>                   { typedef typename intrinsic_type<T>::type type; };
template <typename T> struct intrinsic_type<T&&>                  { typedef typename intrinsic_type<T>::type type; };
template <typename T, size_t N> struct intrinsic_type<const T[N]> { typedef typename intrinsic_type<T>::type type; };
template <typename T, size_t N> struct intrinsic_type<T[N]>       { typedef typename intrinsic_type<T>::type type; };
template <typename T> using intrinsic_t = typename intrinsic_type<T>::type;

/// Helper type to replace 'void' in some expressions
struct void_type { };

/// Helper template which holds a list of types
template <typename...> struct type_list { };

/// Compile-time integer sum
#ifdef __cpp_fold_expressions
template <typename... Ts> constexpr size_t constexpr_sum(Ts... ns) { return (0 + ... + size_t{ns}); }
#else
constexpr size_t constexpr_sum() { return 0; }
template <typename T, typename... Ts>
constexpr size_t constexpr_sum(T n, Ts... ns) { return size_t{n} + constexpr_sum(ns...); }
#endif

NAMESPACE_BEGIN(constexpr_impl)
/// Implementation details for constexpr functions
constexpr int first(int i) { return i; }
template <typename T, typename... Ts>
constexpr int first(int i, T v, Ts... vs) { return v ? i : first(i + 1, vs...); }

constexpr int last(int /*i*/, int result) { return result; }
template <typename T, typename... Ts>
constexpr int last(int i, int result, T v, Ts... vs) { return last(i + 1, v ? i : result, vs...); }
NAMESPACE_END(constexpr_impl)

/// Return the index of the first type in Ts which satisfies Predicate<T>.  Returns sizeof...(Ts) if
/// none match.
template <template<typename> class Predicate, typename... Ts>
constexpr int constexpr_first() { return constexpr_impl::first(0, Predicate<Ts>::value...); }

/// Return the index of the last type in Ts which satisfies Predicate<T>, or -1 if none match.
template <template<typename> class Predicate, typename... Ts>
constexpr int constexpr_last() { return constexpr_impl::last(0, -1, Predicate<Ts>::value...); }

// Extracts the first type from the template parameter pack matching the predicate, or Default if none match.
template <template<class> class Predicate, class Default, class... Ts> struct first_of;
template <template<class> class Predicate, class Default> struct first_of<Predicate, Default> {
    using type = Default;
};
template <template<class> class Predicate, class Default, class T, class... Ts>
struct first_of<Predicate, Default, T, Ts...> {
    using type = typename std::conditional<
        Predicate<T>::value,
        T,
        typename first_of<Predicate, Default, Ts...>::type
    >::type;
};
template <template<class> class Predicate, class Default, class... T> using first_of_t = typename first_of<Predicate, Default, T...>::type;

/// Defer the evaluation of type T until types Us are instantiated
template <typename T, typename... /*Us*/> struct deferred_type { using type = T; };
template <typename T, typename... Us> using deferred_t = typename deferred_type<T, Us...>::type;

template <template<typename...> class Base>
struct is_template_base_of_impl {
    template <typename... Us> static std::true_type check(Base<Us...> *);
    static std::false_type check(...);
};

/// Check if a template is the base of a type. For example:
/// `is_template_base_of<Base, T>` is true if `struct T : Base<U> {}` where U can be anything
template <template<typename...> class Base, typename T>
#if !defined(_MSC_VER)
using is_template_base_of = decltype(is_template_base_of_impl<Base>::check((remove_cv_t<T>*)nullptr));
#else // MSVC2015 has trouble with decltype in template aliases
struct is_template_base_of : decltype(is_template_base_of_impl<Base>::check((remove_cv_t<T>*)nullptr)) { };
#endif

/// Check if T is std::shared_ptr<U> where U can be anything
template <typename T> struct is_shared_ptr : std::false_type { };
template <typename U> struct is_shared_ptr<std::shared_ptr<U>> : std::true_type { };

/// Ignore that a variable is unused in compiler warnings
inline void ignore_unused(const int *) { }

NAMESPACE_END(detail)

/// Returns a named pointer that is shared among all extension modules (using the same
/// pybind11 version) running in the current interpreter. Names starting with underscores
/// are reserved for internal usage. Returns `nullptr` if no matching entry was found.
inline PYBIND11_NOINLINE void* get_shared_data(const std::string& name) {
    auto& internals = detail::get_internals();
    auto it = internals.shared_data.find(name);
    return it != internals.shared_data.end() ? it->second : nullptr;
}

/// Set the shared data that can be later recovered by `get_shared_data()`.
inline PYBIND11_NOINLINE void *set_shared_data(const std::string& name, void *data) {
    detail::get_internals().shared_data[name] = data;
    return data;
}

/// Returns a typed reference to a shared data entry (by using `get_shared_data()`) if
/// such entry exists. Otherwise, a new object of default-constructible type `T` is
/// added to the shared data under the given name and a reference to it is returned.
template<typename T> T& get_or_create_shared_data(const std::string& name) {
    auto& internals = detail::get_internals();
    auto it = internals.shared_data.find(name);
    T* ptr = (T*) (it != internals.shared_data.end() ? it->second : nullptr);
    if (!ptr) {
        ptr = new T();
        internals.shared_data[name] = ptr;
    }
    return *ptr;
}

/// Fetch and hold an error which was already set in Python
class error_already_set : public std::runtime_error {
public:
    error_already_set() : std::runtime_error(detail::error_string()) {
        PyErr_Fetch(&type, &value, &trace);
    }

    error_already_set(const error_already_set &) = delete;

    error_already_set(error_already_set &&e)
        : std::runtime_error(e.what()), type(e.type), value(e.value),
          trace(e.trace) { e.type = e.value = e.trace = nullptr; }

    inline ~error_already_set(); // implementation in pybind11.h

    error_already_set& operator=(const error_already_set &) = delete;

    /// Give the error back to Python
    void restore() { PyErr_Restore(type, value, trace); type = value = trace = nullptr; }

    /// Clear the held Python error state (the C++ `what()` message remains intact)
    void clear() { restore(); PyErr_Clear(); }

private:
    PyObject *type, *value, *trace;
};

/// C++ bindings of builtin Python exceptions
class builtin_exception : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
    /// Set the error using the Python C API
    virtual void set_error() const = 0;
};

#define PYBIND11_RUNTIME_EXCEPTION(name, type) \
    class name : public builtin_exception { public: \
        using builtin_exception::builtin_exception; \
        name() : name("") { } \
        void set_error() const override { PyErr_SetString(type, what()); } \
    };

PYBIND11_RUNTIME_EXCEPTION(stop_iteration, PyExc_StopIteration)
PYBIND11_RUNTIME_EXCEPTION(index_error, PyExc_IndexError)
PYBIND11_RUNTIME_EXCEPTION(key_error, PyExc_KeyError)
PYBIND11_RUNTIME_EXCEPTION(value_error, PyExc_ValueError)
PYBIND11_RUNTIME_EXCEPTION(type_error, PyExc_TypeError)
PYBIND11_RUNTIME_EXCEPTION(cast_error, PyExc_RuntimeError) /// Thrown when pybind11::cast or handle::call fail due to a type casting error
PYBIND11_RUNTIME_EXCEPTION(reference_cast_error, PyExc_RuntimeError) /// Used internally

[[noreturn]] PYBIND11_NOINLINE inline void pybind11_fail(const char *reason) { throw std::runtime_error(reason); }
[[noreturn]] PYBIND11_NOINLINE inline void pybind11_fail(const std::string &reason) { throw std::runtime_error(reason); }

template <typename T, typename SFINAE = void> struct format_descriptor { };

NAMESPACE_BEGIN(detail)
// Returns the index of the given type in the type char array below, and in the list in numpy.h
// The order here is: bool; 8 ints ((signed,unsigned)x(8,16,32,64)bits); float,double,long double;
// complex float,double,long double.  Note that the long double types only participate when long
// double is actually longer than double (it isn't under MSVC).
// NB: not only the string below but also complex.h and numpy.h rely on this order.
template <typename T, typename SFINAE = void> struct is_fmt_numeric { static constexpr bool value = false; };
template <typename T> struct is_fmt_numeric<T, enable_if_t<std::is_arithmetic<T>::value>> {
    static constexpr bool value = true;
    static constexpr int index = std::is_same<T, bool>::value ? 0 : 1 + (
        std::is_integral<T>::value ? detail::log2(sizeof(T))*2 + std::is_unsigned<T>::value : 8 + (
        std::is_same<T, double>::value ? 1 : std::is_same<T, long double>::value ? 2 : 0));
};
NAMESPACE_END(detail)

template <typename T> struct format_descriptor<T, detail::enable_if_t<detail::is_fmt_numeric<T>::value>> {
    static constexpr const char c = "?bBhHiIqQfdgFDG"[detail::is_fmt_numeric<T>::index];
    static constexpr const char value[2] = { c, '\0' };
    static std::string format() { return std::string(1, c); }
};

template <typename T> constexpr const char format_descriptor<
    T, detail::enable_if_t<detail::is_fmt_numeric<T>::value>>::value[2];

NAMESPACE_BEGIN(detail)

template <typename T, typename SFINAE = void> struct compare_buffer_info {
    static bool compare(const buffer_info& b) {
        return b.format == format_descriptor<T>::format() && b.itemsize == sizeof(T);
    }
};

template <typename T> struct compare_buffer_info<T, detail::enable_if_t<std::is_integral<T>::value>> {
    static bool compare(const buffer_info& b) {
        return b.itemsize == sizeof(T) && (b.format == format_descriptor<T>::value ||
            ((sizeof(T) == sizeof(long)) && b.format == (std::is_unsigned<T>::value ? "L" : "l")) ||
            ((sizeof(T) == sizeof(size_t)) && b.format == (std::is_unsigned<T>::value ? "N" : "n")));
    }
};

NAMESPACE_END(detail)

/// RAII wrapper that temporarily clears any Python error state
struct error_scope {
    PyObject *type, *value, *trace;
    error_scope() { PyErr_Fetch(&type, &value, &trace); }
    ~error_scope() { PyErr_Restore(type, value, trace); }
};

/// Dummy destructor wrapper that can be used to expose classes with a private destructor
struct nodelete { template <typename T> void operator()(T*) { } };

// overload_cast requires variable templates: C++14 or MSVC
#if defined(PYBIND11_CPP14) || defined(_MSC_VER)
#define PYBIND11_OVERLOAD_CAST 1

NAMESPACE_BEGIN(detail)
template <typename... Args>
struct overload_cast_impl {
    template <typename Return>
    constexpr auto operator()(Return (*pf)(Args...)) const noexcept
                              -> decltype(pf) { return pf; }

    template <typename Return, typename Class>
    constexpr auto operator()(Return (Class::*pmf)(Args...), std::false_type = {}) const noexcept
                              -> decltype(pmf) { return pmf; }

    template <typename Return, typename Class>
    constexpr auto operator()(Return (Class::*pmf)(Args...) const, std::true_type) const noexcept
                              -> decltype(pmf) { return pmf; }
};
NAMESPACE_END(detail)

/// Syntax sugar for resolving overloaded function pointers:
///  - regular: static_cast<Return (Class::*)(Arg0, Arg1, Arg2)>(&Class::func)
///  - sweet:   overload_cast<Arg0, Arg1, Arg2>(&Class::func)
template <typename... Args>
static constexpr detail::overload_cast_impl<Args...> overload_cast = {};
// MSVC 2015 only accepts this particular initialization syntax for this variable template.

/// Const member function selector for overload_cast
///  - regular: static_cast<Return (Class::*)(Arg) const>(&Class::func)
///  - sweet:   overload_cast<Arg>(&Class::func, const_)
static constexpr auto const_ = std::true_type{};

#endif // overload_cast

NAMESPACE_END(pybind11)
