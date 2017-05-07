/*
    pybind11/pybind11.h: Main header file of the C++11 python
    binding generator library

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4100) // warning C4100: Unreferenced formal parameter
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#  pragma warning(disable: 4512) // warning C4512: Assignment operator was implicitly defined as deleted
#  pragma warning(disable: 4800) // warning C4800: 'int': forcing value to bool 'true' or 'false' (performance warning)
#  pragma warning(disable: 4996) // warning C4996: The POSIX name for this item is deprecated. Instead, use the ISO C and C++ conformant name
#  pragma warning(disable: 4702) // warning C4702: unreachable code
#  pragma warning(disable: 4522) // warning C4522: multiple assignment operators specified
#elif defined(__INTEL_COMPILER)
#  pragma warning(push)
#  pragma warning(disable: 186)   // pointless comparison of unsigned integer with zero
#  pragma warning(disable: 1334)  // the "template" keyword used for syntactic disambiguation may only be used within a template
#  pragma warning(disable: 2196)  // warning #2196: routine is both "inline" and "noinline"
#elif defined(__GNUG__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#  pragma GCC diagnostic ignored "-Wattributes"
#endif

#include "attr.h"
#include "options.h"

NAMESPACE_BEGIN(pybind11)

/// Wraps an arbitrary C++ function/method/lambda function/.. into a callable Python object
class cpp_function : public function {
public:
    cpp_function() { }

    /// Construct a cpp_function from a vanilla function pointer
    template <typename Return, typename... Args, typename... Extra /*,*/ PYBIND11_NOEXCEPT_TPL_ARG>
    cpp_function(Return (*f)(Args...) PYBIND11_NOEXCEPT_SPECIFIER, const Extra&... extra) {
        initialize(f, f, extra...);
    }

    /// Construct a cpp_function from a lambda function (possibly with internal state)
    template <typename Func, typename... Extra> cpp_function(Func &&f, const Extra&... extra) {
        initialize(std::forward<Func>(f),
                   (typename detail::remove_class<decltype(
                       &std::remove_reference<Func>::type::operator())>::type *) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (non-const)
    template <typename Return, typename Class, typename... Arg, typename... Extra /*,*/ PYBIND11_NOEXCEPT_TPL_ARG>
    cpp_function(Return (Class::*f)(Arg...) PYBIND11_NOEXCEPT_SPECIFIER, const Extra&... extra) {
        initialize([f](Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                   (Return (*) (Class *, Arg...) PYBIND11_NOEXCEPT_SPECIFIER) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (const)
    template <typename Return, typename Class, typename... Arg, typename... Extra /*,*/ PYBIND11_NOEXCEPT_TPL_ARG>
    cpp_function(Return (Class::*f)(Arg...) const PYBIND11_NOEXCEPT_SPECIFIER, const Extra&... extra) {
        initialize([f](const Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                   (Return (*)(const Class *, Arg ...) PYBIND11_NOEXCEPT_SPECIFIER) nullptr, extra...);
    }

    /// Return the function name
    object name() const { return attr("__name__"); }

protected:
    /// Space optimization: don't inline this frequently instantiated fragment
    PYBIND11_NOINLINE detail::function_record *make_function_record() {
        return new detail::function_record();
    }

    /// Special internal constructor for functors, lambda functions, etc.
    template <typename Func, typename Return, typename... Args, typename... Extra /*,*/ PYBIND11_NOEXCEPT_TPL_ARG>
    void initialize(Func &&f, Return (*)(Args...) PYBIND11_NOEXCEPT_SPECIFIER, const Extra&... extra) {
        static_assert(detail::expected_num_args<Extra...>(sizeof...(Args)),
                      "The number of named arguments does not match the function signature");

        struct capture { typename std::remove_reference<Func>::type f; };

        /* Store the function including any extra state it might have (e.g. a lambda capture object) */
        auto rec = make_function_record();

        /* Store the capture object directly in the function record if there is enough space */
        if (sizeof(capture) <= sizeof(rec->data)) {
            /* Without these pragmas, GCC warns that there might not be
               enough space to use the placement new operator. However, the
               'if' statement above ensures that this is the case. */
#if defined(__GNUG__) && !defined(__clang__) && __GNUC__ >= 6
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wplacement-new"
#endif
            new ((capture *) &rec->data) capture { std::forward<Func>(f) };
#if defined(__GNUG__) && !defined(__clang__) && __GNUC__ >= 6
#  pragma GCC diagnostic pop
#endif
            if (!std::is_trivially_destructible<Func>::value)
                rec->free_data = [](detail::function_record *r) { ((capture *) &r->data)->~capture(); };
        } else {
            rec->data[0] = new capture { std::forward<Func>(f) };
            rec->free_data = [](detail::function_record *r) { delete ((capture *) r->data[0]); };
        }

        /* Type casters for the function arguments and return value */
        using cast_in = detail::argument_loader<Args...>;
        using cast_out = detail::make_caster<
            detail::conditional_t<std::is_void<Return>::value, detail::void_type, Return>
        >;

        /* Dispatch code which converts function arguments and performs the actual function call */
        rec->impl = [](detail::function_record *rec, handle args, handle kwargs, handle parent) -> handle {
            cast_in args_converter;

            /* Try to cast the function arguments into the C++ domain */
            if (!args_converter.load_args(args, kwargs))
                return PYBIND11_TRY_NEXT_OVERLOAD;

            /* Invoke call policy pre-call hook */
            detail::process_attributes<Extra...>::precall(args);

            /* Get a pointer to the capture object */
            capture *cap = (capture *) (sizeof(capture) <= sizeof(rec->data)
                                        ? &rec->data : rec->data[0]);

            /* Override policy for rvalues -- always move */
            constexpr auto is_rvalue = !std::is_pointer<Return>::value
                                       && !std::is_lvalue_reference<Return>::value;
            const auto policy = is_rvalue ? return_value_policy::move : rec->policy;

            /* Perform the function call */
            handle result = cast_out::cast(args_converter.template call<Return>(cap->f),
                                           policy, parent);

            /* Invoke call policy post-call hook */
            detail::process_attributes<Extra...>::postcall(args, result);

            return result;
        };

        /* Process any user-provided function attributes */
        detail::process_attributes<Extra...>::init(extra..., rec);

        /* Generate a readable signature describing the function's arguments and return value types */
        using detail::descr; using detail::_;
        PYBIND11_DESCR signature = _("(") + cast_in::arg_names() + _(") -> ") + cast_out::name();

        /* Register the function with Python from generic (non-templated) code */
        initialize_generic(rec, signature.text(), signature.types(), sizeof...(Args));

        if (cast_in::has_args) rec->has_args = true;
        if (cast_in::has_kwargs) rec->has_kwargs = true;

        /* Stash some additional information used by an important optimization in 'functional.h' */
        using FunctionType = Return (*)(Args...) PYBIND11_NOEXCEPT_SPECIFIER;
        constexpr bool is_function_ptr =
            std::is_convertible<Func, FunctionType>::value &&
            sizeof(capture) == sizeof(void *);
        if (is_function_ptr) {
            rec->is_stateless = true;
            rec->data[1] = (void *) &typeid(FunctionType);
        }
    }

    /// Register a function call with Python (generic non-templated code goes here)
    void initialize_generic(detail::function_record *rec, const char *text,
                            const std::type_info *const *types, size_t args) {

        /* Create copies of all referenced C-style strings */
        rec->name = strdup(rec->name ? rec->name : "");
        if (rec->doc) rec->doc = strdup(rec->doc);
        for (auto &a: rec->args) {
            if (a.name)
                a.name = strdup(a.name);
            if (a.descr)
                a.descr = strdup(a.descr);
            else if (a.value)
                a.descr = strdup(a.value.attr("__repr__")().cast<std::string>().c_str());
        }

        /* Generate a proper function signature */
        std::string signature;
        size_t type_depth = 0, char_index = 0, type_index = 0, arg_index = 0;
        while (true) {
            char c = text[char_index++];
            if (c == '\0')
                break;

            if (c == '{') {
                // Write arg name for everything except *args, **kwargs and return type.
                if (type_depth == 0 && text[char_index] != '*' && arg_index < args) {
                    if (!rec->args.empty()) {
                        signature += rec->args[arg_index].name;
                    } else if (arg_index == 0 && rec->is_method) {
                        signature += "self";
                    } else {
                        signature += "arg" + std::to_string(arg_index - (rec->is_method ? 1 : 0));
                    }
                    signature += ": ";
                }
                ++type_depth;
            } else if (c == '}') {
                --type_depth;
                if (type_depth == 0) {
                    if (arg_index < rec->args.size() && rec->args[arg_index].descr) {
                        signature += "=";
                        signature += rec->args[arg_index].descr;
                    }
                    arg_index++;
                }
            } else if (c == '%') {
                const std::type_info *t = types[type_index++];
                if (!t)
                    pybind11_fail("Internal error while parsing type signature (1)");
                if (auto tinfo = detail::get_type_info(*t)) {
#if defined(PYPY_VERSION)
                    signature += handle((PyObject *) tinfo->type)
                                     .attr("__module__")
                                     .cast<std::string>() + ".";
#endif
                    signature += tinfo->type->tp_name;
                } else {
                    std::string tname(t->name());
                    detail::clean_type_id(tname);
                    signature += tname;
                }
            } else {
                signature += c;
            }
        }
        if (type_depth != 0 || types[type_index] != nullptr)
            pybind11_fail("Internal error while parsing type signature (2)");

        #if !defined(PYBIND11_CPP14)
            delete[] types;
            delete[] text;
        #endif

#if PY_MAJOR_VERSION < 3
        if (strcmp(rec->name, "__next__") == 0) {
            std::free(rec->name);
            rec->name = strdup("next");
        } else if (strcmp(rec->name, "__bool__") == 0) {
            std::free(rec->name);
            rec->name = strdup("__nonzero__");
        }
#endif
        rec->signature = strdup(signature.c_str());
        rec->args.shrink_to_fit();
        rec->is_constructor = !strcmp(rec->name, "__init__") || !strcmp(rec->name, "__setstate__");
        rec->nargs = (uint16_t) args;

#if PY_MAJOR_VERSION < 3
        if (rec->sibling && PyMethod_Check(rec->sibling.ptr()))
            rec->sibling = PyMethod_GET_FUNCTION(rec->sibling.ptr());
#endif

        detail::function_record *chain = nullptr, *chain_start = rec;
        if (rec->sibling) {
            if (PyCFunction_Check(rec->sibling.ptr())) {
                auto rec_capsule = reinterpret_borrow<capsule>(PyCFunction_GET_SELF(rec->sibling.ptr()));
                chain = (detail::function_record *) rec_capsule;
                /* Never append a method to an overload chain of a parent class;
                   instead, hide the parent's overloads in this case */
                if (chain->scope != rec->scope)
                    chain = nullptr;
            }
            // Don't trigger for things like the default __init__, which are wrapper_descriptors that we are intentionally replacing
            else if (!rec->sibling.is_none() && rec->name[0] != '_')
                pybind11_fail("Cannot overload existing non-function object \"" + std::string(rec->name) +
                        "\" with a function of the same name");
        }

        if (!chain) {
            /* No existing overload was found, create a new function object */
            rec->def = new PyMethodDef();
            memset(rec->def, 0, sizeof(PyMethodDef));
            rec->def->ml_name = rec->name;
            rec->def->ml_meth = reinterpret_cast<PyCFunction>(*dispatcher);
            rec->def->ml_flags = METH_VARARGS | METH_KEYWORDS;

            capsule rec_capsule(rec, [](PyObject *o) {
                destruct((detail::function_record *) PyCapsule_GetPointer(o, nullptr));
            });

            object scope_module;
            if (rec->scope) {
                if (hasattr(rec->scope, "__module__")) {
                    scope_module = rec->scope.attr("__module__");
                } else if (hasattr(rec->scope, "__name__")) {
                    scope_module = rec->scope.attr("__name__");
                }
            }

            m_ptr = PyCFunction_NewEx(rec->def, rec_capsule.ptr(), scope_module.ptr());
            if (!m_ptr)
                pybind11_fail("cpp_function::cpp_function(): Could not allocate function object");
        } else {
            /* Append at the end of the overload chain */
            m_ptr = rec->sibling.ptr();
            inc_ref();
            chain_start = chain;
            while (chain->next)
                chain = chain->next;
            chain->next = rec;
        }

        std::string signatures;
        int index = 0;
        /* Create a nice pydoc rec including all signatures and
           docstrings of the functions in the overload chain */
        if (chain && options::show_function_signatures()) {
            // First a generic signature
            signatures += rec->name;
            signatures += "(*args, **kwargs)\n";
            signatures += "Overloaded function.\n\n";
        }
        // Then specific overload signatures
        for (auto it = chain_start; it != nullptr; it = it->next) {
            if (options::show_function_signatures()) {
                if (chain)
                    signatures += std::to_string(++index) + ". ";
                signatures += rec->name;
                signatures += it->signature;
                signatures += "\n";
            }
            if (it->doc && strlen(it->doc) > 0 && options::show_user_defined_docstrings()) {
                if (options::show_function_signatures()) signatures += "\n";
                signatures += it->doc;
                if (options::show_function_signatures()) signatures += "\n";
            }
            if (it->next)
                signatures += "\n";
        }

        /* Install docstring */
        PyCFunctionObject *func = (PyCFunctionObject *) m_ptr;
        if (func->m_ml->ml_doc)
            std::free((char *) func->m_ml->ml_doc);
        func->m_ml->ml_doc = strdup(signatures.c_str());

        if (rec->is_method) {
            m_ptr = PYBIND11_INSTANCE_METHOD_NEW(m_ptr, rec->scope.ptr());
            if (!m_ptr)
                pybind11_fail("cpp_function::cpp_function(): Could not allocate instance method object");
            Py_DECREF(func);
        }
    }

    /// When a cpp_function is GCed, release any memory allocated by pybind11
    static void destruct(detail::function_record *rec) {
        while (rec) {
            detail::function_record *next = rec->next;
            if (rec->free_data)
                rec->free_data(rec);
            std::free((char *) rec->name);
            std::free((char *) rec->doc);
            std::free((char *) rec->signature);
            for (auto &arg: rec->args) {
                std::free((char *) arg.name);
                std::free((char *) arg.descr);
                arg.value.dec_ref();
            }
            if (rec->def) {
                std::free((char *) rec->def->ml_doc);
                delete rec->def;
            }
            delete rec;
            rec = next;
        }
    }

    /// Main dispatch logic for calls to functions bound using pybind11
    static PyObject *dispatcher(PyObject *self, PyObject *args, PyObject *kwargs) {
        /* Iterator over the list of potentially admissible overloads */
        detail::function_record *overloads = (detail::function_record *) PyCapsule_GetPointer(self, nullptr),
                                *it = overloads;

        /* Need to know how many arguments + keyword arguments there are to pick the right overload */
        size_t nargs = (size_t) PyTuple_GET_SIZE(args),
               nkwargs = kwargs ? (size_t) PyDict_Size(kwargs) : 0;

        handle parent = nargs > 0 ? PyTuple_GET_ITEM(args, 0) : nullptr,
               result = PYBIND11_TRY_NEXT_OVERLOAD;
        try {
            for (; it != nullptr; it = it->next) {
                auto args_ = reinterpret_borrow<tuple>(args);
                size_t kwargs_consumed = 0;

                /* For each overload:
                   1. If the required list of arguments is longer than the
                      actually provided amount, create a copy of the argument
                      list and fill in any available keyword/default arguments.
                   2. Ensure that all keyword arguments were "consumed"
                   3. Call the function call dispatcher (function_record::impl)
                 */
                size_t nargs_ = nargs;
                if (nargs < it->args.size()) {
                    nargs_ = it->args.size();
                    args_ = tuple(nargs_);
                    for (size_t i = 0; i < nargs; ++i) {
                        handle item = PyTuple_GET_ITEM(args, i);
                        PyTuple_SET_ITEM(args_.ptr(), i, item.inc_ref().ptr());
                    }

                    int arg_ctr = 0;
                    for (auto const &it2 : it->args) {
                        int index = arg_ctr++;
                        if (PyTuple_GET_ITEM(args_.ptr(), index))
                            continue;

                        handle value;
                        if (kwargs)
                            value = PyDict_GetItemString(kwargs, it2.name);

                        if (value)
                            kwargs_consumed++;
                        else if (it2.value)
                            value = it2.value;

                        if (value) {
                            PyTuple_SET_ITEM(args_.ptr(), index, value.inc_ref().ptr());
                        } else {
                            kwargs_consumed = (size_t) -1; /* definite failure */
                            break;
                        }
                    }
                }

                try {
                    if ((kwargs_consumed == nkwargs || it->has_kwargs) &&
                        (nargs_ == it->nargs || it->has_args))
                        result = it->impl(it, args_, kwargs, parent);
                } catch (reference_cast_error &) {
                    result = PYBIND11_TRY_NEXT_OVERLOAD;
                }

                if (result.ptr() != PYBIND11_TRY_NEXT_OVERLOAD)
                    break;
            }
        } catch (error_already_set &e) {
            e.restore();
            return nullptr;
        } catch (...) {
            /* When an exception is caught, give each registered exception
               translator a chance to translate it to a Python exception
               in reverse order of registration.

               A translator may choose to do one of the following:

                - catch the exception and call PyErr_SetString or PyErr_SetObject
                  to set a standard (or custom) Python exception, or
                - do nothing and let the exception fall through to the next translator, or
                - delegate translation to the next translator by throwing a new type of exception. */

            auto last_exception = std::current_exception();
            auto &registered_exception_translators = pybind11::detail::get_internals().registered_exception_translators;
            for (auto& translator : registered_exception_translators) {
                try {
                    translator(last_exception);
                } catch (...) {
                    last_exception = std::current_exception();
                    continue;
                }
                return nullptr;
            }
            PyErr_SetString(PyExc_SystemError, "Exception escaped from default exception translator!");
            return nullptr;
        }

        if (result.ptr() == PYBIND11_TRY_NEXT_OVERLOAD) {
            if (overloads->is_operator)
                return handle(Py_NotImplemented).inc_ref().ptr();

            std::string msg = std::string(overloads->name) + "(): incompatible " +
                std::string(overloads->is_constructor ? "constructor" : "function") +
                " arguments. The following argument types are supported:\n";

            int ctr = 0;
            for (detail::function_record *it2 = overloads; it2 != nullptr; it2 = it2->next) {
                msg += "    "+ std::to_string(++ctr) + ". ";

                bool wrote_sig = false;
                if (overloads->is_constructor) {
                    // For a constructor, rewrite `(self: Object, arg0, ...) -> NoneType` as `Object(arg0, ...)`
                    std::string sig = it2->signature;
                    size_t start = sig.find('(') + 7; // skip "(self: "
                    if (start < sig.size()) {
                        // End at the , for the next argument
                        size_t end = sig.find(", "), next = end + 2;
                        size_t ret = sig.rfind(" -> ");
                        // Or the ), if there is no comma:
                        if (end >= sig.size()) next = end = sig.find(')');
                        if (start < end && next < sig.size()) {
                            msg.append(sig, start, end - start);
                            msg += '(';
                            msg.append(sig, next, ret - next);
                            wrote_sig = true;
                        }
                    }
                }
                if (!wrote_sig) msg += it2->signature;

                msg += "\n";
            }
            msg += "\nInvoked with: ";
            auto args_ = reinterpret_borrow<tuple>(args);
            for (size_t ti = overloads->is_constructor ? 1 : 0; ti < args_.size(); ++ti) {
                msg += pybind11::repr(args_[ti]);
                if ((ti + 1) != args_.size() )
                    msg += ", ";
            }
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return nullptr;
        } else if (!result) {
            std::string msg = "Unable to convert function return value to a "
                              "Python type! The signature was\n\t";
            msg += it->signature;
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return nullptr;
        } else {
            if (overloads->is_constructor) {
                /* When a constructor ran successfully, the corresponding
                   holder type (e.g. std::unique_ptr) must still be initialized. */
                PyObject *inst = PyTuple_GET_ITEM(args, 0);
                auto tinfo = detail::get_type_info(Py_TYPE(inst));
                tinfo->init_holder(inst, nullptr);
            }
            return result.ptr();
        }
    }
};

/// Wrapper for Python extension modules
class module : public object {
public:
    PYBIND11_OBJECT_DEFAULT(module, object, PyModule_Check)

    explicit module(const char *name, const char *doc = nullptr) {
        if (!options::show_user_defined_docstrings()) doc = nullptr;
#if PY_MAJOR_VERSION >= 3
        PyModuleDef *def = new PyModuleDef();
        memset(def, 0, sizeof(PyModuleDef));
        def->m_name = name;
        def->m_doc = doc;
        def->m_size = -1;
        Py_INCREF(def);
        m_ptr = PyModule_Create(def);
#else
        m_ptr = Py_InitModule3(name, nullptr, doc);
#endif
        if (m_ptr == nullptr)
            pybind11_fail("Internal error in module::module()");
        inc_ref();
    }

    template <typename Func, typename... Extra>
    module &def(const char *name_, Func &&f, const Extra& ... extra) {
        cpp_function func(std::forward<Func>(f), name(name_), scope(*this),
                          sibling(getattr(*this, name_, none())), extra...);
        // NB: allow overwriting here because cpp_function sets up a chain with the intention of
        // overwriting (and has already checked internally that it isn't overwriting non-functions).
        add_object(name_, func, true /* overwrite */);
        return *this;
    }

    module def_submodule(const char *name, const char *doc = nullptr) {
        std::string full_name = std::string(PyModule_GetName(m_ptr))
            + std::string(".") + std::string(name);
        auto result = reinterpret_borrow<module>(PyImport_AddModule(full_name.c_str()));
        if (doc && options::show_user_defined_docstrings())
            result.attr("__doc__") = pybind11::str(doc);
        attr(name) = result;
        return result;
    }

    static module import(const char *name) {
        PyObject *obj = PyImport_ImportModule(name);
        if (!obj)
            throw error_already_set();
        return reinterpret_steal<module>(obj);
    }

    // Adds an object to the module using the given name.  Throws if an object with the given name
    // already exists.
    //
    // overwrite should almost always be false: attempting to overwrite objects that pybind11 has
    // established will, in most cases, break things.
    PYBIND11_NOINLINE void add_object(const char *name, object &obj, bool overwrite = false) {
        if (!overwrite && hasattr(*this, name))
            pybind11_fail("Error during initialization: multiple incompatible definitions with name \"" +
                    std::string(name) + "\"");

        obj.inc_ref(); // PyModule_AddObject() steals a reference
        PyModule_AddObject(ptr(), name, obj.ptr());
    }
};

NAMESPACE_BEGIN(detail)
extern "C" inline PyObject *get_dict(PyObject *op, void *) {
    PyObject *&dict = *_PyObject_GetDictPtr(op);
    if (!dict)
        dict = PyDict_New();
    Py_XINCREF(dict);
    return dict;
}

extern "C" inline int set_dict(PyObject *op, PyObject *new_dict, void *) {
    if (!PyDict_Check(new_dict)) {
        PyErr_Format(PyExc_TypeError, "__dict__ must be set to a dictionary, not a '%.200s'",
                     Py_TYPE(new_dict)->tp_name);
        return -1;
    }
    PyObject *&dict = *_PyObject_GetDictPtr(op);
    Py_INCREF(new_dict);
    Py_CLEAR(dict);
    dict = new_dict;
    return 0;
}

static PyGetSetDef generic_getset[] = {
    {const_cast<char*>("__dict__"), get_dict, set_dict, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}
};

/// Generic support for creating new Python heap types
class generic_type : public object {
    template <typename...> friend class class_;
public:
    PYBIND11_OBJECT_DEFAULT(generic_type, object, PyType_Check)
protected:
    void initialize(type_record *rec) {
        auto &internals = get_internals();
        auto tindex = std::type_index(*(rec->type));

        if (get_type_info(*(rec->type)))
            pybind11_fail("generic_type: type \"" + std::string(rec->name) +
                          "\" is already registered!");

        auto name = reinterpret_steal<object>(PYBIND11_FROM_STRING(rec->name));
        object scope_module;
        if (rec->scope) {
            if (hasattr(rec->scope, rec->name))
                pybind11_fail("generic_type: cannot initialize type \"" + std::string(rec->name) +
                        "\": an object with that name is already defined");

            if (hasattr(rec->scope, "__module__")) {
                scope_module = rec->scope.attr("__module__");
            } else if (hasattr(rec->scope, "__name__")) {
                scope_module = rec->scope.attr("__name__");
            }
        }

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 3
        /* Qualified names for Python >= 3.3 */
        object scope_qualname;
        if (rec->scope && hasattr(rec->scope, "__qualname__"))
            scope_qualname = rec->scope.attr("__qualname__");
        object ht_qualname, ht_qualname_meta;
        if (scope_qualname)
            ht_qualname = reinterpret_steal<object>(PyUnicode_FromFormat(
                "%U.%U", scope_qualname.ptr(), name.ptr()));
        else
            ht_qualname = name;
        if (rec->metaclass)
            ht_qualname_meta = reinterpret_steal<object>(
                PyUnicode_FromFormat("%U__Meta", ht_qualname.ptr()));
#endif

#if !defined(PYPY_VERSION)
        std::string full_name = (scope_module ? ((std::string) pybind11::str(scope_module) + "." + rec->name)
                                              : std::string(rec->name));
#else
        std::string full_name = std::string(rec->name);
#endif

        /* Create a custom metaclass if requested (used for static properties) */
        object metaclass;
        if (rec->metaclass) {
            std::string meta_name_ = full_name + "__Meta";
            object meta_name = reinterpret_steal<object>(PYBIND11_FROM_STRING(meta_name_.c_str()));
            metaclass = reinterpret_steal<object>(PyType_Type.tp_alloc(&PyType_Type, 0));
            if (!metaclass || !name)
                pybind11_fail("generic_type::generic_type(): unable to create metaclass!");

            /* Danger zone: from now (and until PyType_Ready), make sure to
               issue no Python C API calls which could potentially invoke the
               garbage collector (the GC will call type_traverse(), which will in
               turn find the newly constructed type in an invalid state) */

            auto type = (PyHeapTypeObject*) metaclass.ptr();
            type->ht_name = meta_name.release().ptr();

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 3
            /* Qualified names for Python >= 3.3 */
            type->ht_qualname = ht_qualname_meta.release().ptr();
#endif
            type->ht_type.tp_name = strdup(meta_name_.c_str());
            type->ht_type.tp_base = &PyType_Type;
            type->ht_type.tp_flags |= (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE) &
                                      ~Py_TPFLAGS_HAVE_GC;

            if (PyType_Ready(&type->ht_type) < 0)
                pybind11_fail("generic_type::generic_type(): failure in PyType_Ready() for metaclass!");
        }

        size_t num_bases = rec->bases.size();
        auto bases = tuple(rec->bases);

        char *tp_doc = nullptr;
        if (rec->doc && options::show_user_defined_docstrings()) {
            /* Allocate memory for docstring (using PyObject_MALLOC, since
               Python will free this later on) */
            size_t size = strlen(rec->doc) + 1;
            tp_doc = (char *) PyObject_MALLOC(size);
            memcpy((void *) tp_doc, rec->doc, size);
        }

        /* Danger zone: from now (and until PyType_Ready), make sure to
           issue no Python C API calls which could potentially invoke the
           garbage collector (the GC will call type_traverse(), which will in
           turn find the newly constructed type in an invalid state) */

        auto type_holder = reinterpret_steal<object>(PyType_Type.tp_alloc(&PyType_Type, 0));
        auto type = (PyHeapTypeObject*) type_holder.ptr();

        if (!type_holder || !name)
            pybind11_fail(std::string(rec->name) + ": Unable to create type object!");

        /* Register supplemental type information in C++ dict */
        detail::type_info *tinfo = new detail::type_info();
        tinfo->type = (PyTypeObject *) type;
        tinfo->type_size = rec->type_size;
        tinfo->init_holder = rec->init_holder;
        tinfo->direct_conversions = &internals.direct_conversions[tindex];
        internals.registered_types_cpp[tindex] = tinfo;
        internals.registered_types_py[type] = tinfo;

        /* Basic type attributes */
        type->ht_type.tp_name = strdup(full_name.c_str());
        type->ht_type.tp_basicsize = (ssize_t) rec->instance_size;

        if (num_bases > 0) {
            type->ht_type.tp_base = (PyTypeObject *) ((object) bases[0]).inc_ref().ptr();
            type->ht_type.tp_bases = bases.release().ptr();
            rec->multiple_inheritance |= num_bases > 1;
        }

        type->ht_name = name.release().ptr();

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 3
        type->ht_qualname = ht_qualname.release().ptr();
#endif

        /* Metaclass */
        PYBIND11_OB_TYPE(type->ht_type) = (PyTypeObject *) metaclass.release().ptr();

        /* Supported protocols */
        type->ht_type.tp_as_number = &type->as_number;
        type->ht_type.tp_as_sequence = &type->as_sequence;
        type->ht_type.tp_as_mapping = &type->as_mapping;

        /* Supported elementary operations */
        type->ht_type.tp_init = (initproc) init;
        type->ht_type.tp_new = (newfunc) new_instance;
        type->ht_type.tp_dealloc = rec->dealloc;

        /* Support weak references (needed for the keep_alive feature) */
        type->ht_type.tp_weaklistoffset = offsetof(instance_essentials<void>, weakrefs);

        /* Flags */
        type->ht_type.tp_flags |= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
#if PY_MAJOR_VERSION < 3
        type->ht_type.tp_flags |= Py_TPFLAGS_CHECKTYPES;
#endif
        type->ht_type.tp_flags &= ~Py_TPFLAGS_HAVE_GC;

        /* Support dynamic attributes */
        if (rec->dynamic_attr) {
            #if defined(PYPY_VERSION)
                pybind11_fail(std::string(rec->name) + ": dynamic attributes are "
                                                       "currently not supported in "
                                                       "conunction with PyPy!");
            #endif
            type->ht_type.tp_flags |= Py_TPFLAGS_HAVE_GC;
            type->ht_type.tp_dictoffset = type->ht_type.tp_basicsize; // place the dict at the end
            type->ht_type.tp_basicsize += sizeof(PyObject *); // and allocate enough space for it
            type->ht_type.tp_getset = generic_getset;
            type->ht_type.tp_traverse = traverse;
            type->ht_type.tp_clear = clear;
        }

        if (rec->buffer_protocol) {
            type->ht_type.tp_as_buffer = &type->as_buffer;
#if PY_MAJOR_VERSION < 3
            type->ht_type.tp_flags |= Py_TPFLAGS_HAVE_NEWBUFFER;
#endif
            type->as_buffer.bf_getbuffer = getbuffer;
            type->as_buffer.bf_releasebuffer = releasebuffer;
        }

        type->ht_type.tp_doc = tp_doc;

        m_ptr = type_holder.ptr();

        if (PyType_Ready(&type->ht_type) < 0)
            pybind11_fail(std::string(rec->name) + ": PyType_Ready failed (" +
                          detail::error_string() + ")!");

        if (scope_module) // Needed by pydoc
            attr("__module__") = scope_module;

        /* Register type with the parent scope */
        if (rec->scope)
            rec->scope.attr(handle(type->ht_name)) = *this;

        if (rec->multiple_inheritance)
            mark_parents_nonsimple(&type->ht_type);

        type_holder.release();
    }

    /// Helper function which tags all parents of a type using mult. inheritance
    void mark_parents_nonsimple(PyTypeObject *value) {
        auto t = reinterpret_borrow<tuple>(value->tp_bases);
        for (handle h : t) {
            auto tinfo2 = get_type_info((PyTypeObject *) h.ptr());
            if (tinfo2)
                tinfo2->simple_type = false;
            mark_parents_nonsimple((PyTypeObject *) h.ptr());
        }
    }

    static int init(void *self, PyObject *, PyObject *) {
        PyTypeObject *type = Py_TYPE(self);
        std::string msg;
#if defined(PYPY_VERSION)
        msg += handle((PyObject *) type).attr("__module__").cast<std::string>() + ".";
#endif
        msg += type->tp_name;
        msg += ": No constructor defined!";
        PyErr_SetString(PyExc_TypeError, msg.c_str());
        return -1;
    }

    static PyObject *new_instance(PyTypeObject *type, PyObject *, PyObject *) {
        instance<void> *self = (instance<void> *) PyType_GenericAlloc((PyTypeObject *) type, 0);
        auto tinfo = detail::get_type_info(type);
        self->value = ::operator new(tinfo->type_size);
        self->owned = true;
        self->holder_constructed = false;
        detail::get_internals().registered_instances.emplace(self->value, (PyObject *) self);
        return (PyObject *) self;
    }

    static void dealloc(instance<void> *self) {
        if (self->value) {
            auto instance_type = Py_TYPE(self);
            auto &registered_instances = detail::get_internals().registered_instances;
            auto range = registered_instances.equal_range(self->value);
            bool found = false;
            for (auto it = range.first; it != range.second; ++it) {
                if (instance_type == Py_TYPE(it->second)) {
                    registered_instances.erase(it);
                    found = true;
                    break;
                }
            }
            if (!found)
                pybind11_fail("generic_type::dealloc(): Tried to deallocate unregistered instance!");

            if (self->weakrefs)
                PyObject_ClearWeakRefs((PyObject *) self);

            PyObject **dict_ptr = _PyObject_GetDictPtr((PyObject *) self);
            if (dict_ptr)
                Py_CLEAR(*dict_ptr);
        }
        Py_TYPE(self)->tp_free((PyObject*) self);
    }

    static int traverse(PyObject *op, visitproc visit, void *arg) {
        PyObject *&dict = *_PyObject_GetDictPtr(op);
        Py_VISIT(dict);
        return 0;
    }

    static int clear(PyObject *op) {
        PyObject *&dict = *_PyObject_GetDictPtr(op);
        Py_CLEAR(dict);
        return 0;
    }

    void install_buffer_funcs(
            buffer_info *(*get_buffer)(PyObject *, void *),
            void *get_buffer_data) {
        PyHeapTypeObject *type = (PyHeapTypeObject*) m_ptr;
        auto tinfo = detail::get_type_info(&type->ht_type);

        if (!type->ht_type.tp_as_buffer)
            pybind11_fail(
                "To be able to register buffer protocol support for the type '" +
                std::string(tinfo->type->tp_name) +
                "' the associated class<>(..) invocation must "
                "include the pybind11::buffer_protocol() annotation!");

        tinfo->get_buffer = get_buffer;
        tinfo->get_buffer_data = get_buffer_data;
    }

    static int getbuffer(PyObject *obj, Py_buffer *view, int flags) {
        auto tinfo = detail::get_type_info(Py_TYPE(obj));
        if (view == nullptr || obj == nullptr || !tinfo || !tinfo->get_buffer) {
            if (view)
                view->obj = nullptr;
            PyErr_SetString(PyExc_BufferError, "generic_type::getbuffer(): Internal error");
            return -1;
        }
        memset(view, 0, sizeof(Py_buffer));
        buffer_info *info = tinfo->get_buffer(obj, tinfo->get_buffer_data);
        view->obj = obj;
        view->ndim = 1;
        view->internal = info;
        view->buf = info->ptr;
        view->itemsize = (ssize_t) info->itemsize;
        view->len = view->itemsize;
        for (auto s : info->shape)
            view->len *= s;
        if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT)
            view->format = const_cast<char *>(info->format.c_str());
        if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
            view->ndim = (int) info->ndim;
            view->strides = (ssize_t *) &info->strides[0];
            view->shape = (ssize_t *) &info->shape[0];
        }
        Py_INCREF(view->obj);
        return 0;
    }

    static void releasebuffer(PyObject *, Py_buffer *view) { delete (buffer_info *) view->internal; }

    void def_property_static_impl(const char *name,
                                  handle fget, handle fset,
                                  detail::function_record *rec_fget) {
        pybind11::str doc_obj = pybind11::str(
            (rec_fget->doc && pybind11::options::show_user_defined_docstrings())
                ? rec_fget->doc : "");
        const auto property = reinterpret_steal<object>(
            PyObject_CallFunctionObjArgs((PyObject *) &PyProperty_Type, fget.ptr() ? fget.ptr() : Py_None,
                                         fset.ptr() ? fset.ptr() : Py_None, Py_None, doc_obj.ptr(), nullptr));
        if (rec_fget->is_method && rec_fget->scope) {
            attr(name) = property;
        } else {
            auto mclass = handle((PyObject *) PYBIND11_OB_TYPE(*((PyTypeObject *) m_ptr)));

            if ((PyTypeObject *) mclass.ptr() == &PyType_Type)
                pybind11_fail(
                    "Adding static properties to the type '" +
                    std::string(((PyTypeObject *) m_ptr)->tp_name) +
                    "' requires the type to have a custom metaclass. Please "
                    "ensure that one is created by supplying the pybind11::metaclass() "
                    "annotation to the associated class_<>(..) invocation.");
            mclass.attr(name) = property;
        }
    }
};

NAMESPACE_END(detail)

template <typename type_, typename... options>
class class_ : public detail::generic_type {
    template <typename T> using is_holder = detail::is_holder_type<type_, T>;
    template <typename T> using is_subtype = detail::bool_constant<std::is_base_of<type_, T>::value && !std::is_same<T, type_>::value>;
    template <typename T> using is_base = detail::bool_constant<std::is_base_of<T, type_>::value && !std::is_same<T, type_>::value>;
    // struct instead of using here to help MSVC:
    template <typename T> struct is_valid_class_option :
        detail::any_of<is_holder<T>, is_subtype<T>, is_base<T>> {};

public:
    using type = type_;
    using type_alias = detail::first_of_t<is_subtype, void, options...>;
    constexpr static bool has_alias = !std::is_void<type_alias>::value;
    using holder_type = detail::first_of_t<is_holder, std::unique_ptr<type>, options...>;
    using instance_type = detail::instance<type, holder_type>;

    static_assert(detail::all_of<is_valid_class_option<options>...>::value,
            "Unknown/invalid class_ template parameters provided");

    PYBIND11_OBJECT(class_, generic_type, PyType_Check)

    template <typename... Extra>
    class_(handle scope, const char *name, const Extra &... extra) {
        detail::type_record record;
        record.scope = scope;
        record.name = name;
        record.type = &typeid(type);
        record.type_size = sizeof(detail::conditional_t<has_alias, type_alias, type>);
        record.instance_size = sizeof(instance_type);
        record.init_holder = init_holder;
        record.dealloc = dealloc;

        /* Register base classes specified via template arguments to class_, if any */
        bool unused[] = { (add_base<options>(record), false)..., false };
        (void) unused;

        /* Process optional arguments, if any */
        detail::process_attributes<Extra...>::init(extra..., &record);

        detail::generic_type::initialize(&record);

        if (has_alias) {
            auto &instances = pybind11::detail::get_internals().registered_types_cpp;
            instances[std::type_index(typeid(type_alias))] = instances[std::type_index(typeid(type))];
        }
    }

    template <typename Base, detail::enable_if_t<is_base<Base>::value, int> = 0>
    static void add_base(detail::type_record &rec) {
        rec.add_base(&typeid(Base), [](void *src) -> void * {
            return static_cast<Base *>(reinterpret_cast<type *>(src));
        });
    }

    template <typename Base, detail::enable_if_t<!is_base<Base>::value, int> = 0>
    static void add_base(detail::type_record &) { }

    template <typename Func, typename... Extra>
    class_ &def(const char *name_, Func&& f, const Extra&... extra) {
        cpp_function cf(std::forward<Func>(f), name(name_), is_method(*this),
                        sibling(getattr(*this, name_, none())), extra...);
        attr(cf.name()) = cf;
        return *this;
    }

    template <typename Func, typename... Extra> class_ &
    def_static(const char *name_, Func f, const Extra&... extra) {
        cpp_function cf(std::forward<Func>(f), name(name_), scope(*this),
                        sibling(getattr(*this, name_, none())), extra...);
        attr(cf.name()) = cf;
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R, typename... Extra>
    class_ &def(const detail::op_<id, ot, L, R> &op, const Extra&... extra) {
        op.execute(*this, extra...);
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R, typename... Extra>
    class_ & def_cast(const detail::op_<id, ot, L, R> &op, const Extra&... extra) {
        op.execute_cast(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(const detail::init<Args...> &init, const Extra&... extra) {
        init.execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(const detail::init_alias<Args...> &init, const Extra&... extra) {
        init.execute(*this, extra...);
        return *this;
    }

    template <typename Func> class_& def_buffer(Func &&func) {
        struct capture { Func func; };
        capture *ptr = new capture { std::forward<Func>(func) };
        install_buffer_funcs([](PyObject *obj, void *ptr) -> buffer_info* {
            detail::make_caster<type> caster;
            if (!caster.load(obj, false))
                return nullptr;
            return new buffer_info(((capture *) ptr)->func(caster));
        }, ptr);
        return *this;
    }

    template <typename C, typename D, typename... Extra>
    class_ &def_readwrite(const char *name, D C::*pm, const Extra&... extra) {
        cpp_function fget([pm](const C &c) -> const D &{ return c.*pm; }, is_method(*this)),
                     fset([pm](C &c, const D &value) { c.*pm = value; }, is_method(*this));
        def_property(name, fget, fset, return_value_policy::reference_internal, extra...);
        return *this;
    }

    template <typename C, typename D, typename... Extra>
    class_ &def_readonly(const char *name, const D C::*pm, const Extra& ...extra) {
        cpp_function fget([pm](const C &c) -> const D &{ return c.*pm; }, is_method(*this));
        def_property_readonly(name, fget, return_value_policy::reference_internal, extra...);
        return *this;
    }

    template <typename D, typename... Extra>
    class_ &def_readwrite_static(const char *name, D *pm, const Extra& ...extra) {
        cpp_function fget([pm](object) -> const D &{ return *pm; }, scope(*this)),
                     fset([pm](object, const D &value) { *pm = value; }, scope(*this));
        def_property_static(name, fget, fset, return_value_policy::reference, extra...);
        return *this;
    }

    template <typename D, typename... Extra>
    class_ &def_readonly_static(const char *name, const D *pm, const Extra& ...extra) {
        cpp_function fget([pm](object) -> const D &{ return *pm; }, scope(*this));
        def_property_readonly_static(name, fget, return_value_policy::reference, extra...);
        return *this;
    }

    /// Uses return_value_policy::reference_internal by default
    template <typename Getter, typename... Extra>
    class_ &def_property_readonly(const char *name, const Getter &fget, const Extra& ...extra) {
        return def_property_readonly(name, cpp_function(fget), return_value_policy::reference_internal, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property_readonly(const char *name, const cpp_function &fget, const Extra& ...extra) {
        return def_property(name, fget, cpp_function(), extra...);
    }

    /// Uses return_value_policy::reference by default
    template <typename Getter, typename... Extra>
    class_ &def_property_readonly_static(const char *name, const Getter &fget, const Extra& ...extra) {
        return def_property_readonly_static(name, cpp_function(fget), return_value_policy::reference, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property_readonly_static(const char *name, const cpp_function &fget, const Extra& ...extra) {
        return def_property_static(name, fget, cpp_function(), extra...);
    }

    /// Uses return_value_policy::reference_internal by default
    template <typename Getter, typename... Extra>
    class_ &def_property(const char *name, const Getter &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property(name, cpp_function(fget), fset, return_value_policy::reference_internal, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property(const char *name, const cpp_function &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property_static(name, fget, fset, is_method(*this), extra...);
    }

    /// Uses return_value_policy::reference by default
    template <typename Getter, typename... Extra>
    class_ &def_property_static(const char *name, const Getter &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property_static(name, cpp_function(fget), fset, return_value_policy::reference, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property_static(const char *name, const cpp_function &fget, const cpp_function &fset, const Extra& ...extra) {
        auto rec_fget = get_function_record(fget), rec_fset = get_function_record(fset);
        char *doc_prev = rec_fget->doc; /* 'extra' field may include a property-specific documentation string */
        detail::process_attributes<Extra...>::init(extra..., rec_fget);
        if (rec_fget->doc && rec_fget->doc != doc_prev) {
            free(doc_prev);
            rec_fget->doc = strdup(rec_fget->doc);
        }
        if (rec_fset) {
            doc_prev = rec_fset->doc;
            detail::process_attributes<Extra...>::init(extra..., rec_fset);
            if (rec_fset->doc && rec_fset->doc != doc_prev) {
                free(doc_prev);
                rec_fset->doc = strdup(rec_fset->doc);
            }
        }
        def_property_static_impl(name, fget, fset, rec_fget);
        return *this;
    }

private:
    /// Initialize holder object, variant 1: object derives from enable_shared_from_this
    template <typename T>
    static void init_holder_helper(instance_type *inst, const holder_type * /* unused */, const std::enable_shared_from_this<T> * /* dummy */) {
        try {
            new (&inst->holder) holder_type(std::static_pointer_cast<typename holder_type::element_type>(inst->value->shared_from_this()));
            inst->holder_constructed = true;
        } catch (const std::bad_weak_ptr &) {
            if (inst->owned) {
                new (&inst->holder) holder_type(inst->value);
                inst->holder_constructed = true;
            }
        }
    }

    /// Initialize holder object, variant 2: try to construct from existing holder object, if possible
    template <typename T = holder_type,
              detail::enable_if_t<std::is_copy_constructible<T>::value, int> = 0>
    static void init_holder_helper(instance_type *inst, const holder_type *holder_ptr, const void * /* dummy */) {
        if (holder_ptr) {
            new (&inst->holder) holder_type(*holder_ptr);
            inst->holder_constructed = true;
        } else if (inst->owned || detail::always_construct_holder<holder_type>::value) {
            new (&inst->holder) holder_type(inst->value);
            inst->holder_constructed = true;
        }
    }

    /// Initialize holder object, variant 3: holder is not copy constructible (e.g. unique_ptr), always initialize from raw pointer
    template <typename T = holder_type,
              detail::enable_if_t<!std::is_copy_constructible<T>::value, int> = 0>
    static void init_holder_helper(instance_type *inst, const holder_type * /* unused */, const void * /* dummy */) {
        if (inst->owned || detail::always_construct_holder<holder_type>::value) {
            new (&inst->holder) holder_type(inst->value);
            inst->holder_constructed = true;
        }
    }

    /// Initialize holder object of an instance, possibly given a pointer to an existing holder
    static void init_holder(PyObject *inst_, const void *holder_ptr) {
        auto inst = (instance_type *) inst_;
        init_holder_helper(inst, (const holder_type *) holder_ptr, inst->value);
    }

    static void dealloc(PyObject *inst_) {
        instance_type *inst = (instance_type *) inst_;
        if (inst->holder_constructed)
            inst->holder.~holder_type();
        else if (inst->owned)
            ::operator delete(inst->value);

        generic_type::dealloc((detail::instance<void> *) inst);
    }

    static detail::function_record *get_function_record(handle h) {
        h = detail::get_function(h);
        return h ? (detail::function_record *) reinterpret_borrow<capsule>(PyCFunction_GET_SELF(h.ptr()))
                 : nullptr;
    }
};

/// Binds C++ enumerations and enumeration classes to Python
template <typename Type> class enum_ : public class_<Type> {
public:
    using class_<Type>::def;
    using Scalar = typename std::underlying_type<Type>::type;
    template <typename T> using arithmetic_tag = std::is_same<T, arithmetic>;

    template <typename... Extra>
    enum_(const handle &scope, const char *name, const Extra&... extra)
      : class_<Type>(scope, name, extra...), m_parent(scope) {

        constexpr bool is_arithmetic =
            !std::is_same<detail::first_of_t<arithmetic_tag, void, Extra...>,
                          void>::value;

        auto entries = new std::unordered_map<Scalar, const char *>();
        def("__repr__", [name, entries](Type value) -> std::string {
            auto it = entries->find((Scalar) value);
            return std::string(name) + "." +
                ((it == entries->end()) ? std::string("???")
                                        : std::string(it->second));
        });
        def("__init__", [](Type& value, Scalar i) { value = (Type)i; });
        def("__init__", [](Type& value, Scalar i) { new (&value) Type((Type) i); });
        def("__int__", [](Type value) { return (Scalar) value; });
        def("__eq__", [](const Type &value, Type *value2) { return value2 && value == *value2; });
        def("__ne__", [](const Type &value, Type *value2) { return !value2 || value != *value2; });
        if (is_arithmetic) {
            def("__lt__", [](const Type &value, Type *value2) { return value2 && value < *value2; });
            def("__gt__", [](const Type &value, Type *value2) { return value2 && value > *value2; });
            def("__le__", [](const Type &value, Type *value2) { return value2 && value <= *value2; });
            def("__ge__", [](const Type &value, Type *value2) { return value2 && value >= *value2; });
        }
        if (std::is_convertible<Type, Scalar>::value) {
            // Don't provide comparison with the underlying type if the enum isn't convertible,
            // i.e. if Type is a scoped enum, mirroring the C++ behaviour.  (NB: we explicitly
            // convert Type to Scalar below anyway because this needs to compile).
            def("__eq__", [](const Type &value, Scalar value2) { return (Scalar) value == value2; });
            def("__ne__", [](const Type &value, Scalar value2) { return (Scalar) value != value2; });
            if (is_arithmetic) {
                def("__lt__", [](const Type &value, Scalar value2) { return (Scalar) value < value2; });
                def("__gt__", [](const Type &value, Scalar value2) { return (Scalar) value > value2; });
                def("__le__", [](const Type &value, Scalar value2) { return (Scalar) value <= value2; });
                def("__ge__", [](const Type &value, Scalar value2) { return (Scalar) value >= value2; });
                def("__invert__", [](const Type &value) { return ~((Scalar) value); });
                def("__and__", [](const Type &value, Scalar value2) { return (Scalar) value & value2; });
                def("__or__", [](const Type &value, Scalar value2) { return (Scalar) value | value2; });
                def("__xor__", [](const Type &value, Scalar value2) { return (Scalar) value ^ value2; });
                def("__rand__", [](const Type &value, Scalar value2) { return (Scalar) value & value2; });
                def("__ror__", [](const Type &value, Scalar value2) { return (Scalar) value | value2; });
                def("__rxor__", [](const Type &value, Scalar value2) { return (Scalar) value ^ value2; });
                def("__and__", [](const Type &value, const Type &value2) { return (Scalar) value & (Scalar) value2; });
                def("__or__", [](const Type &value, const Type &value2) { return (Scalar) value | (Scalar) value2; });
                def("__xor__", [](const Type &value, const Type &value2) { return (Scalar) value ^ (Scalar) value2; });
            }
        }
        def("__hash__", [](const Type &value) { return (Scalar) value; });
        // Pickling and unpickling -- needed for use with the 'multiprocessing' module
        def("__getstate__", [](const Type &value) { return pybind11::make_tuple((Scalar) value); });
        def("__setstate__", [](Type &p, tuple t) { new (&p) Type((Type) t[0].cast<Scalar>()); });
        m_entries = entries;
    }

    /// Export enumeration entries into the parent scope
    enum_ &export_values() {
#if !defined(PYPY_VERSION)
        PyObject *dict = ((PyTypeObject *) this->m_ptr)->tp_dict;
        PyObject *key, *value;
        ssize_t pos = 0;

        while (PyDict_Next(dict, &pos, &key, &value)) {
            if (PyObject_IsInstance(value, this->m_ptr))
                m_parent.attr(key) = value;
        }
#else
        /* PyPy's cpyext still has difficulties with the above
           CPython API calls; emulate using Python code. */
        dict d; d["t"] = *this; d["p"] = m_parent;
        PyObject *result = PyRun_String(
            "for k, v in t.__dict__.items():\n"
            "    if isinstance(v, t):\n"
            "        setattr(p, k, v)\n",
            Py_file_input, d.ptr(), d.ptr());
        if (result == nullptr)
            throw error_already_set();
        Py_DECREF(result);
#endif

        return *this;
    }

    /// Add an enumeration entry
    enum_& value(char const* name, Type value) {
        this->attr(name) = pybind11::cast(value, return_value_policy::copy);
        (*m_entries)[(Scalar) value] = name;
        return *this;
    }
private:
    std::unordered_map<Scalar, const char *> *m_entries;
    handle m_parent;
};

NAMESPACE_BEGIN(detail)
template <typename... Args> struct init {
    template <typename Class, typename... Extra, enable_if_t<!Class::has_alias, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        using Base = typename Class::type;
        /// Function which calls a specific C++ in-place constructor
        cl.def("__init__", [](Base *self_, Args... args) { new (self_) Base(args...); }, extra...);
    }

    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias &&
                          std::is_constructible<typename Class::type, Args...>::value, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        using Base = typename Class::type;
        using Alias = typename Class::type_alias;
        handle cl_type = cl;
        cl.def("__init__", [cl_type](handle self_, Args... args) {
                if (self_.get_type() == cl_type)
                    new (self_.cast<Base *>()) Base(args...);
                else
                    new (self_.cast<Alias *>()) Alias(args...);
            }, extra...);
    }

    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias &&
                          !std::is_constructible<typename Class::type, Args...>::value, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        init_alias<Args...>::execute(cl, extra...);
    }
};
template <typename... Args> struct init_alias {
    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias && std::is_constructible<typename Class::type_alias, Args...>::value, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        using Alias = typename Class::type_alias;
        cl.def("__init__", [](Alias *self_, Args... args) { new (self_) Alias(args...); }, extra...);
    }
};


inline void keep_alive_impl(handle nurse, handle patient) {
    /* Clever approach based on weak references taken from Boost.Python */
    if (!nurse || !patient)
        pybind11_fail("Could not activate keep_alive!");

    if (patient.is_none() || nurse.is_none())
        return; /* Nothing to keep alive or nothing to be kept alive by */

    cpp_function disable_lifesupport(
        [patient](handle weakref) { patient.dec_ref(); weakref.dec_ref(); });

    weakref wr(nurse, disable_lifesupport);

    patient.inc_ref(); /* reference patient and leak the weak reference */
    (void) wr.release();
}

PYBIND11_NOINLINE inline void keep_alive_impl(int Nurse, int Patient, handle args, handle ret) {
    handle nurse  (Nurse   > 0 ? PyTuple_GetItem(args.ptr(), Nurse   - 1) : ret.ptr());
    handle patient(Patient > 0 ? PyTuple_GetItem(args.ptr(), Patient - 1) : ret.ptr());

    keep_alive_impl(nurse, patient);
}

template <typename Iterator, typename Sentinel, bool KeyIterator, return_value_policy Policy>
struct iterator_state {
    Iterator it;
    Sentinel end;
    bool first;
};

NAMESPACE_END(detail)

template <typename... Args> detail::init<Args...> init() { return detail::init<Args...>(); }
template <typename... Args> detail::init_alias<Args...> init_alias() { return detail::init_alias<Args...>(); }

template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Iterator,
          typename Sentinel,
          typename ValueType = decltype(*std::declval<Iterator>()),
          typename... Extra>
iterator make_iterator(Iterator first, Sentinel last, Extra &&... extra) {
    typedef detail::iterator_state<Iterator, Sentinel, false, Policy> state;

    if (!detail::get_type_info(typeid(state), false)) {
        class_<state>(handle(), "iterator")
            .def("__iter__", [](state &s) -> state& { return s; })
            .def("__next__", [](state &s) -> ValueType {
                if (!s.first)
                    ++s.it;
                else
                    s.first = false;
                if (s.it == s.end)
                    throw stop_iteration();
                return *s.it;
            }, std::forward<Extra>(extra)..., Policy);
    }

    return (iterator) cast(state { first, last, true });
}

template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Iterator,
          typename Sentinel,
          typename KeyType = decltype((*std::declval<Iterator>()).first),
          typename... Extra>
iterator make_key_iterator(Iterator first, Sentinel last, Extra &&... extra) {
    typedef detail::iterator_state<Iterator, Sentinel, true, Policy> state;

    if (!detail::get_type_info(typeid(state), false)) {
        class_<state>(handle(), "iterator")
            .def("__iter__", [](state &s) -> state& { return s; })
            .def("__next__", [](state &s) -> KeyType {
                if (!s.first)
                    ++s.it;
                else
                    s.first = false;
                if (s.it == s.end)
                    throw stop_iteration();
                return (*s.it).first;
            }, std::forward<Extra>(extra)..., Policy);
    }

    return (iterator) cast(state { first, last, true });
}

template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Type, typename... Extra> iterator make_iterator(Type &value, Extra&&... extra) {
    return make_iterator<Policy>(std::begin(value), std::end(value), extra...);
}

template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Type, typename... Extra> iterator make_key_iterator(Type &value, Extra&&... extra) {
    return make_key_iterator<Policy>(std::begin(value), std::end(value), extra...);
}

template <typename InputType, typename OutputType> void implicitly_convertible() {
    auto implicit_caster = [](PyObject *obj, PyTypeObject *type) -> PyObject * {
        if (!detail::make_caster<InputType>().load(obj, false))
            return nullptr;
        tuple args(1);
        args[0] = obj;
        PyObject *result = PyObject_Call((PyObject *) type, args.ptr(), nullptr);
        if (result == nullptr)
            PyErr_Clear();
        return result;
    };

    if (auto tinfo = detail::get_type_info(typeid(OutputType)))
        tinfo->implicit_conversions.push_back(implicit_caster);
    else
        pybind11_fail("implicitly_convertible: Unable to find type " + type_id<OutputType>());
}

template <typename ExceptionTranslator>
void register_exception_translator(ExceptionTranslator&& translator) {
    detail::get_internals().registered_exception_translators.push_front(
        std::forward<ExceptionTranslator>(translator));
}

/* Wrapper to generate a new Python exception type.
 *
 * This should only be used with PyErr_SetString for now.
 * It is not (yet) possible to use as a py::base.
 * Template type argument is reserved for future use.
 */
template <typename type>
class exception : public object {
public:
    exception(handle scope, const char *name, PyObject *base = PyExc_Exception) {
        std::string full_name = scope.attr("__name__").cast<std::string>() +
                                std::string(".") + name;
        m_ptr = PyErr_NewException((char *) full_name.c_str(), base, NULL);
        if (hasattr(scope, name))
            pybind11_fail("Error during initialization: multiple incompatible "
                          "definitions with name \"" + std::string(name) + "\"");
        scope.attr(name) = *this;
    }

    // Sets the current python exception to this exception object with the given message
    void operator()(const char *message) {
        PyErr_SetString(m_ptr, message);
    }
};

/** Registers a Python exception in `m` of the given `name` and installs an exception translator to
 * translate the C++ exception to the created Python exception using the exceptions what() method.
 * This is intended for simple exception translations; for more complex translation, register the
 * exception object and translator directly.
 */
template <typename CppException>
exception<CppException> &register_exception(handle scope,
                                            const char *name,
                                            PyObject *base = PyExc_Exception) {
    static exception<CppException> ex(scope, name, base);
    register_exception_translator([](std::exception_ptr p) {
        if (!p) return;
        try {
            std::rethrow_exception(p);
        } catch (const CppException &e) {
            ex(e.what());
        }
    });
    return ex;
}

NAMESPACE_BEGIN(detail)
PYBIND11_NOINLINE inline void print(tuple args, dict kwargs) {
    auto strings = tuple(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        strings[i] = str(args[i]);
    }
    auto sep = kwargs.contains("sep") ? kwargs["sep"] : cast(" ");
    auto line = sep.attr("join")(strings);

    object file;
    if (kwargs.contains("file")) {
        file = kwargs["file"].cast<object>();
    } else {
        try {
            file = module::import("sys").attr("stdout");
        } catch (const error_already_set &) {
            /* If print() is called from code that is executed as
               part of garbage collection during interpreter shutdown,
               importing 'sys' can fail. Give up rather than crashing the
               interpreter in this case. */
            return;
        }
    }

    auto write = file.attr("write");
    write(line);
    write(kwargs.contains("end") ? kwargs["end"] : cast("\n"));

    if (kwargs.contains("flush") && kwargs["flush"].cast<bool>())
        file.attr("flush")();
}
NAMESPACE_END(detail)

template <return_value_policy policy = return_value_policy::automatic_reference, typename... Args>
void print(Args &&...args) {
    auto c = detail::collect_arguments<policy>(std::forward<Args>(args)...);
    detail::print(c.args(), c.kwargs());
}

#if defined(WITH_THREAD) && !defined(PYPY_VERSION)

/* The functions below essentially reproduce the PyGILState_* API using a RAII
 * pattern, but there are a few important differences:
 *
 * 1. When acquiring the GIL from an non-main thread during the finalization
 *    phase, the GILState API blindly terminates the calling thread, which
 *    is often not what is wanted. This API does not do this.
 *
 * 2. The gil_scoped_release function can optionally cut the relationship
 *    of a PyThreadState and its associated thread, which allows moving it to
 *    another thread (this is a fairly rare/advanced use case).
 *
 * 3. The reference count of an acquired thread state can be controlled. This
 *    can be handy to prevent cases where callbacks issued from an external
 *    thread would otherwise constantly construct and destroy thread state data
 *    structures.
 *
 * See the Python bindings of NanoGUI (http://github.com/wjakob/nanogui) for an
 * example which uses features 2 and 3 to migrate the Python thread of
 * execution to another thread (to run the event loop on the original thread,
 * in this case).
 */

class gil_scoped_acquire {
public:
    PYBIND11_NOINLINE gil_scoped_acquire() {
        auto const &internals = detail::get_internals();
        tstate = (PyThreadState *) PyThread_get_key_value(internals.tstate);

        if (!tstate) {
            tstate = PyThreadState_New(internals.istate);
            #if !defined(NDEBUG)
                if (!tstate)
                    pybind11_fail("scoped_acquire: could not create thread state!");
            #endif
            tstate->gilstate_counter = 0;
            #if PY_MAJOR_VERSION < 3
                PyThread_delete_key_value(internals.tstate);
            #endif
            PyThread_set_key_value(internals.tstate, tstate);
        } else {
            release = detail::get_thread_state_unchecked() != tstate;
        }

        if (release) {
            /* Work around an annoying assertion in PyThreadState_Swap */
            #if defined(Py_DEBUG)
                PyInterpreterState *interp = tstate->interp;
                tstate->interp = nullptr;
            #endif
            PyEval_AcquireThread(tstate);
            #if defined(Py_DEBUG)
                tstate->interp = interp;
            #endif
        }

        inc_ref();
    }

    void inc_ref() {
        ++tstate->gilstate_counter;
    }

    PYBIND11_NOINLINE void dec_ref() {
        --tstate->gilstate_counter;
        #if !defined(NDEBUG)
            if (detail::get_thread_state_unchecked() != tstate)
                pybind11_fail("scoped_acquire::dec_ref(): thread state must be current!");
            if (tstate->gilstate_counter < 0)
                pybind11_fail("scoped_acquire::dec_ref(): reference count underflow!");
        #endif
        if (tstate->gilstate_counter == 0) {
            #if !defined(NDEBUG)
                if (!release)
                    pybind11_fail("scoped_acquire::dec_ref(): internal error!");
            #endif
            PyThreadState_Clear(tstate);
            PyThreadState_DeleteCurrent();
            PyThread_delete_key_value(detail::get_internals().tstate);
            release = false;
        }
    }

    PYBIND11_NOINLINE ~gil_scoped_acquire() {
        dec_ref();
        if (release)
           PyEval_SaveThread();
    }
private:
    PyThreadState *tstate = nullptr;
    bool release = true;
};

class gil_scoped_release {
public:
    explicit gil_scoped_release(bool disassoc = false) : disassoc(disassoc) {
        tstate = PyEval_SaveThread();
        if (disassoc) {
            auto key = detail::get_internals().tstate;
            #if PY_MAJOR_VERSION < 3
                PyThread_delete_key_value(key);
            #else
                PyThread_set_key_value(key, nullptr);
            #endif
        }
    }
    ~gil_scoped_release() {
        if (!tstate)
            return;
        PyEval_RestoreThread(tstate);
        if (disassoc) {
            auto key = detail::get_internals().tstate;
            #if PY_MAJOR_VERSION < 3
                PyThread_delete_key_value(key);
            #endif
            PyThread_set_key_value(key, tstate);
        }
    }
private:
    PyThreadState *tstate;
    bool disassoc;
};
#elif defined(PYPY_VERSION)
class gil_scoped_acquire {
    PyGILState_STATE state;
public:
    gil_scoped_acquire() { state = PyGILState_Ensure(); }
    ~gil_scoped_acquire() { PyGILState_Release(state); }
};

class gil_scoped_release {
    PyThreadState *state;
public:
    gil_scoped_release() { state = PyEval_SaveThread(); }
    ~gil_scoped_release() { PyEval_RestoreThread(state); }
};
#else
class gil_scoped_acquire { };
class gil_scoped_release { };
#endif

error_already_set::~error_already_set() {
    if (value) {
        gil_scoped_acquire gil;
        PyErr_Restore(type, value, trace);
        PyErr_Clear();
    }
}

inline function get_type_overload(const void *this_ptr, const detail::type_info *this_type, const char *name)  {
    handle self = detail::get_object_handle(this_ptr, this_type);
    if (!self)
        return function();
    handle type = self.get_type();
    auto key = std::make_pair(type.ptr(), name);

    /* Cache functions that aren't overloaded in Python to avoid
       many costly Python dictionary lookups below */
    auto &cache = detail::get_internals().inactive_overload_cache;
    if (cache.find(key) != cache.end())
        return function();

    function overload = getattr(self, name, function());
    if (overload.is_cpp_function()) {
        cache.insert(key);
        return function();
    }

    /* Don't call dispatch code if invoked from overridden function.
       Unfortunately this doesn't work on PyPy. */
#if !defined(PYPY_VERSION)
    PyFrameObject *frame = PyThreadState_Get()->frame;
    if (frame && (std::string) str(frame->f_code->co_name) == name &&
        frame->f_code->co_argcount > 0) {
        PyFrame_FastToLocals(frame);
        PyObject *self_caller = PyDict_GetItem(
            frame->f_locals, PyTuple_GET_ITEM(frame->f_code->co_varnames, 0));
        if (self_caller == self.ptr())
            return function();
    }
#else
    /* PyPy currently doesn't provide a detailed cpyext emulation of
       frame objects, so we have to emulate this using Python. This
       is going to be slow..*/
    dict d; d["self"] = self; d["name"] = pybind11::str(name);
    PyObject *result = PyRun_String(
        "import inspect\n"
        "frame = inspect.currentframe()\n"
        "if frame is not None:\n"
        "    frame = frame.f_back\n"
        "    if frame is not None and str(frame.f_code.co_name) == name and "
        "frame.f_code.co_argcount > 0:\n"
        "        self_caller = frame.f_locals[frame.f_code.co_varnames[0]]\n"
        "        if self_caller == self:\n"
        "            self = None\n",
        Py_file_input, d.ptr(), d.ptr());
    if (result == nullptr)
        throw error_already_set();
    if ((handle) d["self"] == Py_None)
        return function();
    Py_DECREF(result);
#endif

    return overload;
}

template <class T> function get_overload(const T *this_ptr, const char *name) {
    auto tinfo = detail::get_type_info(typeid(T));
    return tinfo ? get_type_overload(this_ptr, tinfo, name) : function();
}

#define PYBIND11_OVERLOAD_INT(ret_type, cname, name, ...) { \
        pybind11::gil_scoped_acquire gil; \
        pybind11::function overload = pybind11::get_overload(static_cast<const cname *>(this), name); \
        if (overload) { \
            auto o = overload(__VA_ARGS__); \
            if (pybind11::detail::cast_is_temporary_value_reference<ret_type>::value) { \
                static pybind11::detail::overload_caster_t<ret_type> caster; \
                return pybind11::detail::cast_ref<ret_type>(std::move(o), caster); \
            } \
            else return pybind11::detail::cast_safe<ret_type>(std::move(o)); \
        } \
    }

#define PYBIND11_OVERLOAD_NAME(ret_type, cname, name, fn, ...) \
    PYBIND11_OVERLOAD_INT(ret_type, cname, name, __VA_ARGS__) \
    return cname::fn(__VA_ARGS__)

#define PYBIND11_OVERLOAD_PURE_NAME(ret_type, cname, name, fn, ...) \
    PYBIND11_OVERLOAD_INT(ret_type, cname, name, __VA_ARGS__) \
    pybind11::pybind11_fail("Tried to call pure virtual function \"" #cname "::" name "\"");

#define PYBIND11_OVERLOAD(ret_type, cname, fn, ...) \
    PYBIND11_OVERLOAD_NAME(ret_type, cname, #fn, fn, __VA_ARGS__)

#define PYBIND11_OVERLOAD_PURE(ret_type, cname, fn, ...) \
    PYBIND11_OVERLOAD_PURE_NAME(ret_type, cname, #fn, fn, __VA_ARGS__)

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#  pragma warning(pop)
#elif defined(__INTEL_COMPILER)
/* Leave ignored warnings on */
#elif defined(__GNUG__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif
