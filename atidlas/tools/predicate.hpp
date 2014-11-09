#ifndef ATIDLAS_PREDICATE_HPP_
#define ATIDLAS_PREDICATE_HPP_

namespace atidlas
{

/** @brief Helper class for checking whether a type is a primitive type. */
template<class T>
struct is_primitive_type{ enum {value = false}; };

template<> struct is_primitive_type<float>         { enum { value = true }; };
template<> struct is_primitive_type<double>        { enum { value = true }; };
template<> struct is_primitive_type<unsigned int>  { enum { value = true }; };
template<> struct is_primitive_type<int>           { enum { value = true }; };
template<> struct is_primitive_type<unsigned char> { enum { value = true }; };
template<> struct is_primitive_type<char>          { enum { value = true }; };
template<> struct is_primitive_type<unsigned long> { enum { value = true }; };
template<> struct is_primitive_type<long>          { enum { value = true }; };
template<> struct is_primitive_type<unsigned short>{ enum { value = true }; };
template<> struct is_primitive_type<short>         { enum { value = true }; };


/** @brief Helper class for checking whether a particular type is a native OpenCL type. */
template<class T>
struct is_cl_type{ enum { value = false }; };

template<> struct is_cl_type<cl_float> { enum { value = true }; };
template<> struct is_cl_type<cl_double>{ enum { value = true }; };
template<> struct is_cl_type<cl_uint>  { enum { value = true }; };
template<> struct is_cl_type<cl_int>   { enum { value = true }; };
template<> struct is_cl_type<cl_uchar> { enum { value = true }; };
template<> struct is_cl_type<cl_char>  { enum { value = true }; };
template<> struct is_cl_type<cl_ulong> { enum { value = true }; };
template<> struct is_cl_type<cl_long>  { enum { value = true }; };
template<> struct is_cl_type<cl_ushort>{ enum { value = true }; };
template<> struct is_cl_type<cl_short> { enum { value = true }; };

/** @brief Helper class for checking whether a particular type is a floating point type. */
template<class T> struct is_floating_point { enum { value = false }; };
template<> struct is_floating_point<float> { enum { value = true }; };
template<> struct is_floating_point<double> { enum { value = true }; };


}


#endif
