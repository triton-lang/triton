/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef HELPER_MATH_H_
#define HELPER_MATH_H_


template <class DTYPE>
inline __device__ DTYPE infinity() { return __int_as_float(0x7f800000); }

template<>
inline __device__ double infinity<double>() { return __hiloint2double(0x7ff00000, 0x00000000) ; }

typedef unsigned int uint; 
typedef unsigned short ushort;

template<bool B, class T = void>
struct enable_if {};

template<class T>
struct enable_if<true, T> { typedef T type; };

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

#define GENERATE_CONSTRUCTORS(TYPE)\
inline __host__ __device__  TYPE make_ ## TYPE(TYPE s)\
{ return s; }\
\
inline __host__ __device__  TYPE ## 2 make_ ## TYPE ## 2(TYPE ## 2 s)\
{ return s;}\
\
inline __host__ __device__  TYPE ## 2 make_ ## TYPE ## 2(TYPE s)\
{ return make_ ## TYPE ## 2(s, s);}\
\
inline __host__ __device__  TYPE ## 2 make_ ## TYPE ## 2( ## TYPE ## 3 a)\
{ return make_ ## TYPE ## 2(a.x, a.y); }\
\
inline __host__ __device__  TYPE ## 3 make_ ## TYPE ## 3(TYPE s)\
{ return make_ ## TYPE ## 3(s, s, s); }\
\
inline __host__ __device__  TYPE ## 3 make_ ## TYPE ## 3(TYPE ## 3 s)\
{ return s;}\
\
inline __host__ __device__  TYPE ## 3 make_ ## TYPE ## 3( ## TYPE ## 2 a)\
{ return make_ ## TYPE ## 3(a.x, a.y, 0.0f); }\
\
inline __host__ __device__  TYPE ## 3 make_ ## TYPE ## 3( ## TYPE ## 2 a, TYPE s)\
{ return make_ ## TYPE ## 3(a.x, a.y, s);}\
\
inline __host__ __device__  TYPE ## 3 make_ ## TYPE ## 3( ## TYPE ## 4 a)\
{ return make_ ## TYPE ## 3(a.x, a.y, a.z); }\
\
inline __host__ __device__  TYPE ## 4 make_ ## TYPE ## 4(TYPE s)\
{ return make_ ## TYPE ## 4(s, s, s, s); }\
\
inline __host__ __device__  TYPE ## 4 make_ ## TYPE ## 4(TYPE ## 4 s)\
{ return s;}\
\
inline __host__ __device__  TYPE ## 4 make_ ## TYPE ## 4( ## TYPE ## 3 a)\
{ return make_ ## TYPE ## 4(a.x, a.y, a.z, 0.0f); }\
\
inline __host__ __device__  TYPE ## 4 make_ ## TYPE ## 4( ## TYPE ## 3 a, TYPE w)\
{ return make_ ## TYPE ## 4(a.x, a.y, a.z, w); }\


#define GENERATE_CONSTRUCTORS_FROM(TYPE, FROM)\
inline __host__ __device__  TYPE ## 2 make_ ## TYPE ## 2(FROM ## 2 a)\
{ return make_ ## TYPE ## 2(TYPE(a.x), TYPE(a.y)); }\
\
inline __host__ __device__  TYPE ## 3 make_ ## TYPE ## 3(FROM ## 3 a)\
{ return make_ ## TYPE ## 3(TYPE(a.x), TYPE(a.y), TYPE(a.z)); }\
\
inline __host__ __device__  TYPE ## 4 make_ ## TYPE ## 4(FROM ## 4 a)\
{ return make_ ## TYPE ## 4(TYPE(a.x), TYPE(a.y), TYPE(a.z), TYPE(a.w)); }


GENERATE_CONSTRUCTORS(float)
GENERATE_CONSTRUCTORS_FROM(float, int)
GENERATE_CONSTRUCTORS_FROM(float, uint)

GENERATE_CONSTRUCTORS(double)
GENERATE_CONSTRUCTORS_FROM(double, int)
GENERATE_CONSTRUCTORS_FROM(double, uint)

template<class T> struct base_of;
template<class T> struct is_vector2_impl { enum {value = false}; };
template<class T> struct is_vector3_impl { enum {value = false}; };
template<class T> struct is_vector4_impl { enum {value = false}; };
template<class T> inline __host__ __device__ T make(typename base_of<T>::type x, typename base_of<T>::type y);
template<class T> inline __host__ __device__ T make(typename base_of<T>::type x, typename base_of<T>::type y, typename base_of<T>::type z);
template<class T> inline __host__ __device__ T make(typename base_of<T>::type x, typename base_of<T>::type y, typename base_of<T>::type z, typename base_of<T>::type w);

template<class T> struct is_scalar { enum {value = !is_vector2_impl<T>::value && !is_vector3_impl<T>::value && !is_vector4_impl<T>::value}; };
template<class T> struct is_vector2 { enum {value = is_vector2_impl<T>::value && !is_vector3_impl<T>::value && !is_vector4_impl<T>::value}; };
template<class T> struct is_vector3 { enum {value = !is_vector2_impl<T>::value && is_vector3_impl<T>::value && !is_vector4_impl<T>::value}; };
template<class T> struct is_vector4 { enum {value = !is_vector2_impl<T>::value && !is_vector3_impl<T>::value && is_vector4_impl<T>::value}; };

#define INSTANTIATE_VECTOR_TYPE(NAME) \
  template<> struct base_of<NAME> { typedef NAME type; };\
  template<> struct base_of<NAME ## 2> { typedef NAME type; };\
  template<> struct base_of<NAME ## 3> { typedef NAME type; };\
  template<> struct base_of<NAME ## 4> { typedef NAME type; };\
  template<> struct is_vector2<NAME ## 2> { enum{value = true}; };\
  template<> struct is_vector3<NAME ## 3> { enum{value = true}; };\
  template<> struct is_vector4<NAME ## 4> { enum{value = true}; };\
  template<> inline __host__ __device__ NAME ## 2 make<NAME ## 2>(NAME x, NAME y) { return make_ ## NAME ## 2(x, y); }\
  template<> inline __host__ __device__ NAME ## 3 make<NAME ## 3>(NAME x, NAME y, NAME z) { return make_ ## NAME ## 3(x, y, z); }\
  template<> inline __host__ __device__ NAME ## 4 make<NAME ## 4>(NAME x, NAME y, NAME z, NAME w) { return make_ ## NAME ## 4(x, y, z, w); }


INSTANTIATE_VECTOR_TYPE(float)
INSTANTIATE_VECTOR_TYPE(double)






////////////////////////////////////////////////////////////////////////////////
// Unary
////////////////////////////////////////////////////////////////////////////////

#define ADD_UNARY_OPERATOR(RET, SYMBOL) \
template<class T>\
inline __host__ __device__ typename enable_if<is_vector2<T>::value, RET>::type operator ## SYMBOL (T &a)\
{ return make<T>(SYMBOL a.x, SYMBOL a.y); }\
\
template<class T>\
inline __host__ __device__ typename enable_if<is_vector3<T>::value, RET>::type operator ## SYMBOL(T &a)\
{ return make<T>(SYMBOL a.x, SYMBOL a.y, SYMBOL a.z); }\
\
template<class T>\
inline __host__ __device__ typename enable_if<is_vector4<T>::value, RET>::type operator ## SYMBOL(T &a)\
{ return make<T>(SYMBOL a.x, SYMBOL a.y, SYMBOL a.z, SYMBOL a.w); }

#define ADD_UNARY_FUNCTION(RET, FUN) \
template<class T>\
inline __host__ __device__ typename enable_if<is_vector2<T>::value, RET>::type FUN (T a)\
{ return make<T>(FUN (a.x), FUN (a.y)); }\
\
template<class T>\
inline __host__ __device__ typename enable_if<is_vector3<T>::value, RET>::type FUN(T a)\
{ return make<T>(FUN (a.x), FUN (a.y), FUN (a.z)); }\
\
template<class T>\
inline __host__ __device__ typename enable_if<is_vector4<T>::value, RET>::type FUN(T a)\
{ return make<T>(FUN (a.x), FUN (a.y), FUN (a.z), FUN (a.w)); }

ADD_UNARY_OPERATOR(T, -)
ADD_UNARY_FUNCTION(T, exp)
ADD_UNARY_FUNCTION(T, acos)
ADD_UNARY_FUNCTION(T, asin)
ADD_UNARY_FUNCTION(T, atan)
ADD_UNARY_FUNCTION(T, ceil)
ADD_UNARY_FUNCTION(T, cos)
ADD_UNARY_FUNCTION(T, cosh)
ADD_UNARY_FUNCTION(T, fabs)
ADD_UNARY_FUNCTION(T, floor)
ADD_UNARY_FUNCTION(T, floorf)
ADD_UNARY_FUNCTION(T, fracf)
ADD_UNARY_FUNCTION(T, log)
ADD_UNARY_FUNCTION(T, log10)
ADD_UNARY_FUNCTION(T, sin)
ADD_UNARY_FUNCTION(T, sinh)
ADD_UNARY_FUNCTION(T, sqrt)
ADD_UNARY_FUNCTION(T, tan)
ADD_UNARY_FUNCTION(T, tanh)

////////////////////////////////////////////////////////////////////////////////
// Binary
////////////////////////////////////////////////////////////////////////////////
#define ADD_BINARY_OPERATOR(RET, SYMBOL) \
  template<class T>\
  inline __host__ __device__ typename enable_if<is_vector2<T>::value, RET>::type operator ## SYMBOL(T a, T b)\
  { return make<T>(a.x SYMBOL b.x, a.y SYMBOL b.y); }\
\
  template<class T>\
  inline __host__ __device__ typename enable_if<is_vector2<T>::value, RET>::type operator ## SYMBOL(typename base_of<T>::type a, T b)\
  { return make<T>(a SYMBOL b.x, a SYMBOL b.y); }\
\
  template<class T>\
  inline __host__ __device__ typename enable_if<is_vector2<T>::value, RET>::type operator ## SYMBOL(T a, typename base_of<T>::type b)\
  { return make<T>(a.x SYMBOL b, a.y SYMBOL b); }\
\
  template<class T>\
  inline __host__ __device__ typename enable_if<is_vector3<T>::value, RET>::type operator ## SYMBOL(T a, T b)\
  { return make<T>(a.x SYMBOL b.x, a.y SYMBOL b.y, a.z SYMBOL b.z); }\
\
  template<class T>\
  inline __host__ __device__ typename enable_if<is_vector3<T>::value, RET>::type operator ## SYMBOL(typename base_of<T>::type a, T b)\
  { return make<T>(a SYMBOL b.x, a SYMBOL b.y, a SYMBOL b.z); }\
\
  template<class T>\
  inline __host__ __device__ typename enable_if<is_vector3<T>::value, RET>::type operator ## SYMBOL(T a, typename base_of<T>::type b)\
  { return make<T>(a.x SYMBOL b, a.y SYMBOL b, a.z SYMBOL b); }\
\
  template<class T>\
  inline __host__ __device__ typename enable_if<is_vector4<T>::value, RET>::type operator ## SYMBOL(T a, T b)\
  { return make<T>(a.x SYMBOL b.x, a.y SYMBOL b.y, a.z SYMBOL b.z, a.w SYMBOL b.w); }\
\
  template<class T>\
  inline __host__ __device__ typename enable_if<is_vector4<T>::value, RET>::type operator ## SYMBOL(typename base_of<T>::type a, T b)\
  { return make<T>(a SYMBOL b.x, a SYMBOL b.y, a SYMBOL b.z, a SYMBOL b.w); }\
\
  template<class T>\
  inline __host__ __device__ typename enable_if<is_vector4<T>::value, RET>::type operator ## SYMBOL(T a, typename base_of<T>::type b)\
  { return make<T>(a.x SYMBOL b, a.y SYMBOL b, a.z SYMBOL b, a.w SYMBOL b); }


#define ADD_INPLACE_OPERATOR(SYMBOL)\
  template<class T>\
  inline __host__ __device__ typename enable_if<is_vector2<T>::value, void>::type operator ## SYMBOL (T& a, T b)\
  {\
    a.x SYMBOL b.x;\
    a.y SYMBOL b.y;\
  }\
\
  template<class T>\
  inline __host__ __device__ typename enable_if<is_vector3<T>::value, void>::type operator ## SYMBOL (T& a, T b)\
  {\
    a.x SYMBOL b.x;\
    a.y SYMBOL b.y;\
    a.z SYMBOL b.z;\
  }\
\
  template<class T>\
  inline __host__ __device__ typename enable_if<is_vector4<T>::value, void>::type operator ## SYMBOL (T& a, T b)\
  {\
    a.x SYMBOL b.x;\
    a.y SYMBOL b.y;\
    a.z SYMBOL b.z;\
    a.w SYMBOL b.w;\
  }\

#define ADD_BINARY_FUNCTION(RET, FUN) \
template<class T>\
inline __host__ __device__ typename enable_if<is_vector2<T>::value, RET>::type FUN (T a, T b)\
{ return make<T>(FUN (a.x, b.x), FUN (a.y, b.y)); }\
\
template<class T>\
inline __host__ __device__ typename enable_if<is_vector3<T>::value, RET>::type FUN(T a, T b)\
{ return make<T>(FUN (a.x, b.x), FUN (a.y, b.y), FUN (a.z, b.z)); }\
\
template<class T>\
inline __host__ __device__ typename enable_if<is_vector4<T>::value, RET>::type FUN(T a, T b)\
{ return make<T>(FUN (a.x, b.x), FUN (a.y, b.y), FUN (a.z, b.z), FUN (a.w, b.w)); }


ADD_BINARY_OPERATOR(T, ==)
ADD_BINARY_OPERATOR(T, !=)
ADD_BINARY_OPERATOR(T, <=)
ADD_BINARY_OPERATOR(T, <)
ADD_BINARY_OPERATOR(T, >=)
ADD_BINARY_OPERATOR(T, >)
ADD_BINARY_OPERATOR(T, +)
ADD_BINARY_OPERATOR(T, -)
ADD_BINARY_OPERATOR(T, *)
ADD_BINARY_OPERATOR(T, /)

ADD_INPLACE_OPERATOR(+=)
ADD_INPLACE_OPERATOR(-=)
ADD_INPLACE_OPERATOR(*=)
ADD_INPLACE_OPERATOR(/=)

ADD_BINARY_FUNCTION(T, pow)
ADD_BINARY_FUNCTION(T, fminf)
ADD_BINARY_FUNCTION(T, fmaxf)
ADD_BINARY_FUNCTION(T, fmodf)
ADD_BINARY_FUNCTION(T, min)
ADD_BINARY_FUNCTION(T, max)

template<class T>
inline __host__ __device__ typename enable_if<is_scalar<T>::value || is_vector2<T>::value || is_vector3<T>::value || is_vector4<T>::value, T>::type
        clamp (T f, typename base_of<T>::type a, typename base_of<T>::type b)
{ return fmaxf(a, fminf(f, b)); }

template<class T>
inline __host__ __device__ typename enable_if<is_vector2<T>::value || is_vector3<T>::value || is_vector4<T>::value, T>::type clamp (T f, T a, T b)
{ return fmaxf(a, fminf(f, b)); }

//dot
template<class T>
inline __host__ __device__ typename enable_if<is_vector2<T>::value, typename base_of<T>::type>::type FUN (T a, T b)
{ return a.x * b.x + a.y * b.y; }

template<class T>
inline __host__ __device__ typename enable_if<is_vector3<T>::value, typename base_of<T>::type>::type FUN(T a, T b)
{ return a.x * b.x + a.y * b.y + a.z * b.z; }\

template<class T>
inline __host__ __device__ typename enable_if<is_vector4<T>::value, typename base_of<T>::type>::type FUN(T a, T b)
{ return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

#endif
