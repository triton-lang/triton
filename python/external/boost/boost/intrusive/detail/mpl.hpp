/////////////////////////////////////////////////////////////////////////////
//
// (C) Copyright Ion Gaztanaga        2006-2014
// (C) Copyright Microsoft Corporation  2014
//
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/libs/intrusive for documentation.
//
/////////////////////////////////////////////////////////////////////////////

#ifndef BOOST_INTRUSIVE_DETAIL_MPL_HPP
#define BOOST_INTRUSIVE_DETAIL_MPL_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/intrusive/detail/config_begin.hpp>
#include <cstddef>

namespace boost {
namespace intrusive {
namespace detail {

template <typename T, typename U>
struct is_same
{
   static const bool value = false;
};

template <typename T>
struct is_same<T, T>
{
   static const bool value = true;
};

template<typename T>
struct add_const
{  typedef const T type;   };

template<typename T>
struct remove_const
{  typedef  T type;   };

template<typename T>
struct remove_const<const T>
{  typedef T type;   };

template<typename T>
struct remove_cv
{  typedef  T type;   };

template<typename T>
struct remove_cv<const T>
{  typedef T type;   };

template<typename T>
struct remove_cv<const volatile T>
{  typedef T type;   };

template<typename T>
struct remove_cv<volatile T>
{  typedef T type;   };

template<class T>
struct remove_reference
{
   typedef T type;
};

template<class T>
struct remove_reference<T&>
{
   typedef T type;
};

template<class T>
struct remove_pointer
{
   typedef T type;
};

template<class T>
struct remove_pointer<T*>
{
   typedef T type;
};

template<class T>
struct add_pointer
{
   typedef T *type;
};

typedef char one;
struct two {one _[2];};

template< bool C_ >
struct bool_
{
   static const bool value = C_;
};

template< class Integer, Integer Value >
struct integer
{
   static const Integer value = Value;
};

typedef bool_<true>        true_;
typedef bool_<false>       false_;

typedef true_  true_type;
typedef false_ false_type;

typedef char yes_type;
struct no_type
{
   char padding[8];
};

template <bool B, class T = void>
struct enable_if_c {
  typedef T type;
};

template <class T>
struct enable_if_c<false, T> {};

template <class Cond, class T = void>
struct enable_if : public enable_if_c<Cond::value, T>{};

template<class F, class Param>
struct apply
{
   typedef typename F::template apply<Param>::type type;
};

#if defined(_MSC_VER) && (_MSC_VER >= 1400)

template <class T, class U>
struct is_convertible
{
   static const bool value = __is_convertible_to(T, U);
};

#else

template <class T, class U>
class is_convertible
{
   typedef char true_t;
   class false_t { char dummy[2]; };
   //use any_conversion as first parameter since in MSVC
   //overaligned types can't go through ellipsis
   static false_t dispatch(...);
   static true_t  dispatch(U);
   static typename remove_reference<T>::type &trigger();
   public:
   static const bool value = sizeof(dispatch(trigger())) == sizeof(true_t);
};

#endif

template<
      bool C
    , typename T1
    , typename T2
    >
struct if_c
{
    typedef T1 type;
};

template<
      typename T1
    , typename T2
    >
struct if_c<false,T1,T2>
{
    typedef T2 type;
};

template<
      typename C
    , typename T1
    , typename T2
    >
struct if_
{
   typedef typename if_c<0 != C::value, T1, T2>::type type;
};

template<
      bool C
    , typename F1
    , typename F2
    >
struct eval_if_c
    : if_c<C,F1,F2>::type
{};

template<
      typename C
    , typename T1
    , typename T2
    >
struct eval_if
    : if_<C,T1,T2>::type
{};

// identity is an extension: it is not part of the standard.
template <class T>
struct identity
{
   typedef T type;
};

template<class T, bool Add>
struct add_const_if_c
{
   typedef typename if_c
      < Add
      , typename add_const<T>::type
      , T
      >::type type;
};


//boost::alignment_of yields to 10K lines of preprocessed code, so we
//need an alternative
template <typename T> struct alignment_of;

template <typename T>
struct alignment_of_hack
{
    char c;
    T t;
    alignment_of_hack();
};

template <unsigned A, unsigned S>
struct alignment_logic
{
   static const std::size_t value = A < S ? A : S;
};

template< typename T >
struct alignment_of
{
   static const std::size_t value = alignment_logic
            < sizeof(alignment_of_hack<T>) - sizeof(T)
            , sizeof(T)
            >::value;
};

template<class Class>
class is_empty_class
{
   template <typename T>
   struct empty_helper_t1 : public T
   {
      empty_helper_t1();
      int i[256];
   };

   struct empty_helper_t2
   { int i[256]; };

   public:
   static const bool value = sizeof(empty_helper_t1<Class>) == sizeof(empty_helper_t2);
};

template<std::size_t S>
struct ls_zeros
{
   static const std::size_t value = (S & std::size_t(1)) ? 0 : (1 + ls_zeros<(S>>1u)>::value);
};

template<>
struct ls_zeros<0>
{
   static const std::size_t value = 0;
};

template<>
struct ls_zeros<1>
{
   static const std::size_t value = 0;
};

template <typename T> struct unvoid_ref { typedef T &type; };
template <> struct unvoid_ref<void> { struct type_impl { }; typedef type_impl & type; };
template <> struct unvoid_ref<const void> { struct type_impl { }; typedef type_impl & type; };

// Infrastructure for providing a default type for T::TNAME if absent.
#define BOOST_INTRUSIVE_INSTANTIATE_DEFAULT_TYPE_TMPLT(TNAME)     \
   template <typename T, typename DefaultType>                    \
   struct boost_intrusive_default_type_ ## TNAME                  \
   {                                                              \
      template <typename X>                                       \
      static char test(int, typename X::TNAME*);                  \
                                                                  \
      template <typename X>                                       \
      static int test(...);                                       \
                                                                  \
      struct DefaultWrap { typedef DefaultType TNAME; };          \
                                                                  \
      static const bool value = (1 == sizeof(test<T>(0, 0)));     \
                                                                  \
      typedef typename                                            \
         ::boost::intrusive::detail::if_c                         \
            <value, T, DefaultWrap>::type::TNAME type;            \
   };                                                             \
   //

#define BOOST_INTRUSIVE_OBTAIN_TYPE_WITH_DEFAULT(INSTANTIATION_NS_PREFIX, T, TNAME, TIMPL)   \
      typename INSTANTIATION_NS_PREFIX                                                       \
         boost_intrusive_default_type_ ## TNAME< T, TIMPL >::type                            \
//

#define BOOST_INTRUSIVE_INSTANTIATE_EVAL_DEFAULT_TYPE_TMPLT(TNAME)\
   template <typename T, typename DefaultType>                    \
   struct boost_intrusive_eval_default_type_ ## TNAME             \
   {                                                              \
      template <typename X>                                       \
      static char test(int, typename X::TNAME*);                  \
                                                                  \
      template <typename X>                                       \
      static int test(...);                                       \
                                                                  \
      struct DefaultWrap                                          \
      { typedef typename DefaultType::type TNAME; };              \
                                                                  \
      static const bool value = (1 == sizeof(test<T>(0, 0)));     \
                                                                  \
      typedef typename                                            \
         ::boost::intrusive::detail::eval_if_c                    \
            < value                                               \
            , ::boost::intrusive::detail::identity<T>             \
            , ::boost::intrusive::detail::identity<DefaultWrap>   \
            >::type::TNAME type;                                  \
   };                                                             \
//

#define BOOST_INTRUSIVE_OBTAIN_TYPE_WITH_EVAL_DEFAULT(INSTANTIATION_NS_PREFIX, T, TNAME, TIMPL) \
      typename INSTANTIATION_NS_PREFIX                                                          \
         boost_intrusive_eval_default_type_ ## TNAME< T, TIMPL >::type                          \
//

#define BOOST_INTRUSIVE_INTERNAL_STATIC_BOOL_IS_TRUE(TRAITS_PREFIX, TYPEDEF_TO_FIND) \
template <class T>\
struct TRAITS_PREFIX##_bool\
{\
   template<bool Add>\
   struct two_or_three {one _[2 + Add];};\
   template <class U> static one test(...);\
   template <class U> static two_or_three<U::TYPEDEF_TO_FIND> test (int);\
   static const std::size_t value = sizeof(test<T>(0));\
};\
\
template <class T>\
struct TRAITS_PREFIX##_bool_is_true\
{\
   static const bool value = TRAITS_PREFIX##_bool<T>::value > sizeof(one)*2;\
};\
//

#define BOOST_INTRUSIVE_HAS_STATIC_MEMBER_FUNC_SIGNATURE(TRAITS_NAME, FUNC_NAME) \
  template <typename U, typename Signature> \
  class TRAITS_NAME \
  { \
  private: \
  template<Signature> struct helper;\
  template<typename T> \
  static ::boost::intrusive::detail::yes_type check(helper<&T::FUNC_NAME>*); \
  template<typename T> static ::boost::intrusive::detail::no_type check(...); \
  public: \
  static const bool value = sizeof(check<U>(0)) == sizeof(::boost::intrusive::detail::yes_type); \
  }; \
//

#define BOOST_INTRUSIVE_HAS_MEMBER_FUNC_CALLED(TRAITS_NAME, FUNC_NAME) \
template <typename Type> \
struct TRAITS_NAME \
{ \
   struct BaseMixin \
   { \
      void FUNC_NAME(); \
   }; \
   struct Base : public Type, public BaseMixin { Base(); }; \
   template <typename T, T t> class Helper{}; \
   template <typename U> \
   static ::boost::intrusive::detail::no_type  check(U*, Helper<void (BaseMixin::*)(), &U::FUNC_NAME>* = 0); \
   static ::boost::intrusive::detail::yes_type check(...); \
   static const bool value = sizeof(::boost::intrusive::detail::yes_type) == sizeof(check((Base*)(0))); \
};\
//

#define BOOST_INTRUSIVE_HAS_MEMBER_FUNC_CALLED_IGNORE_SIGNATURE(TRAITS_NAME, FUNC_NAME) \
BOOST_INTRUSIVE_HAS_MEMBER_FUNC_CALLED(TRAITS_NAME##_ignore_signature, FUNC_NAME) \
\
template <typename Type, class> \
struct TRAITS_NAME \
   : public TRAITS_NAME##_ignore_signature<Type> \
{};\
//


template <typename T>
inline T* addressof(T& obj)
{
   return static_cast<T*>
      (static_cast<void*>
         (const_cast<char*>
            (&reinterpret_cast<const char&>(obj))
         )
      );
}

} //namespace detail
} //namespace intrusive
} //namespace boost

#include <boost/intrusive/detail/config_end.hpp>

#endif //BOOST_INTRUSIVE_DETAIL_MPL_HPP
