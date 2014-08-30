#ifndef TEST_COMMON_HPP_
#define TEST_COMMON_HPP_

#include "vector"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "atidlas/forwards.h"

typedef atidlas::atidlas_int_t  int_t;

/*---------
 * Vector
 * -------*/

template<class NumericT>
class simple_vector
{
public:
    typedef NumericT value_type;
    typedef size_t size_type;

    simple_vector(size_t N) : N_(N), data_(N){ }
    size_t size() const { return N_; }
    value_type & operator[](size_t i) { return data_[i]; }
    value_type operator[](size_t i) const { return data_[i]; }
    size_t internal_size() const { return data_.size(); }
    typename std::vector<value_type>::iterator begin() { return data_.begin(); }
    typename std::vector<value_type>::iterator end() { return data_.begin() + size(); }
    typename std::vector<value_type>::const_iterator begin() const { return data_.begin(); }
    typename std::vector<value_type>::const_iterator end() const { return data_.begin() + size(); }
private:
    size_t N_;
    std::vector<value_type> data_;
};

template<class T>
class simple_vector_range
{
public:
    typedef typename T::value_type value_type;
    typedef typename T::size_type size_type;

    simple_vector_range(simple_vector<value_type> & data, viennacl::range const & r) : data_(data), r_(r) { }
    size_t size() const { return r_.size(); }
    viennacl::range const & range() const { return r_; }
    value_type & operator[](size_t i) { return data_[i]; }
    value_type operator[](size_t i) const { return data_[i]; }
    size_t internal_size() const { return data_.size(); }
    typename std::vector<value_type>::iterator begin() { return data_.begin(); }
    typename std::vector<value_type>::iterator end() { return data_.begin() + size(); }
    typename std::vector<value_type>::const_iterator begin() const { return data_.begin(); }
    typename std::vector<value_type>::const_iterator end() const { return data_.begin() + size(); }
private:
    simple_vector<value_type> & data_;
    viennacl::range r_;
};

template<class T>
class simple_vector_slice
{
public:
    typedef typename T::value_type value_type;
    typedef typename T::size_type size_type;

    simple_vector_slice(simple_vector<value_type> & data, viennacl::slice const & s) : data_(data), s_(s) { }
    size_type size() const { return s_.size(); }
    size_t internal_size() const { return data_.size(); }
    viennacl::slice const & slice() const { return s_; }
    value_type & operator[](size_t i) { return data_[i]; }
    value_type operator[](size_t i) const { return data_[i]; }
private:
    simple_vector<value_type> & data_;
    viennacl::slice s_;
};


//Helper to initialize a viennacl vector from a simple type

template<class SimpleType>
struct vector_maker;

template<class NumericT>
struct vector_maker< simple_vector<NumericT> >
{
  typedef viennacl::vector<NumericT> result_type;
  static result_type make(viennacl::vector<NumericT> const &, simple_vector<NumericT> & base)
  {
    viennacl::vector<NumericT> result(base.size());
    viennacl::copy(base, result);
    return result;
  }
};

template<class NumericT>
struct vector_maker< simple_vector_range< simple_vector<NumericT> > >
{
  typedef viennacl::vector_range< viennacl::vector<NumericT> > result_type;
  static result_type make(viennacl::vector<NumericT> & x, simple_vector_range< simple_vector<NumericT> > & base)
  {
    result_type result(x, base.range());
    viennacl::copy(base, result);
    return result;
  }
};

template<class NumericT>
struct vector_maker< simple_vector_slice<simple_vector<NumericT> > >
{
  typedef viennacl::vector_slice< viennacl::vector<NumericT> > result_type;
  static result_type make(viennacl::vector<NumericT> & M, simple_vector_slice< simple_vector<NumericT> > & base)
  {
    result_type result(M, base.slice());
    viennacl::copy(base, result);
    return result;
  }
};


/*---------
 * Matrix
 * -------*/

template<class NumericT>
class simple_matrix
{
public:
    typedef NumericT value_type;
    typedef size_t size_type;

    simple_matrix(size_t M, size_t N) : data_(M*N), M_(M), N_(N){ }
    size_t size1() const { return M_; }
    size_t size2() const { return N_; }
    value_type & operator()(size_t i, size_t j) { return data_[i + M_*j]; }
    value_type operator()(size_t i, size_t j) const { return data_[i + M_*j]; }
private:
    std::vector<value_type> data_;
    size_t M_;
    size_t N_;
};

template<class T>
class simple_matrix_range
{
public:
    typedef typename T::value_type value_type;

    simple_matrix_range(T & A, viennacl::range const & r1, viennacl::range const & r2) : A_(A), r1_(r1), r2_(r2){ }
    size_t size1() const { return r1_.size(); }
    size_t size2() const { return r2_.size(); }
    viennacl::range const & r1() const { return r1_; }
    viennacl::range const & r2() const { return r2_; }
    value_type & operator()(size_t i, size_t j) { return A_(i+r1_.start(), j+r2_.start()); }
    value_type operator()(size_t i, size_t j) const { return A_(i+r1_.start(), j+r2_.start()); }
private:
    T & A_;
    viennacl::range r1_;
    viennacl::range r2_;
};

template<class T>
class simple_matrix_slice
{
public:
    typedef typename T::value_type value_type;

    simple_matrix_slice(T & A,  viennacl::slice const & s1,  viennacl::slice const & s2) : A_(A), s1_(s1), s2_(s2){ }
    viennacl::slice::size_type size1() const { return s1_.size(); }
    viennacl::slice::size_type size2() const { return s2_.size(); }
    viennacl::slice const & s1() const { return s1_; }
    viennacl::slice const & s2() const { return s2_; }
    value_type & operator()(size_t i, size_t j) { return A_(i*s1_.stride() + s1_.start(), j*s2_.stride() + s2_.start()); }
    value_type operator()(size_t i, size_t j) const { return A_(i+s1_.start(), j+s2_.start()); }
private:
    T & A_;
    viennacl::slice s1_;
    viennacl::slice s2_;
};


/*-------
 * Helpers
 *-------*/

template<typename T>
void init_rand(simple_matrix<T> & A)
{
  for (unsigned int i = 0; i < A.size1(); ++i)
    for (unsigned int j = 0; j < A.size2(); ++j)
      A(i, j) = T(0.1) * rand()/RAND_MAX;
}


template<typename T>
void init_rand(simple_vector<T> & x)
{
  for (unsigned int i = 0; i < x.size(); ++i)
    x[i] = T(0.1) * rand()/RAND_MAX;
}

template<class T>
simple_matrix<T> simple_trans(simple_matrix<T> const & A)
{
  int M = A.size1();
  int N = A.size2();
  simple_matrix<T> result(N, M);

  for(int i = 0; i < N; ++i)
    for(int j = 0; j < M; ++j)
      result(i,j) = A(j,i);

  return result;
}

template<class T, class U, class V>
simple_matrix<T> simple_prod(U const & A, V const & B)
{
    int M = A.size1();
    int N = B.size2();
    int K = A.size2();
    simple_matrix<T> result(M, N);

    for(int i = 0 ; i < M ; ++i)
      for(int j = 0 ; j < N ; ++j)
      {
        T val = 0;
        for(int k = 0 ; k < K ; ++k)
          val+= A(i, k)*B(k,j);
        result(i, j) = val;
      }

    return result;
}

template<class SimpleType, class F>
struct matrix_maker;

template<class T, class F>
struct matrix_maker< simple_matrix<T>, F>
{
  typedef viennacl::matrix<T, F> result_type;
  static result_type make(viennacl::matrix<T, F> const &, simple_matrix<T> & base)
  {
    viennacl::matrix<T, F> result(base.size1(), base.size2());
    viennacl::copy(base, result);
    return result;
  }
};

template<class MatrixT, class F>
struct matrix_maker< simple_matrix_range<MatrixT>, F>
{
  typedef typename MatrixT::value_type T;
  typedef viennacl::matrix_range< viennacl::matrix<T, F> > result_type;

  static result_type make(viennacl::matrix<T, F> & M, simple_matrix_range<MatrixT> & base)
  {
    result_type result(M, base.r1(), base.r2());
    viennacl::copy(base, result);
    return result;
  }
};

template<class MatrixT, class F>
struct matrix_maker< simple_matrix_slice<MatrixT>, F>
{
  typedef typename MatrixT::value_type T;
  typedef viennacl::matrix_slice< viennacl::matrix<T, F> > result_type;

  static result_type make(viennacl::matrix<T, F> & M, simple_matrix_slice<MatrixT> & base)
  {
    result_type result(M, base.s1(), base.s2());
    viennacl::copy(base, result);
    return result;
  }
};

template<class VectorType>
bool failure_vector(VectorType const & x, VectorType const & y, typename VectorType::value_type epsilon)
{
  typedef typename VectorType::value_type value_type;
  for(int_t i = 0 ; i < x.size() ; ++i)
  {
    value_type delta = std::abs(x[i] - y[i]);
    if(delta > epsilon)
      return true;
  }
  return false;
}

template<class NumericT>
bool failure(simple_matrix<NumericT> const & A, simple_matrix<NumericT> const & B, NumericT epsilon)
{
  int M = A.size1();
  int N = A.size2();
  for(int i = 0 ; i < M ; ++i)
    for(int j = 0 ; j < N ; ++j)
    {
      NumericT delta = std::abs(A(i,j) - B(i,j));
      if(delta > epsilon)
        return true;
    }
  return false;
}

namespace viennacl
{
namespace traits
{
template<class T> int size1(simple_matrix<T> const & M) { return M.size1(); }
template<class T> int size2(simple_matrix<T> const & M) { return M.size2(); }
}
}

#endif
