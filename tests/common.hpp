#ifndef TEST_COMMON_HPP_
#define TEST_COMMON_HPP_

#include <vector>
#include "atidlas/array.h"

typedef atidlas::int_t int_t;

/*------ Simple Vector ---------*/
template<class T>
class simple_vector_base
{
public:
    typedef T value_type;

    simple_vector_base(int_t start, int_t end, int_t stride, std::vector<T> & data) : N_((end-start)/stride), start_(start),
                                                                                     stride_(stride), data_(data)
    { }
    int_t size() const { return N_; }
    int_t start() const { return start_; }
    int_t stride() const { return stride_; }
    std::vector<T> & data() { return data_; }
    T & operator[](int_t i) { return data_[start_ + i*stride_]; }
    T operator[](int_t i) const { return data_[start_ + i*stride_]; }
private:
    int_t N_;
    int_t start_;
    int_t stride_;
    std::vector<T> & data_;
};

template<class T>
class simple_vector : public simple_vector_base<T>
{
public:
    simple_vector(size_t N) :  simple_vector_base<T>(0, N, 1, data_), data_(N){}
private:
    std::vector<T> data_;
};

template<class T>
class simple_vector_slice : public simple_vector_base<T>
{
public:
    simple_vector_slice(simple_vector_base<T> & x, int_t start, int_t end, int_t stride) :
        simple_vector_base<T>(start, end, stride, x.data()){ }
};

/*------ Simple Matrix ---------*/
template<class T>
class simple_matrix_base
{
public:
    simple_matrix_base(int_t start1, int_t end1, int_t stride1,
                       int_t start2, int_t end2, int_t stride2,
                       int_t ld, std::vector<T> & data) : M_((end1 - start1)/stride1), start1_(start1), stride1_(stride1),
                                                                                         N_((end2-start2)/stride2), start2_(start2), stride2_(stride2),
                                                                                         ld_(ld), data_(data)
    {}

    int_t size1() const { return M_; }
    int_t start1() const { return start1_; }
    int_t stride1() const { return stride1_; }

    int_t size2() const { return N_; }
    int_t start2() const { return start2_; }
    int_t stride2() const { return stride2_; }

    std::vector<T> & data() { return data_; }

    int_t ld() const { return ld_; }

    T operator()(int_t i, int_t j) const
    {
      int_t ii = start1_ + i*stride1_;
      int_t jj = start2_ + j*stride2_;
      return data_[ii + jj*ld_];
    }

    T& operator()(int_t i, int_t j)
    {
      int_t ii = start1_ + i*stride1_;
      int_t jj = start2_ + j*stride2_;
      return data_[ii + jj*ld_];
    }
private:
    int_t M_;
    int_t start1_;
    int_t stride1_;

    int_t N_;
    int_t start2_;
    int_t stride2_;

    int_t ld_;

    std::vector<T> & data_;
};


template<class T>
class simple_matrix : public simple_matrix_base<T>
{
public:
    simple_matrix(int_t M, int_t N) :  simple_matrix_base<T>(0, M, 1, 0, N, 1, M, data_), data_(M*N){}
private:
    std::vector<T> data_;
};

template<class T>
class simple_matrix_slice : public simple_matrix_base<T>
{
public:
    simple_matrix_slice(simple_matrix_base<T> & A,
                        int_t start1, int_t end1, int_t stride1,
                        int_t start2, int_t end2, int_t stride2) :
        simple_matrix_base<T>(start1, end1, stride1, start2, end2, stride2, A.ld(), A.data()){ }
};

/*------ Initializer ---------*/
template<typename T>
void init_rand(simple_vector_base<T> & x)
{
  for (unsigned int i = 0; i < x.size(); ++i)
    x[i] = (T)rand()/RAND_MAX;
}

template<typename T>
void init_rand(simple_matrix_base<T> & A)
{
  for (unsigned int i = 0; i < A.size1(); ++i)
    for(unsigned int j = 0 ; j < A.size2() ; ++j)
      A(i,j) = i+j;
}

template<typename T>
simple_matrix<T> simple_trans(simple_matrix_base<T> const & A)
{
  simple_matrix<T> res(A.size2(), A.size1());
  for(unsigned int i = 0 ; i < A.size1(); ++i)
    for(unsigned int j = 0; j < A.size2(); ++j)
      res(j, i) = A(i, j);
  return res;
}

/*------ Compare -----------*/
template<class VecType1, class VecType2>
bool failure_vector(VecType1 const & x, VecType2 const & y, typename VecType1::value_type epsilon)
{
  typedef typename VecType1::value_type T;
  for(int_t i = 0 ; i < x.size() ; ++i)
  {
    T delta = std::abs(x[i] - y[i]);
    if(std::max(x[i], y[i])!=0)
      delta/=std::abs(std::max(x[i], y[i]));
    if(delta > epsilon)
    {
      return true;
    }
  }
  return false;
}


#define INIT_VECTOR(N, SUBN, START, STRIDE, CPREFIX, PREFIX) \
    simple_vector<T> CPREFIX ## _full(N);\
    simple_vector_slice<T> CPREFIX ## _slice(CPREFIX ## _full, START, START + STRIDE*SUBN, STRIDE);\
    init_rand(CPREFIX ## _full);\
    atidlas::array PREFIX ## _full(CPREFIX ## _full.data());\
    atidlas::array PREFIX ## _slice = PREFIX ## _full[atidlas::_(START, START + STRIDE*SUBN, STRIDE)];

#define INIT_MATRIX(M, SUBM, START1, STRIDE1, N, SUBN, START2, STRIDE2, CPREFIX, PREFIX) \
    simple_matrix<T> CPREFIX ## _full(M, N);\
    simple_matrix_slice<T> CPREFIX ## _slice(CPREFIX ## _full, START1, START1 + STRIDE1*SUBM, STRIDE1,\
                                                                 START2, START2 + STRIDE2*SUBN, STRIDE2);\
    init_rand(CPREFIX ## _full);\
    atidlas::array PREFIX ## _full(M, N, CPREFIX ## _full.data());\
    atidlas::array PREFIX ## _slice(PREFIX ## _full(atidlas::_(START1, START1 + STRIDE1*SUBM, STRIDE1),\
                                                        atidlas::_(START2, START2 + STRIDE2*SUBN, STRIDE2)));\
    simple_matrix<T> CPREFIX ## T_full = simple_trans(CPREFIX ## _full);\
    atidlas::array PREFIX ## T_full(N, M, CPREFIX ## T_full.data());\
    atidlas::array PREFIX ## T_slice(PREFIX ## T_full(atidlas::_(START2, START2 + STRIDE2*SUBN, STRIDE2),\
                                                              atidlas::_(START1, START1 + STRIDE1*SUBM, STRIDE1)));\


#endif
