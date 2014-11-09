#ifndef ATIDLAS_FORWARDS_H
#define ATIDLAS_FORWARDS_H

#include "CL/cl.hpp"

namespace atidlas
{

typedef int atidlas_int_t;

enum numeric_type
{
  INVALID_NUMERIC_TYPE = 0,
  CHAR_TYPE,
  UCHAR_TYPE,
  SHORT_TYPE,
  USHORT_TYPE,
  INT_TYPE,
  UINT_TYPE,
  LONG_TYPE,
  ULONG_TYPE,
  HALF_TYPE,
  FLOAT_TYPE,
  DOUBLE_TYPE
};

inline unsigned int size_of(numeric_type type)
{
  switch (type)
  {
  case UCHAR_TYPE:
  case CHAR_TYPE: return 1;

  case USHORT_TYPE:
  case SHORT_TYPE:
  case HALF_TYPE: return 2;

  case UINT_TYPE:
  case INT_TYPE:
  case FLOAT_TYPE: return 4;

  case ULONG_TYPE:
  case LONG_TYPE:
  case DOUBLE_TYPE: return 8;

  default: throw "Unsupported numeric type";
  }
}

template<class LHS, class RHS, class OP>
class vector_expression;

class vector_base
{
public:
    vector_base(atidlas_int_t size, numeric_type dtype, cl::Context context) : dtype_(dtype), size_(size), internal_size_(size), start_(0), stride_(1), context_(context), data_(context, CL_MEM_READ_WRITE, size_of(dtype_)*internal_size()) {}
    vector_base(cl::Buffer data, atidlas_int_t size, numeric_type dtype, atidlas_int_t start, atidlas_int_t stride): dtype_(dtype), size_(size), internal_size_(size), start_(start), stride_(stride), context_(data.getInfo<CL_MEM_CONTEXT>()), data_(data){ }

    numeric_type dtype() const { return dtype_; }
    atidlas_int_t size() const { return size_; }
    atidlas_int_t internal_size() const { return internal_size_; }
    atidlas_int_t start() const { return start_; }
    atidlas_int_t stride() const { return stride_; }

    template<class LHS, class RHS, class OP>
    vector_base& operator=(vector_expression<LHS, RHS, OP> const &);

    cl::Context const & context() const { return context_; }
    cl::Buffer const & data() const { return data_; }


private:
    numeric_type dtype_;

    atidlas_int_t size_;
    atidlas_int_t internal_size_;
    atidlas_int_t start_;
    atidlas_int_t stride_;

    cl::Context context_;
    cl::Buffer data_;
};

class matrix_base
{
public:
    matrix_base(atidlas_int_t size1, atidlas_int_t size2, numeric_type dtype, cl::Context context) : dtype_(dtype), size1_(size1), internal_size1_(size1), start1_(0), stride1_(1),
                                                                                size2_(size2), internal_size2_(size2), start2_(0), stride2_(2),
                                                                                context_(context), data_(context, CL_MEM_READ_WRITE, size_of(dtype_)*internal_size()) {}
    matrix_base(cl::Buffer data, atidlas_int_t size1, atidlas_int_t start1, atidlas_int_t stride1,
                                 atidlas_int_t size2, atidlas_int_t start2, atidlas_int_t stride2,
                                 numeric_type dtype): dtype_(dtype), size1_(size1), start1_(start1), stride1_(stride1),
                                                       size2_(size2), start2_(start2), stride2_(stride2), context_(data.getInfo<CL_MEM_CONTEXT>()), data_(data){ }

    numeric_type dtype() const { return dtype_; }

    atidlas_int_t size1() const { return size1_; }
    atidlas_int_t internal_size1() const { return size1_; }
    atidlas_int_t start1() const { return start1_; }
    atidlas_int_t stride1() const { return stride1_; }

    atidlas_int_t size2() const { return size2_; }
    atidlas_int_t internal_size2() const { return size2_; }
    atidlas_int_t start2() const { return start2_; }
    atidlas_int_t stride2() const { return stride2_; }

    atidlas_int_t internal_size() const { return internal_size1_*internal_size2_; }

    cl::Context const & context() const { return context_; }
    cl::Buffer const & data() const { return data_; }


private:
    numeric_type dtype_;

    atidlas_int_t size1_;
    atidlas_int_t internal_size1_;
    atidlas_int_t start1_;
    atidlas_int_t stride1_;

    atidlas_int_t size2_;
    atidlas_int_t internal_size2_;
    atidlas_int_t start2_;
    atidlas_int_t stride2_;

    cl::Context context_;
    cl::Buffer data_;
};

/** @brief A tag class representing assignment */
struct op_assign {};
/** @brief A tag class representing inplace addition */
struct op_inplace_add {};
/** @brief A tag class representing inplace subtraction */
struct op_inplace_sub {};

/** @brief A tag class representing addition */
struct op_add {};
/** @brief A tag class representing subtraction */
struct op_sub {};
/** @brief A tag class representing multiplication by a scalar */
struct op_mult {};
/** @brief A tag class representing matrix-vector products and element-wise multiplications*/
struct op_prod {};
/** @brief A tag class representing matrix-matrix products */
struct op_mat_mat_prod {};
/** @brief A tag class representing division */
struct op_div {};
/** @brief A tag class representing the power function */
struct op_pow {};

/** @brief A tag class representing equality */
struct op_eq {};
/** @brief A tag class representing inequality */
struct op_neq {};
/** @brief A tag class representing greater-than */
struct op_greater {};
/** @brief A tag class representing less-than */
struct op_less {};
/** @brief A tag class representing greater-than-or-equal-to */
struct op_geq {};
/** @brief A tag class representing less-than-or-equal-to */
struct op_leq {};

template<class T>
struct op_reduce_vector{ };

template<class T>
struct op_reduce_rows{ };

template<class T>
struct op_reduce_columns{ };

/** @brief A tag class representing element-wise casting operations on vectors and matrices */
template<typename OP>
struct op_element_cast {};

/** @brief A tag class representing element-wise binary operations (like multiplication) on vectors or matrices */
template<typename OP>
struct op_element_binary {};

/** @brief A tag class representing element-wise unary operations (like sin()) on vectors or matrices */
template<typename OP>
struct op_element_unary {};

/** @brief A tag class representing the modulus function for integers */
struct op_abs {};
/** @brief A tag class representing the acos() function */
struct op_acos {};
/** @brief A tag class representing the asin() function */
struct op_asin {};
/** @brief A tag class for representing the argmax() function */
struct op_argmax {};
/** @brief A tag class for representing the argmin() function */
struct op_argmin {};
/** @brief A tag class representing the atan() function */
struct op_atan {};
/** @brief A tag class representing the atan2() function */
struct op_atan2 {};
/** @brief A tag class representing the ceil() function */
struct op_ceil {};
/** @brief A tag class representing the cos() function */
struct op_cos {};
/** @brief A tag class representing the cosh() function */
struct op_cosh {};
/** @brief A tag class representing the exp() function */
struct op_exp {};
/** @brief A tag class representing the fabs() function */
struct op_fabs {};
/** @brief A tag class representing the fdim() function */
struct op_fdim {};
/** @brief A tag class representing the floor() function */
struct op_floor {};
/** @brief A tag class representing the fmax() function */
struct op_fmax {};
/** @brief A tag class representing the fmin() function */
struct op_fmin {};
/** @brief A tag class representing the fmod() function */
struct op_fmod {};
/** @brief A tag class representing the log() function */
struct op_log {};
/** @brief A tag class representing the log10() function */
struct op_log10 {};
/** @brief A tag class representing the sin() function */
struct op_sin {};
/** @brief A tag class representing the sinh() function */
struct op_sinh {};
/** @brief A tag class representing the sqrt() function */
struct op_sqrt {};
/** @brief A tag class representing the tan() function */
struct op_tan {};
/** @brief A tag class representing the tanh() function */
struct op_tanh {};

/** @brief A tag class representing the (off-)diagonal of a matrix */
struct op_matrix_diag {};

/** @brief A tag class representing a matrix given by a vector placed on a certain (off-)diagonal */
struct op_vector_diag {};

/** @brief A tag class representing the extraction of a matrix row to a vector */
struct op_row {};

/** @brief A tag class representing the extraction of a matrix column to a vector */
struct op_column {};

/** @brief A tag class representing inner products of two vectors */
struct op_inner_prod {};

/** @brief A tag class representing the 1-norm of a vector */
struct op_norm_1 {};

/** @brief A tag class representing the 2-norm of a vector */
struct op_norm_2 {};

/** @brief A tag class representing the inf-norm of a vector */
struct op_norm_inf {};

/** @brief A tag class representing the maximum of a vector */
struct op_max {};

/** @brief A tag class representing the minimum of a vector */
struct op_min {};


/** @brief A tag class representing the Frobenius-norm of a matrix */
struct op_norm_frobenius {};

/** @brief A tag class representing transposed matrices */
struct op_trans {};

/** @brief A tag class representing sign flips (for scalars only. Vectors and matrices use the standard multiplication by the scalar -1.0) */
struct op_flip_sign {};

template<typename LHS, typename RHS, typename OP>
class vector_expression;

template<typename LHS, typename RHS, typename OP>
class matrix_expression;

template<typename LHS, typename RHS, typename OP>
class scalar_expression;

}
#endif
