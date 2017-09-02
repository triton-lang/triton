/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef ISAAC_TOOLS_MATRIX_HPP_
#define ISAAC_TOOLS_MATRIX_HPP_

#include <cstddef>

inline void read_inc(void* dst, u_char*& data, size_t nbytes){
  std::memcpy(dst, (void*)data, nbytes);
  data += nbytes;
}

template<class T>
void gemm(uint32_t M, uint32_t N, uint32_t K, T alpha, T* A, uint32_t lda, T* B, uint32_t ldb, T, T* C, uint32_t ldc, T* bias){
  for(uint32_t i = 0; i < M ; ++i)
    for(uint32_t j = 0; j < N ; ++j){
      T acc = 0;
      for(uint32_t k = 0; k < K; ++k)
        acc += A[i*lda + k] * B[k*ldb + j];
      C[i*ldc + j] = alpha*acc + bias[j];
    }
}

template<class T>
class matrix{
  typedef std::array<uint32_t, 2> shapes_t;

public:
  matrix(u_char*& data){
    read_inc((void*)shapes_.data(), data, 8);
    values_.resize(shapes_[0]*shapes_[1]);
    ld_ = shapes_[1];
    read_inc((void*)values_.data(), data, values_.size()*4);
    data_ = values_.data();
  }
  matrix(shapes_t const & shapes, size_t ld, T* data): shapes_(shapes), ld_(ld), data_(data){}
  matrix(shapes_t const & shapes): shapes_(shapes), ld_(shapes.back()), values_(shapes[0]*shapes[1]), data_(values_.data()){}

  shapes_t const & shapes() const
  { return shapes_; }

  T const & operator()(size_t i, size_t j) const
  { return data_[i*ld_ + j]; }
  T & operator ()(size_t i, size_t j)
  { return data_[i*ld_ + j]; }

  T* data() const
  { return data_; }
  T* data()
  { return data_; }

  uint32_t ld() const
  { return ld_; }

private:
  shapes_t shapes_;
  size_t ld_;
  std::vector<T> values_;
  T* data_;
};

template<class T>
matrix<T> pad_left(matrix<T> const & in, uint32_t npad){
  uint32_t M = in.shapes()[0], N = in.shapes()[1];
  matrix<T> result({M, N + npad});
  for(size_t i = 0; i < M; ++i)
    for(size_t j = 0; j < N; ++j)
      result(i,  npad + j) = in(i, j);
  return result;
}

#endif
