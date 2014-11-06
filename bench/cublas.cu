#include <ctime>
#include <cstdio>
#include <common_functions.h>
#include <cublas.h>
#include <vector>
#include <algorithm>
#include <iostream>

#include "common.hpp"

template<class NumericT>
void run()
{
    #define FILL_TIMINGS(OP, timings) \
    {\
      float elapsedTime = 0; \
      float total_time = 0; \
      OP;\
      while(total_time < 1e-1) \
      { \
        cudaEvent_t start, stop; \
        cudaEventCreate(&start); \
        cudaEventRecord(start,0); \
        OP; \
        cudaThreadSynchronize(); \
        cudaEventCreate(&stop); \
        cudaEventRecord(stop,0); \
        cudaEventSynchronize(stop); \
        cudaEventElapsedTime(&elapsedTime, start,stop); \
        timings.push_back(elapsedTime/1e3); \
        total_time += elapsedTime/1e3; \
      }\
    }

    //AXPY
    std::cout << "#vector-axpy" << std::endl;
    std::cout << "#N Perf" << std::endl;
    for(std::vector<int>::const_iterator it = BLAS1_N.begin() ; it != BLAS1_N.end() ; ++it)
    {
      int N = *it;
      NumericT *x, *y;
      cudaMalloc((void**) &x, N * sizeof(NumericT));
      cudaMalloc((void**) &y, N * sizeof(NumericT));
      //Bench
      std::vector<float> timings;
      FILL_TIMINGS(cublasSaxpy(N, 2, x, 1, y, 1), timings);
      std::cout << N << " " << 3*N*sizeof(NumericT)*1e-9/median(timings) << std::endl;
      //Free
      cudaFree(x);
      cudaFree(y);
    }
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "#dot" << std::endl;
    std::cout << "#N Perf" << std::endl;
    for(std::vector<int>::const_iterator it = BLAS1_N.begin() ; it != BLAS1_N.end() ; ++it)
    {
      int N = *it;
      NumericT *x, *y;
      cudaMalloc((void**) &x, N * sizeof(NumericT));
      cudaMalloc((void**) &y, N * sizeof(NumericT));
      //Bench
      std::vector<float> timings;
      FILL_TIMINGS(cublasSdot(N, x, 1, y, 1), timings);
      std::cout << N << " " << 2*N*sizeof(NumericT)*1e-9/median(timings) << std::endl;
      //Free
      cudaFree(x);
      cudaFree(y);
    }
    std::cout << std::endl;
    std::cout << std::endl;


    std::cout << "#GEMV" << std::endl;
    std::cout << "#N Perf" << std::endl;
    for(std::vector<int>::const_iterator Mit = BLAS2_M.begin() ; Mit != BLAS2_M.end() ; ++Mit)
    {
        for(std::vector<int>::const_iterator it = BLAS2_N.begin() ; it != BLAS2_N.end() ; ++it)
        {
          int M = *Mit;
          int N = *it;
          NumericT *x, *y, *A;
          cudaMalloc((void**) &A, M * N * sizeof(NumericT));
          cudaMalloc((void**) &x, M * sizeof(NumericT));
          cudaMalloc((void**) &y, N * sizeof(NumericT));
          //Bench
          std::vector<float> timings;
          FILL_TIMINGS(cublasSgemv('N', M, N, 1.0, A, M, x, 1, 1.0, y, 1), timings);
          std::cout << N << " " << (M + N + M*N)*sizeof(NumericT)*1e-9/median(timings) << std::endl;
          //Free
          cudaFree(A);
          cudaFree(x);
          cudaFree(y);
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }

    std::cout << "#GEMM" << std::endl;
    std::cout << "#N Perf" << std::endl;
    for(std::vector<int>::const_iterator it = BLAS3_N.begin() ; it != BLAS3_N.end() ; ++it)
    {
      int M = *it;
      int N = *it;
      int K = *it;
      NumericT *C, *A, *B;
      cudaMalloc((void**) &A, M * K * sizeof(NumericT));
      cudaMalloc((void**) &B, K * N * sizeof(NumericT));
      cudaMalloc((void**) &C, M * N * sizeof(NumericT));
      //Bench
      std::vector<float> timings;
      FILL_TIMINGS(cublasSgemm('N','T',M,N,K,1.0,A,M,B,K,1.0,C,M), timings);
      std::cout << N << " " << 2.0*M*N*K*1e-9/median(timings) << std::endl;
      //Free
      cudaFree(A);
      cudaFree(B);
      cudaFree(C);
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

int main(int argc, char** argv)
{
  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Device: " << prop.name << std::endl;
  run<float>();
}
