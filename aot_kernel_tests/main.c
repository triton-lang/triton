#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "add_kernel0.h"

void errMsg(char* msg, int code) {
  if (code != CUDA_SUCCESS) {
   printf("%s with %d\n", msg, code); 
  }
}

#define CHECK_CUDA(code, msg) errMsg(msg, code)

#define GET_DEVICE(id) 

void arange(double *arr, size_t start, size_t size) {

  double val = (double)start;

  for (size_t i = 0; i < size; i++) {
    arr[i] = (double)val;
    val++;
  }
}

#define AllocFloatVec(N) (double *)malloc(N * sizeof(double));
#define cuAllocFloatVec(N,ptr) CHECK_CUDA(cuMemAlloc(ptr, N * sizeof(double)), "Allocate Float");

#define VectoDevice(N, src, dst) CHECK_CUDA(cuMemcpyHtoD(dst, src, N*sizeof(double)), "Copy HtoD"); 

void main() {

  int device_id = 0;
  int max_shared_mem;

  double *u =  AllocFloatVec(100000);
  double *v =  AllocFloatVec(100000);
  arange(u, 0, 100000);
  arange(v, 0, 100000);

  CHECK_CUDA(cuInit(0), "Cuda init failed");

  CUdevice device;
  CHECK_CUDA(cuDeviceGet(&device, device_id), "Get device failed");
  cuDeviceGetAttribute(&max_shared_mem,
                       CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                       device);
  
  printf("Max shared mem: %u \n", max_shared_mem);
  CUcontext ctx;
  cuCtxCreate(&ctx, CU_CTX_SCHED_AUTO, device);
  
  CUdeviceptr u_cu, v_cu, out_cu;
  cuAllocFloatVec(100000, &u_cu);
  cuAllocFloatVec(100000, &v_cu);
  cuAllocFloatVec(100000, &out_cu);

  VectoDevice(100000, u, u_cu);
  VectoDevice(100000, v, v_cu);

  GridWarps g = {32, 0, 0, 3};
  CUstream stream;

  CHECK_CUDA(add_kernel0(stream, g, u_cu, v_cu, out_cu, 100000), "kernel run");
}
