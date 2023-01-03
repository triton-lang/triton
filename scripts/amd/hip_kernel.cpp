#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__global__ void div_kernel(float *in_1, float *in_2, float *out) {
  int i = threadIdx.x;
  out[i] = in_1[i] / in_2[i];
}

int main() {
  // kernel info
#define nBlocks 1
#define nThreads 2

  // vector size
  size_t size = nThreads * sizeof(float);

  // Allocate input vectors h_A and h_B in host memory
  float h_A[nThreads] = {4, 4};
  float h_B[nThreads] = {2, 2};
  float h_C[nThreads] = {};

  // show data
  printf("Input Data\n");
  for (int i = 0; i < nThreads; i++) {
    printf("%f/%f = %f\n", h_A[i], h_B[i], h_C[i]);
  }

  // Allocate vectors in device memory
  float *d_A;
  hipMalloc(&d_A, size);
  float *d_B;
  hipMalloc(&d_B, size);
  float *d_C;
  hipMalloc(&d_C, size);

  // Copy vectors from host memory to device memory
  hipMemcpyHtoD(d_A, h_A, size);
  hipMemcpyHtoD(d_B, h_B, size);

  // launch kernel
  div_kernel<<<nBlocks, nThreads>>>(d_A, d_B, d_C);
  hipDeviceSynchronize(); // wait for kernel before printting

  // check kernel output
  bool pass = true;
  printf("Output Data\n");
  for (int i = 0; i < nThreads; i++) {
    if (d_A[i] / d_B[i] != d_C[i])
      pass = false;
    printf("%f/%f = %f\n", d_A[i], d_B[i], d_C[i]);
  }
  printf("Test %s\n", pass ? "PASS" : "FAIL");
}