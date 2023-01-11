#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>


#define CHECK_CUDA(code, msg) { if ((code) != 0) { printf("%s\n", (msg)); } };

void arange(double* arr, size_t start, size_t size) {

  double val = (double)start;

  for (size_t i =0; i < size; i++ ) {
    arr[i] = (double)val;
    val++;
  } 

}

int main() {
    
    const char* msg = "hello";
    CHECK_CUDA(0, "good");
    CHECK_CUDA(1, "bad");


    double *arr = (double *)malloc((300) * sizeof(double));
    arange(arr, 17, 90);
    printf("test: %f", arr[18]);

}