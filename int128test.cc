#include <iostream>

#include "int128test.h"
#include "cucheck.h"

#ifndef __CUDACC__

int main() {
  return Int128Test::test();
}

#else

__global__ void testKernel() {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    Int128Test::test();
  }
}

int main() {
  printf("running test kernel\n");
  testKernel<<<1, 1>>>();
  CUCHECK(cudaGetLastError());
  CUCHECK(cudaDeviceReset());
}
#endif

