#ifndef __CUDAFREE_H__
#define __CUDAFREE_H__

/* make short macros to replace __host__ and __device__ tags */

#ifdef __CUDACC__
#include "cuda.h"
#define HD __host__ __device__
#define DV __device__
#else
#include <assert.h>
#define HD
#define DV
#endif

#endif // __CUDAFREE_H__
