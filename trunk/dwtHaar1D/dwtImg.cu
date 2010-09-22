// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil_inline.h>

#include "dwtImg.h"

#define HANDLE_ERROR(X, MSG) do { if (cudaSuccess != X) \
 {printf("cuda fejl: %s\n i %s\n", cudaGetErrorString(cudaGetLastError()), MSG); \
   exit(42);} } while (0)



 /* something quick and dirty */
void test_img(const char* file) {
  unsigned char *data;
  unsigned int img_w, img_h;
  cutLoadPPM4ub(file, &data, &img_w, &img_h);


  // allocate device mem

  const unsigned int smem_size = sizeof(int) * img_w * img_h;
  int *d_idata, *d_odata;
  d_odata = d_idata = NULL;
  cutilSafeCall( cudaMalloc( (void**) &d_idata, smem_size));
  cutilSafeCall( cudaMalloc( (void**) &d_odata, smem_size));

  // copy input data to device
  cutilSafeCall( cudaMemcpy( d_idata, data, smem_size, cudaMemcpyHostToDevice) );
  cutilSafeCall( cudaMemcpy( d_odata, data, smem_size, cudaMemcpyHostToDevice) ); 
  
  

  /* here you could do nice stuff like compressing or decompressing... */
  dim3 grid, block;
  block.x = 512;
  grid.x = (img_h *img_w)/block.x;

  simple_copy_kernel<<< grid, block>>>(d_idata, d_odata);

  HANDLE_ERROR(cudaPeekAtLastError(), "afterLaunch and not syncronized");


  /* copy result back */
  cutilSafeCall( cudaMemcpy(data, d_odata, smem_size, cudaMemcpyDeviceToHost) ); 

  /* data = (unsigned char*) back; */
  /* write result out */
  cutSavePPM4ub("output.ppm", data, img_w, img_h);

}


__global__
void simple_copy_kernel(int *in, int *out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = in[(tid%1025)];
}
