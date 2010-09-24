// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil_inline.h>

#include "dwtImg.h"
#include "dwtHaar1D.h"
#include "dwtHaar1D_kernel.h"



#define HANDLE_ERROR(X, MSG) do { if (cudaSuccess != X)			\
      {printf("cuda fejl: %s\n i %s\n", cudaGetErrorString(cudaGetLastError()), MSG); \
	exit(42);} } while (0)



__device__
void decompositionStep(float *C, float *out, int half_step) {
  unsigned int id = threadIdx.x;
  int offset;
  float data0; 
  float data1;
  
  
  if (id < half_step) {
    offset = blockIdx.y * 1024 ;
    data0 = C[offset + 2 * id]; 
    data1 = C[offset + 2 * id + 1];
  }
  __syncthreads(); 
  
  if (id < half_step) {
    out[offset + id] =             (data0 + data1) * INV_SQRT_2;
    out[offset + id + half_step] = (data0 - data1) * INV_SQRT_2;
  }
  __syncthreads(); 
}



__device__
void recomposeStep(float *D, float *out, int half_step, int level) {
  unsigned int id = threadIdx.x;
  if (id < half_step) {
    int offset = blockIdx.y * 1024;
    float coarse = D[offset + id]; 
    float detail = D[offset + id + half_step];
    out[offset + 2 * id] = (coarse + detail )*INV_SQRT_2;
    out[offset + 2 * id + 1] = (coarse - detail )*INV_SQRT_2;
  } 

  __syncthreads();

}


__global__
void recompose(float *D, float * out, int levels) {
  unsigned int half_step = 2 * blockDim.x>>levels;
  unsigned int id = threadIdx.x;
  int offset = blockIdx.y * 1024 ;
  for (int i=0; i < levels; i++) {
    recomposeStep(D, out, half_step, i);
    half_step <<= 1;
    D = out;    
  }
}
  



__global__
void decomposition(float *C, float * out, int levels) {
  unsigned int id = threadIdx.x;
  unsigned int half_step = blockDim.x;
  int offset = blockIdx.y * 1024 ;
  /* normalize dataset */
  C[offset + id] *= INV_SQRT_2;
  C[offset + id + half_step] *= INV_SQRT_2;
  
  __syncthreads();
  
  for (int i=0; i < levels; i++) {
    decompositionStep(C, out, half_step);
    half_step >>= 1; 		/* div 2 */
    C = out;
  }
}


/* something quick and dirty */
void test_img(const char* file, int levels) {
 
  /*   showIds(2, 4); */

  /*   return; */


  unsigned char *data;
  unsigned int img_w, img_h;
  cutLoadPPM4ub(file, &data, &img_w, &img_h);
  printf("img_w= %d, img_h= %d\n", img_w, img_h);


  // allocate device mem

  const unsigned int smem_size = sizeof(int) * img_w * img_h;
  const unsigned int img_smem_size = sizeof(float) * img_w * img_h;
  const unsigned int slength =  img_w * img_h;

  printf("memory use sizeof(float) = %d, sizeof(int) = %d\n", sizeof(float), sizeof(int));

  int *int_image;
  float *img_data;
  int_image = NULL;

  cutilSafeCall( cudaMalloc( (void**) &int_image, smem_size));
  cutilSafeCall( cudaMalloc( (void**) &img_data, img_smem_size));

  // copy input data to device
  cutilSafeCall( cudaMemcpy( int_image, data, smem_size, cudaMemcpyHostToDevice) );


  // device out data
  float* d_odata = NULL;

  cutilSafeCall( cudaMalloc( (void**) &d_odata, img_smem_size));

  
  // clear result memory
  float* tmp = (float*) malloc( img_smem_size);
  for( unsigned int i = 0; i < slength; ++i) {
    tmp[i] = 0.0;
  }
  cutilSafeCall( cudaMemcpy( d_odata, tmp, smem_size, 
			     cudaMemcpyHostToDevice) ); 
  free( tmp);



  //to grayscale
  dim3 grid, block;
  block.x = 512;
  grid.x = (img_h *img_w)/block.x;

  to_grayscale_floats<<< grid, block>>>(int_image, img_data, img_w * img_h);
  from_grayscale_floats_to_ppm("original.ppm", img_data, img_w, img_h);


  // 2D signal so the arrangement of elements is also 2D
  dim3  block_size;
  dim3  grid_size;  

  block_size.x = 512;
  grid_size.x = 1; 		/* one row */
  grid_size.y = 1024;		/* corresponding to cols */

  // run kernel
  decomposition<<<grid_size, block_size >>>( img_data, d_odata, levels);  


  //from grayscale
  from_grayscale_floats_to_ppm("output.ppm", d_odata, img_w, img_h);

  /* recompose */
  cutilSafeCall( cudaMemcpy(img_data, d_odata, smem_size, cudaMemcpyDeviceToDevice) );
  recompose<<<grid_size, block_size >>>( img_data, d_odata, levels);  

  //from grayscale
  from_grayscale_floats_to_ppm("recomposed.ppm", d_odata, img_w, img_h);





}




__global__
void to_grayscale_floats(int *in, float *out, int size) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size)
    return;

  int pixel = in[tid];
  unsigned char r = (pixel >> 16) & 0xFF;
  unsigned char g = (pixel >> 8) & 0xFF;
  unsigned char b = pixel & 0xFF;

  out[tid] = (r + g + b)/3.0f;

}


__global__
void from_grayscale_floats(float *in, int* out, int size) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size)
    return;
  float p = in[tid];
  p = max(min(255.0f, p), 0.0f);
  unsigned char val = (unsigned char) p;
  int pixel = val << 16 | val << 8 | val;
  out[tid] = pixel;

}



__global__
void simple_copy_kernel(int *in, int *out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = in[(tid%1025)];
}


void from_grayscale_floats_to_ppm(const char *filename, float *d_odata, int img_w, int img_h) {

  int img_size = (img_h *img_w);
  int mem_size = img_size*sizeof(int);
  dim3 block, grid;

  block.x = 512;
  grid.x = img_size/block.x;
 


  int *int_image, *data;
  cutilSafeCall( cudaMalloc( (void**) &int_image, mem_size));
  data = (int*) malloc(sizeof(int) * img_size);

  from_grayscale_floats<<< grid, block>>>(d_odata, int_image ,img_w * img_h);
  HANDLE_ERROR(cudaPeekAtLastError(), "after from grayscale");


  /* copy result back */
  cutilSafeCall( cudaMemcpy(data, int_image, mem_size , cudaMemcpyDeviceToHost) );

  /* write result out */
  cutSavePPM4ub(filename, (unsigned char*) data, img_w, img_h);

  /* free stuff */
  free(data);
  cutilSafeCall( cudaFree(int_image));

}

__global__ void 
dwtHaar2D( float* id, float* od, float* approx_final, 
	   const unsigned int dlevels,
	   const unsigned int slength_step_half,
	   const int bdim ) 
{ 
  // shared memory for part of the signal
  extern __shared__ float shared[];  

  // thread runtime environment, 1D parametrization
  const int gdim = gridDim.x;
  // const int bdim = blockDim.x;
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;  

  const int col_offset = 1024 * blockIdx.y;


  // global thread id (w.r.t. to total data set)
  const int tid_global = (bid * bdim) + tid;    
  unsigned int idata = (bid * (2 * bdim)) + tid;

  // read data from global memory
  shared[tid] = id[idata +col_offset];
  shared[tid + bdim] = id[idata + bdim + col_offset];
  __syncthreads();

  // this operation has a two way bank conflicts for all threads, this are two
  // additional cycles for each warp -- all alternatives to avoid this bank
  // conflict are more expensive than the one cycle introduced by serialization
  float data0 = shared[2*tid];
  float data1 = shared[(2*tid) + 1];
  __syncthreads();

  // detail coefficient, not further referenced so directly store in
  // global memory
  od[tid_global + slength_step_half + col_offset] = (data0 - data1) * INV_SQRT_2;

  // offset to avoid bank conflicts
  // see the scan example for a more detailed description
  unsigned int atid = tid + (tid >> LOG_NUM_BANKS);

  // approximation coefficient
  // store in shared memory for further decomposition steps in this global step
  shared[atid] = (data0 + data1) * INV_SQRT_2;

  // all threads have to write approximation coefficient to shared memory before 
  // next steps can take place
  __syncthreads();

  // early out if possible
  // the compiler removes this part from the source because dlevels is 
  // a constant shader input
  // note: syncthreads in bodies of branches can lead to dead-locks unless the
  // the condition evaluates the same way for ALL threads of a block, as in 
  // this case
  if( dlevels > 1) 
    {
      // offset to second element in shared element which has to be used for the 
      // decomposition, effectively 2^(i - 1)
      unsigned int offset_neighbor = 1;
      // number of active threads per decomposition level
      // identiacal to the offset for the detail coefficients
      unsigned int num_threads = bdim >> 1;

      // index for the first element of the pair to process
      // the representation is still compact (and therefore still tid * 2) 
      // because the first step operated on registers and only the result has been
      // written to shared memory
      unsigned int idata0 = tid * 2;

      // offset levels to make the loop more efficient
      for( unsigned int i = 1; i < dlevels; ++i) 
        {
	  // Non-coalesced writes occur if the number of active threads becomes 
	  // less than 16 for a block because the start address for the first 
	  // block is not always aligned with 64 byte which is necessary for 
	  // coalesced access. However, the problem only occurs at high levels 
	  // with only a small number of active threads so that the total number of 
	  // non-coalesced access is rather small and does not justify the 
	  // computations which are necessary to avoid these uncoalesced writes
	  // (this has been tested and verified)
	  if( tid < num_threads) 
            {
	      // update stride, with each decomposition level the stride grows by a 
	      // factor of 2
	      unsigned int idata1 = idata0 + offset_neighbor;

	      // position of write into global memory
	      unsigned int g_wpos = (num_threads * gdim) + (bid * num_threads) + tid +  col_offset;

	      // compute wavelet decomposition step

	      // offset to avoid bank conflicts
	      unsigned int c_idata0 = idata0 + (idata0 >> LOG_NUM_BANKS);
	      unsigned int c_idata1 = idata1 + (idata1 >> LOG_NUM_BANKS);

	      // detail coefficient, not further modified so directly store 
	      // in global memory
	      od[g_wpos] = (shared[c_idata0] - shared[c_idata1]) * INV_SQRT_2;

	      // approximation coefficient
	      // note that the representation in shared memory becomes rather sparse 
	      // (with a lot of holes inbetween) but the storing scheme in global 
	      // memory guarantees that the common representation (approx, detail_0, 
	      // detail_1, ...)
	      // is achieved
	      shared[c_idata0] = (shared[c_idata0] + shared[c_idata1]) * INV_SQRT_2;

	      // update storage offset for details
	      num_threads = num_threads >> 1;   // div 2
	      offset_neighbor <<= 1;   // mul 2 
	      idata0 = idata0 << 1;   // mul 2     
            }

	  // sync after each decomposition step
	  __syncthreads();
        }

      // write the top most level element for the next decomposition steps
      // which are performed after an interblock syncronization on host side
      if( 0 == tid) 
        {
	  approx_final[bid + col_offset] = shared[0];
        }

    } // end early out if possible
}

