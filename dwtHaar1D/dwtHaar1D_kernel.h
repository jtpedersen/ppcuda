#ifndef _DWTHAAR1D_KERNEL_H_
#define _DWTHAAR1D_KERNEL_H_
__global__ void 
dwtHaar1D( float* id, float* od, float* approx_final, 
          const unsigned int dlevels,
          const unsigned int slength_step_half,
	   const int bdim ) ;
#endif // #ifndef _DWTHAAR1D_KERNEL_H_
