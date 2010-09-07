#include "DataVolumeCUDA.h"
#include "float_utils.h"
#include "uint_utils.h"

#include "ThresholdingKernels.cu"

#include <stdio.h>
#include <stdlib.h>

template< class TYPEd, class UINTd, class FLOATd >
void binary_threshold( DataVolumeCUDA<TYPEd, UINTd, FLOATd> *image, TYPEd threshold, TYPEd low_val, TYPEd high_val )
{
	// Setup dimensions of grid/blocks.
	dim3 blockDim(512,1,1);
	dim3 gridDim((unsigned int) ceil((double)(prod(image->dims)/blockDim.x)), 1, 1 );

	// Invoke kernel
	binary_threshold_kernel<<< gridDim, blockDim >>>( image->data, image->dims, threshold, low_val, high_val );

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'binary_threshold': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
}

template void binary_threshold( DataVolumeCUDA<float, uint3, float3> *image, float threshold, float low_val, float high_val );
