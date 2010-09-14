// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  include <windows.h>
#endif
#include <cuda_gl_interop.h>
#include <cutil.h>


#define STENCIL_WIDTH 3
#define STENCIL_HEIGHT 3
#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16


#include "convolution_kernels.cu"

extern "C" void initCuda(int argc, char **argv);
extern "C" void process(int pbo_in, int pbo_out, int width, int height, float * host_stencil_data, int kernel_width, int kernel_height);
extern "C" void pboRegister(int pbo);
extern "C" void pboUnregister(int pbo);

unsigned int timer = 0;

void initCuda(int argc, char **argv)
{
    CUT_DEVICE_INIT(argc, argv);
}


////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void process( int pbo_in, int pbo_out, int width, int height, float * host_stencil_data, int kernel_width, int kernel_height) 
{
    int *in_data;
    int* out_data;

	float * device_stencil_data;
	
	//allocate device memory for the stencil
	cudaMalloc( (void**) &device_stencil_data, kernel_width*kernel_height*sizeof(float) );

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	//transfer stencil from host memory to device memory
	cudaMemcpy( device_stencil_data, host_stencil_data, kernel_width*kernel_height*sizeof(float), cudaMemcpyHostToDevice );

	err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&in_data, pbo_in) );
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&out_data, pbo_out));

/*	// DEBUG
	float* tmp = (float*) calloc( 1, width*height*sizeof(int) );
	cudaMemcpy( tmp, in_data, width*height*sizeof(int), cudaMemcpyDeviceToHost );
	FILE *fout = fopen("input.raw", "wb");
	fwrite( tmp, width*height, sizeof(int), fout );
	fclose(fout);
	delete tmp;
*/

    dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

	// Setup timing
	CUT_SAFE_CALL(cutResetTimer(timer));  
	CUT_SAFE_CALL(cutStartTimer(timer));  

	// Invoke kernel
	cudaProcessEx4<<< dimGrid, dimBlock >>>(in_data, out_data, width, height, device_stencil_data);
/* 		cudaProcess<<< dimGrid, dimBlock >>>(in_data, out_data, width, height, device_stencil_data, kernel_width, kernel_height); */

	// Report timing
	cudaThreadSynchronize();
	CUT_SAFE_CALL(cutStopTimer(timer));  
	double time = cutGetTimerValue( timer ); 
	printf("time: %.2f ms.\n", time ); fflush(stdout);

	err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
/*
	// DEBUG
	float* tmp = (float*) calloc( 1, width*height*sizeof(int) );
	cudaMemcpy( tmp, out_data, width*height*sizeof(int), cudaMemcpyDeviceToHost );
	FILE *fout = fopen("output.raw", "wb");
	fwrite( tmp, width*height, sizeof(int), fout );
	fclose(fout);
	delete tmp;
*/

    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( pbo_in));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( pbo_out));

	cudaFree(device_stencil_data);

}

void pboRegister(int pbo)
{
    // register this buffer object with CUDA
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(pbo));
}
void pboUnregister(int pbo)
{
    // unregister this buffer object with CUDA
    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(pbo));
}
