#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P);


/* save a few registers */
#define TX (threadIdx.x)
#define TY (threadIdx.y)
#define BX (blockIdx.x)
#define BY (blockIdx.y)
#define W (M.width)

// Matrix multiplication kernel thread specification Tiled version
__global__ void MatrixMulKernelTiled(Matrix M, Matrix N, Matrix P);}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
