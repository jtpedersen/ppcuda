/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */
#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"


/*  save a few registers  */
#define TX (threadIdx.x)
#define TY (threadIdx.y)
#define BX (blockIdx.x)
#define BY (blockIdx.y)
#define W (M.width)


/* In general se pp 35 + in CUDA best practices (pedersen) */

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{

  int row = BY * blockDim.y + TY;
  int col = BX * blockDim.x + TX;

  /* we shouldn't calculate these */
  if (row >= P.height || col >= P.width)
    return;

  float pvalue = 0;	
  for(int k = 0; k < M.width ; ++k) {
    pvalue += M.elements[row*M.width + k] * N.elements[k*N.width + col];
  }
  P.elements[row*P.width + col] = pvalue;

}

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernelTiled(Matrix M, Matrix N, Matrix P)
{
   __shared__ float Ms[TILE_WIDTH][TILE_WIDTH];
   __shared__ float Ns[TILE_WIDTH][TILE_WIDTH];

   int row = BY * TILE_WIDTH + TY;
   int col = BX * TILE_WIDTH + TX;

   float pvalue = 0;	
   for(int m = 0; m < (int) ceil( 1.0f * W/TILE_WIDTH); ++m) {

      /* loading  with inline zero padding??*/
     if (row >= P.height || col >= P.width) {
         Ns[TY][TX] =  Ms[TY][TX] = 0.0f;
      } else {
         Ms[TY][TX] = M.elements[row * W + (m*TILE_WIDTH + TX)];
         Ns[TY][TX] = N.elements[(m*TILE_WIDTH + TY) * W + col];
      }
      __syncthreads();
      for(int k = 0; k < TILE_WIDTH; ++k) {
         pvalue += Ms[TY][k] * Ns[k][TX];
      }
      __syncthreads();

   }
   if (row >= P.height || col >= P.width) return;
   P.elements[row*P.width + col] = pvalue;
}

texture <float, 1> Mtex;
texture <float, 1> Ntex;

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernelTextured(Matrix M, Matrix N, Matrix P)
{



  int row = BY * blockDim.y + TY;
  int col = BX * blockDim.x + TX;

  /* we shouldn't calculate these */
  if (row >= P.height || col >= P.width)
    return;

  float pvalue = 0;	
  for(int k = 0; k < M.width ; ++k) {
    pvalue += tex1Dfetch(Mtex, (float) (row*M.width + k))
      * tex1Dfetch(Ntex, (float) (k*N.width + col));
  }
  P.elements[row*P.width + col] = pvalue;
}


#endif // #ifndef _MATRIXMUL_KERNEL_H_
