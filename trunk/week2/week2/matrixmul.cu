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
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include "matrixmul_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P, char type);



#define TEST_SIZE 32

void dumb_compare(Matrix a, Matrix b) {
  for(int i = 0; i < a.width ; i++)
    for(int j = 0; j < a.width ; j++) {
      float diff = a.elements[i*a.width + j] - b.elements[i*a.width + j];
      if ( abs(diff) > TOLERANCE) {
	printf("no match for (%d, %d) a=%f b=%f \tdiff==%f\n", i, j, a.elements[i*a.width + j], b.elements[i*a.width + j], diff);
      }
    }
}

void print_info(Matrix M, Matrix N) {
  printf("multiplying a (%d x %d) matrix with a (%d x %d) matrix\n", M.width, M.height,  N.width, N.height);
}

void print_usage() {
  printf("supports only square matrices!!\n");
  printf(" [s, t, T] size \n");
  printf(" (s)imple, (t)iled, (T)extured\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

  Matrix  M;
  Matrix  N;
  Matrix  P;

  int size;
  char type;
  /* a simple runner for testpurposes */
  if (argc != 3) {
    print_usage();
    return 42;
  }
  type = *argv[1];
  size = atoi(argv[2]);

  M  = AllocateMatrix(size, size, 1);
  N  = AllocateMatrix(M.width, M.height, 1);
  P  = AllocateMatrix(M.height, N.width, 0);
  
  srand(52);


  /* int errorM = 0, errorN = 0; */
  /* if (argc == 2) { */
  /*   M  = AllocateMatrix(TEST_SIZE, TEST_SIZE, 1); */
  /*   N  = AllocateMatrix(M.width, M.height, 1); */
  /*   P  = AllocateMatrix(M.height, N.width, 0); */
  /* } else if(argc != 5 && argc != 4)  */
  /*   { */
  /*     // Allocate and initialize the matrices */
  /*     M  = AllocateMatrix(rand() % 1024, rand() % 1024, 1); */
  /*     N  = AllocateMatrix(M.width, rand() % 1024, 1); */
  /*     P  = AllocateMatrix(M.height, N.width, 0); */
  /*   } */
  /* else */
  /*   { */
  /*     // Allocate and read in matrices from disk */
  /*     int* params = NULL; //(int*)malloc(3 * sizeof(int)); */
  /*     unsigned int data_read = 3; */
  /*     cutReadFilei(argv[1], &params, &data_read, true); */
  /*     if(data_read != 3){ */
  /* 	printf("Error reading parameter file\n"); */
  /* 	return 1; */
  /*     } */

  /*     M  = AllocateMatrix(params[0], params[1], 0); */
  /*     N  = AllocateMatrix(params[1], params[2], 0);		 */
  /*     P  = AllocateMatrix(params[0], params[2], 0); */
  /*     errorM = ReadFile(&M, argv[2]); */
  /*     errorN = ReadFile(&N, argv[3]); */
  /*     if(errorM  || errorN ) */
  /* 	{ */
  /* 	  printf("Error reading input files %d, %d\n", errorM, errorN); */
  /* 	  return 1; */
  /* 	} */
  /*   } */

  print_info(M, N);
  // M * N on the device
  MatrixMulOnDevice(M, N, P, type);
    
  printf("GPU computation complete\n");
  // compute the matrix multiplication on the CPU for comparison
  Matrix reference = AllocateMatrix(P.height, P.width, 0);
  computeGold(reference.elements, M.elements, N.elements, M.height, M.width, N.width);
        
  printf("CPU computation complete\n");
  // in this case check if the result is equivalent to the expected soluion
  CUTBoolean res = cutComparefe(reference.elements, P.elements, 
				P.height*P.width, TOLERANCE );
  printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
  if(0 == res) {
    dumb_compare(reference, P);
  }

/* no file output! */
  /* if(argc == 5) */
  /*   { */
  /*     WriteFile(P, argv[4]); */
  /*   } */
  /* else if(argc == 2) */
  /*   { */
  /*     WriteFile(P, argv[1]); */
  /*   }    */

  // Free matrices
  FreeMatrix(&M);
  FreeMatrix(&N);
  FreeMatrix(&P);
  return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P, char type)
{
  // Load M and N to the device
  Matrix Md = AllocateDeviceMatrix(M);
  CopyToDeviceMatrix(Md, M);
  HANDLE_ERROR( cudaPeekAtLastError(), "to device M");
  Matrix Nd = AllocateDeviceMatrix(N);
  CopyToDeviceMatrix(Nd, N);
  HANDLE_ERROR( cudaPeekAtLastError(), "to device N");

  // Allocate P on the device
  Matrix Pd = AllocateDeviceMatrix(P);
  CopyToDeviceMatrix(Pd, P); // Clear memory
  HANDLE_ERROR( cudaPeekAtLastError(), "to device P");

  /* the texture binding description */
  const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();


  // Setup the execution configuration
  dim3 dimGrid((int)ceil(1.0 * P.height/TILE_WIDTH), (int)ceil(1.0 * P.width/TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

  printf("launch configuration: grid (%d x %d) blocks grid (%d x %d)\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

  // Launch the device computation threads!
  switch (type) {
  case 's':
    MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
    break;
  case 't':
    MatrixMulKernelTiled<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
    break;
  case 'T':
    cudaBindTexture( 0,&Mtex, Md.elements ,&desc, Md.width * Md.height * sizeof(float));
    cudaBindTexture( 0,&Ntex, Nd.elements ,&desc, Nd.width * Nd.height * sizeof(float));
    MatrixMulKernelTextured<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
    cudaUnbindTexture(Mtex);
    cudaUnbindTexture(Ntex);
    break;
  default:
    printf("unknown type!!!!!!!!!!!!!!\n");
    return;
  }
  HANDLE_ERROR( cudaPeekAtLastError(), "launch");
  // Read P from the device
  CopyFromDeviceMatrix(P, Pd); 
  HANDLE_ERROR( cudaPeekAtLastError(), "from device");
  // Free device matrices
  FreeDeviceMatrix(&Md);
  FreeDeviceMatrix(&Nd);
  FreeDeviceMatrix(&Pd);
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
  Matrix Mdevice = M;
  int size = M.width * M.height * sizeof(float);
  cudaMalloc((void**)&Mdevice.elements, size);
  return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
  Matrix M;
  M.width = M.pitch = width;
  M.height = height;
  int size = M.width * M.height;
  M.elements = NULL;
    
  // don't allocate memory on option 2
  if(init == 2)
    return M;
		
  M.elements = (float*) malloc(size*sizeof(float));

  for(unsigned int i = 0; i < M.height * M.width; i++)
    {
      M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
      M.elements[i] = ( M.elements[i] > TOLERANCE ) ? M.elements[i] : 0.0f ;
    }
	

  return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
  /* we could zeropad to fixup non-sq matrices here? */

  int size = Mhost.width * Mhost.height * sizeof(float);
  Mdevice.height = Mhost.height;
  Mdevice.width = Mhost.width;
  Mdevice.pitch = Mhost.pitch;
  cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
	     cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
  int size = Mdevice.width * Mdevice.height * sizeof(float);
  cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
	     cudaMemcpyDeviceToHost);
    /* we could remove zeropad to fixup non-sq matrices here? */
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
  cudaFree(M->elements);
  M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
  free(M->elements);
  M->elements = NULL;
}

// Read a floating point matrix in from file
// Returns zero if the number of elements read is 
//  equals M.height * M.width, and 1 otherwise
int ReadFile(Matrix* M, char* file_name)
{
  unsigned int data_read = M->height*M->width;
  cutReadFilef(file_name, &(M->elements), &data_read, true);
  return (data_read != (M->height * M->width));
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
  cutWriteFilef(file_name, M.elements, M.width*M.height,
		0.0001f);
}
