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

void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P, char type, int repeats);




void dumb_compare(Matrix a, Matrix b) {
  FILE *f;
  f = fopen("multiplication.log", "w");
  for(int i = 0; i < a.width ; i++)
    for(int j = 0; j < a.width ; j++) {
      float diff = a.elements[i*a.width + j] - b.elements[i*a.width + j];
      if ( abs(diff) > TOLERANCE) {
	fprintf(f, "no match for (%d, %d) a=%f b=%f \tdiff==%f\n", i, j, a.elements[i*a.width + j], b.elements[i*a.width + j], diff);
      }
    }
  fclose(f);
  printf("there were errors, please look at the multiplication.log file\n");
}

void print_info(Matrix M, Matrix N) {
  printf("multiplying a (%d x %d) matrix with a (%d x %d) matrix\n", M.height,  M.width, N.height, N.width);
}

void print_usage() {
  printf("Default to  only square matrices!!\n");
  printf(" [s, t, T] size  [r, S repeats]\n");
  printf(" (s)imple, (t)iled, (T)extured\n");
  printf(" r option andom dimensioned matrixes maximum size is the size option\n");
  printf(" s option performs \"repeat\" calculations and times them \n");
}

void dump_matrix(const char *filename, Matrix M) {
  FILE *f;
  f = fopen(filename, "w");
  for(int j = 0; j < M.height ; j++) {
    for(int k = 0; k < M.width ; k++) {
      int idx = j*M.width + k;
      fprintf(f, "%1.3f ", M.elements[idx]);
    }
    fprintf(f, "\n");
  }
  fclose(f);
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
  if (argc < 3) {
    print_usage();
    return 42;
  }
  type = *argv[1];
  size = atoi(argv[2]);
  char matrix_type = 's';
  int repeats = 1;
  char stats = 0;

  if (argc >= 4) {
    /* could be either r andom og S tats which are performed on sq matrices */
    if ('S' == *argv[3]) {
      matrix_type = 's';
      repeats = atoi(argv[4]);
      stats = 1;
    } else {
      matrix_type = 'r';
    }
    
  }
  



  if ('r' == matrix_type ) {
    /* random matrices */
    M  = AllocateMatrix(1+rand() % size, 1+rand() % size, 1); /* rand%size could be zero */
    N  = AllocateMatrix(M.width, 1+rand() % size, 1); 
    P  = AllocateMatrix(M.height, N.width, 0);  
  } else {
    /* square matrices */
    M  = AllocateMatrix(size, size, 1);
    N  = AllocateMatrix(M.width, M.height, 3);
    P  = AllocateMatrix(M.height, N.width, 0);
  }

  srand(52);
  
  print_info(M, N);
  // M * N on the device
  MatrixMulOnDevice(M, N, P, type, repeats);
    
  printf("GPU computation complete\n");
  // compute the matrix multiplication on the CPU for comparison
  if (!stats) {  
    Matrix reference = AllocateMatrix(P.height, P.width, 0);
    computeGold(reference.elements, M.elements, N.elements, M.height, M.width, N.width);
        
    printf("CPU computation complete\n");
    // in this case check if the result is equivalent to the expected soluion
    CUTBoolean res = cutComparefe(reference.elements, P.elements, 
				  P.height*P.width, TOLERANCE );
    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
    fflush(0);
    if(0 == res) {
      dumb_compare(reference, P);
      /* dumb dump */
      dump_matrix("M.txt", M);
      dump_matrix("N.txt", N);
      dump_matrix("P.txt", P);
      /* who wants the results when they are correct? */
    }
    
  }
  // Free matrices
  FreeMatrix(&M);
  FreeMatrix(&N);
  FreeMatrix(&P);
  return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P, char type, int repeats)
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
  if ('T' == type) {
    cudaBindTexture( 0,&Mtex, Md.elements ,&desc, Md.width * Md.height * sizeof(float));
    cudaBindTexture( 0,&Ntex, Nd.elements ,&desc, Nd.width * Nd.height * sizeof(float));
  }


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Setup the execution configuration
  dim3 dimGrid((int)ceil(1.0 * N.width/TILE_WIDTH), (int)ceil(1.0 * M.height/TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

  printf("launch configuration: Blocks (%d x %d) Threads pr block (%d x %d)\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
  printf("will time %d execution(s) of ze kerne\n", repeats);

  for(int i=0; i < repeats; i++) {
    cudaEventRecord(start, 0);

  
    // Launch the device computation threads!
    switch (type) {
    case 's':
      MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
      break;
    case 't':
      MatrixMulKernelTiled<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
      break;
    case 'T':
      MatrixMulKernelTextured<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
      break;
    default:
      printf("unknown type!!!!!!!!!!!!!!\n");
      return;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    HANDLE_ERROR( cudaPeekAtLastError(), "launch");
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("%f\n", time);
  }


  HANDLE_ERROR( cudaPeekAtLastError(), "launch");
  // Read P from the device
  CopyFromDeviceMatrix(P, Pd); 
  HANDLE_ERROR( cudaPeekAtLastError(), "from device");
  if ('T' == type) {
    cudaUnbindTexture(Mtex);
    cudaUnbindTexture(Ntex);
  }

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
//if init == 3 Identity matrix
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

  for(int j = 0; j < M.height ; j++) {
    for(int k = 0; k < M.width ; k++) {
      int i = j *M.width + k;
      if (3 == init ) {
	M.elements[i] = (j == k ) ? 1.0f : 0.0f;
      } else {
	M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
	M.elements[i] = ( M.elements[i] > TOLERANCE ) ? M.elements[i] : 0.0f ;
      }
    }
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
