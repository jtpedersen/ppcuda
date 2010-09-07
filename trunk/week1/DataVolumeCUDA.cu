#include "DataVolumeCUDA.h"
#include "DataVolumeCUDAkernels.cu"

#include "uint_utils_host.h"
#include "float_utils_host.h"

#include <cublas.h>

#include <stdio.h>
#include <stdlib.h>

template< class TYPE, class UINTd, class FLOATd >
DataVolumeCUDA<TYPE, UINTd, FLOATd>::DataVolumeCUDA<TYPE, UINTd, FLOATd>( UINTd dims, FLOATd scale, FLOATd origin)
{
	this->scale = scale; 
	this->dims = dims;
	this->origin = origin;

	in_host_memory = false;
	host_data = 0x0;

//	printf("\nAllocating %d elements of size %d. Total %.2f MB", prod(dims), sizeof(TYPE), (prod(dims)*sizeof(TYPE))/(float)(1024*1024) );
	cudaMalloc( (void**) &data, prod(dims)*sizeof(TYPE));

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		data = 0x0; // Just in case we remove the 'exit' one day...
		printf("\nCuda error detected in 'DataVolumeCUDA::DataVolumeCUDA': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
}

template< class TYPE, class UINTd, class FLOATd >
DataVolumeCUDA<TYPE, UINTd, FLOATd>::~DataVolumeCUDA<TYPE, UINTd, FLOATd>()
{
	if( data ) 
		cudaFree( data );
	data = 0x0;

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		data = 0x0; // Just in case we remove the 'exit' one day...
		printf("\nCuda error detected in 'DataVolumeCUDA::~DataVolumeCUDA': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	if( host_data ) 
		free( host_data );
	host_data = 0x0;
}

template< class TYPE, class UINTd, class FLOATd >
DataVolumeCUDA<TYPE, UINTd, FLOATd>* DataVolumeCUDA<TYPE, UINTd, FLOATd>::downsample(UINTd factors)
{
	if( sizeof(TYPE) == sizeof(float3) ){
		printf("\nWARNING: DataVolumeCUDA<TYPE>::downsample not implemented on float3 type!\n");
		return 0x0;
	}
	else{

		UINTd ones; make_scale_vec( 1, ones );
		UINTd filter = dot_greater( factors, ones );
		UINTd inv_filter = dot_less_eq( factors, ones );
		UINTd new_dims = filter*(dims+factors-ones)/factors + inv_filter*dims;
		FLOATd new_scale = scale*cast_uintd_to_floatd(factors);
/*
		if (factors.x > 1)
			new_dims.x = (new_dims.x+factors.x-1)/factors.x;	
		if (factors.x > 1)
			new_dims.y = (new_dims.y+factors.y-1)/factors.y;	
		if (factors.z > 1)
			new_dims.z = (new_dims.z+factors.z-1)/factors.z;	
*/
		// Allocate memory for result
		DataVolumeCUDA<TYPE, UINTd, FLOATd>* result = new DataVolumeCUDA<TYPE, UINTd, FLOATd>( new_dims, new_scale, origin );

		cudaChannelFormatDesc channelDesc;

		// Copy to array and bind input texture

		switch(sizeof(TYPE)){

		case sizeof(float):
			channelDesc = cudaCreateChannelDesc<float>();	
			cudaBindTexture( 0, f1_1d_tex, data, channelDesc );
			break;

		case sizeof(float2):
			channelDesc = cudaCreateChannelDesc<float2>();	
			cudaBindTexture( 0, f2_1d_tex, data, channelDesc );
			break;

		case sizeof(float4):
			channelDesc = cudaCreateChannelDesc<float4>();	
			cudaBindTexture( 0, f4_1d_tex, data, channelDesc );
			break;
		}

		// Check for errors
		cudaError_t err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::downsample': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		dim3 blockDim(384,1,1);
		dim3 gridDim((unsigned int) ceil((double)prod(new_dims)/(double)blockDim.x), 1, 1 );

		// Make modulus image
		downsample_kernel<TYPE, UINTd, FLOATd><<< gridDim, blockDim >>>( result->data, dims, new_dims, factors );

		// Unbind texture
		switch(sizeof(TYPE)){

		case sizeof(float):
			cudaUnbindTexture( f1_1d_tex );
			break;
		case sizeof(float2):
			cudaUnbindTexture( f2_1d_tex );
			break;
		case sizeof(float4):
			cudaUnbindTexture( f4_1d_tex );
			break;
		}

		err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::downsample': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		return result;
	}
}


template <class TYPE>
static cudaArray* prepare_texture( unsigned int dims, TYPE *data )
{
	printf("\nprepare_texture not yet implemented with \"UINTd==unsigned int\".\n");
	return 0x0;
}

template <class TYPE>
static cudaArray* prepare_texture( uint2 dims, TYPE *data )
{
	// Prepare to build array for input texture
	cudaArray *tmp_image_array;
	cudaChannelFormatDesc channelDesc;
	cudaExtent extent; extent.width = dims.x; extent.height = dims.y; extent.depth = 1;
	cudaMemcpy3DParms cpy_params = {0}; 
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.extent = extent; 

	cudaError_t err;

	// Copy to array and bind input texture

	switch(sizeof(TYPE)){

		case sizeof(float):
			channelDesc = cudaCreateChannelDesc<float>();	
			cudaMalloc3DArray( &tmp_image_array, &channelDesc, extent );

			// Check for errors
			err = cudaGetLastError();
			if( err != cudaSuccess ){
				printf("\nCuda error detected in 'prepare_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
				exit(1);
			}

			cpy_params.dstArray = tmp_image_array;
			cpy_params.srcPtr = make_cudaPitchedPtr( (void*)data, dims.x*sizeof(float), dims.x*sizeof(float), dims.y );
			cudaMemcpy3D( &cpy_params );
			cudaBindTextureToArray( source_f1_2_tex, tmp_image_array, channelDesc );
			break;

		case sizeof(float2):
			channelDesc = cudaCreateChannelDesc<float2>();	
			cudaMalloc3DArray( &tmp_image_array, &channelDesc, extent );

			// Check for errors
			err = cudaGetLastError();
			if( err != cudaSuccess ){
				printf("\nCuda error detected in 'prepare_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
				exit(1);
			}

			cpy_params.dstArray = tmp_image_array;
			cpy_params.srcPtr = make_cudaPitchedPtr( (void*)data, dims.x*sizeof(float2), dims.x*sizeof(float2), dims.y );
			cudaMemcpy3D( &cpy_params );
			cudaBindTextureToArray( source_f2_2_tex, tmp_image_array, channelDesc );
			break;

		case sizeof(float4):
			channelDesc = cudaCreateChannelDesc<float4>();	
			cudaMalloc3DArray( &tmp_image_array, &channelDesc, extent );

			// Check for errors
			err = cudaGetLastError();
			if( err != cudaSuccess ){
				printf("\nCuda error detected in 'prepare_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
				exit(1);
			}

			cpy_params.dstArray = tmp_image_array;
			cpy_params.srcPtr = make_cudaPitchedPtr( (void*)data, dims.x*sizeof(float4), dims.x*sizeof(float4), dims.y );
			cudaMemcpy3D( &cpy_params );
			cudaBindTextureToArray( source_f4_2_tex, tmp_image_array, channelDesc );
			break;
	}

	// Check for errors
	err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'prepare_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	return tmp_image_array;
}

template <class TYPE>
static cudaArray* prepare_texture( uint3 dims, TYPE *data )
{
	// Prepare to build array for input texture
	cudaArray *tmp_image_array;
	cudaChannelFormatDesc channelDesc;
	cudaExtent extent; extent.width = dims.x; extent.height = dims.y; extent.depth = dims.z;
	cudaMemcpy3DParms cpy_params = {0}; 
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.extent = extent; 

	cudaError_t err;

	// Copy to array and bind input texture

	switch(sizeof(TYPE)){

		case sizeof(float):
			channelDesc = cudaCreateChannelDesc<float>();	
			cudaMalloc3DArray( &tmp_image_array, &channelDesc, extent );

			// Check for errors
			err = cudaGetLastError();
			if( err != cudaSuccess ){
				printf("\nCuda error detected in 'prepare_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
				exit(1);
			}

			cpy_params.dstArray = tmp_image_array;
			cpy_params.srcPtr = make_cudaPitchedPtr( (void*)data, dims.x*sizeof(float), dims.x*sizeof(float), dims.y );
			cudaMemcpy3D( &cpy_params );
			cudaBindTextureToArray( source_f1_3_tex, tmp_image_array, channelDesc );
			break;

		case sizeof(float2):
			channelDesc = cudaCreateChannelDesc<float2>();	
			cudaMalloc3DArray( &tmp_image_array, &channelDesc, extent );

			// Check for errors
			err = cudaGetLastError();
			if( err != cudaSuccess ){
				printf("\nCuda error detected in 'prepare_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
				exit(1);
			}

			cpy_params.dstArray = tmp_image_array;
			cpy_params.srcPtr = make_cudaPitchedPtr( (void*)data, dims.x*sizeof(float2), dims.x*sizeof(float2), dims.y );
			cudaMemcpy3D( &cpy_params );
			cudaBindTextureToArray( source_f2_3_tex, tmp_image_array, channelDesc );
			break;

		case sizeof(float4):
			channelDesc = cudaCreateChannelDesc<float4>();	
			cudaMalloc3DArray( &tmp_image_array, &channelDesc, extent );

			// Check for errors
			err = cudaGetLastError();
			if( err != cudaSuccess ){
				printf("\nCuda error detected in 'prepare_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
				exit(1);
			}

			cpy_params.dstArray = tmp_image_array;
			cpy_params.srcPtr = make_cudaPitchedPtr( (void*)data, dims.x*sizeof(float4), dims.x*sizeof(float4), dims.y );
			cudaMemcpy3D( &cpy_params );
			cudaBindTextureToArray( source_f4_3_tex, tmp_image_array, channelDesc );
			break;
	}

	// Check for errors
	err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'prepare_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	return tmp_image_array;
}

template <class TYPE>
static cudaArray* prepare_texture( uint4 dims, TYPE *data )
{
	// Prepare to build array for input texture
	cudaArray *tmp_image_array;
	cudaChannelFormatDesc channelDesc;
	cudaExtent extent; extent.width = dims.x; extent.height = dims.y; extent.depth = dims.z*dims.w;
	cudaMemcpy3DParms cpy_params = {0}; 
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.extent = extent; 

	cudaError_t err;

	// Copy to array and bind input texture

	switch(sizeof(TYPE)){

		case sizeof(float):
			channelDesc = cudaCreateChannelDesc<float>();	
			cudaMalloc3DArray( &tmp_image_array, &channelDesc, extent );

			// Check for errors
			err = cudaGetLastError();
			if( err != cudaSuccess ){
				printf("\nCuda error detected in 'prepare_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
				exit(1);
			}

			cpy_params.dstArray = tmp_image_array;
			cpy_params.srcPtr = make_cudaPitchedPtr( (void*)data, dims.x*sizeof(float), dims.x*sizeof(float), dims.y );
			cudaMemcpy3D( &cpy_params );
			cudaBindTextureToArray( source_f1_4_tex, tmp_image_array, channelDesc );
			break;

		case sizeof(float2):
			channelDesc = cudaCreateChannelDesc<float2>();	
			cudaMalloc3DArray( &tmp_image_array, &channelDesc, extent );

			// Check for errors
			err = cudaGetLastError();
			if( err != cudaSuccess ){
				printf("\nCuda error detected in 'prepare_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
				exit(1);
			}

			cpy_params.dstArray = tmp_image_array;
			cpy_params.srcPtr = make_cudaPitchedPtr( (void*)data, dims.x*sizeof(float2), dims.x*sizeof(float2), dims.y );
			cudaMemcpy3D( &cpy_params );
			cudaBindTextureToArray( source_f2_4_tex, tmp_image_array, channelDesc );
			break;

		case sizeof(float4):
			channelDesc = cudaCreateChannelDesc<float4>();	
			cudaMalloc3DArray( &tmp_image_array, &channelDesc, extent );

			// Check for errors
			err = cudaGetLastError();
			if( err != cudaSuccess ){
				printf("\nCuda error detected in 'prepare_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
				exit(1);
			}

			cpy_params.dstArray = tmp_image_array;
			cpy_params.srcPtr = make_cudaPitchedPtr( (void*)data, dims.x*sizeof(float4), dims.x*sizeof(float4), dims.y );
			cudaMemcpy3D( &cpy_params );
			cudaBindTextureToArray( source_f4_4_tex, tmp_image_array, channelDesc );
			break;
	}

	// Check for errors
	err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'prepare_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	return tmp_image_array;
}

template <class TYPE>
static bool release_texture( unsigned int dims, cudaArray *image_array )
{
	// Unbind texture
	switch(sizeof(TYPE)){

		case sizeof(float):
			cudaUnbindTexture( source_f1_1_tex );
			break;
		case sizeof(float2):
			cudaUnbindTexture( source_f2_1_tex );
			break;
		case sizeof(float4):
			cudaUnbindTexture( source_f4_1_tex );
			break;
	}

	// Free array
	cudaThreadSynchronize(); // This saves us from an unspecified launch error from 'cudaFreeArray' below!!!
	cudaFreeArray(image_array);

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'release_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	return true;
}

template <class TYPE>
static bool release_texture( uint2 dims, cudaArray *image_array )
{
	// Unbind texture
	switch(sizeof(TYPE)){

		case sizeof(float):
			cudaUnbindTexture( source_f1_2_tex );
			break;
		case sizeof(float2):
			cudaUnbindTexture( source_f2_2_tex );
			break;
		case sizeof(float4):
			cudaUnbindTexture( source_f4_2_tex );
			break;
	}

	// Free array
	cudaThreadSynchronize(); // This saves us from an unspecified launch error from 'cudaFreeArray' below!!!
	cudaFreeArray(image_array);

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'release_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	return true;
}

template <class TYPE>
static bool release_texture( uint3 dims, cudaArray *image_array )
{
	// Unbind texture
	switch(sizeof(TYPE)){

		case sizeof(float):
			cudaUnbindTexture( source_f1_3_tex );
			break;
		case sizeof(float2):
			cudaUnbindTexture( source_f2_3_tex );
			break;
		case sizeof(float4):
			cudaUnbindTexture( source_f4_3_tex );
			break;
	}

	// Free array
	cudaThreadSynchronize(); // This saves us from an unspecified launch error from 'cudaFreeArray' below!!!
	cudaFreeArray(image_array);

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'release_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	return true;
}

template <class TYPE>
static bool release_texture( uint4 dims, cudaArray *image_array )
{
	// Unbind texture
	switch(sizeof(TYPE)){

		case sizeof(float):
			cudaUnbindTexture( source_f1_4_tex );
			break;
		case sizeof(float2):
			cudaUnbindTexture( source_f2_4_tex );
			break;
		case sizeof(float4):
			cudaUnbindTexture( source_f4_4_tex );
			break;
	}

	// Free array
	cudaThreadSynchronize(); // This saves us from an unspecified launch error from 'cudaFreeArray' below!!!
	cudaFreeArray(image_array);

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'release_texture': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	return true;
}

template< class TYPE, class UINTd, class FLOATd >
DataVolumeCUDA<TYPE, UINTd, FLOATd>* DataVolumeCUDA<TYPE, UINTd, FLOATd>::upsample(UINTd factors, UINTd new_dims)
{
	if( sizeof(TYPE) == sizeof(float3) ){
		printf("\nWARNING: DataVolumeCUDA<TYPE>::upsample not implemented on the 'float3' type.");
		return 0x0;
	}
	else{

		DataVolumeCUDA<TYPE, UINTd, FLOATd>* result;

		UINTd zeros; make_scale_vec( 0, zeros );
		UINTd ones; make_scale_vec( 1, ones );
		UINTd zero_elements = dot_equal( new_dims, zeros );
		UINTd filter = dot_greater( factors, ones );
		UINTd inv_filter = dot_less_eq( factors, ones );

		if( prod(zero_elements)>0 )
			new_dims = filter*dims*factors + inv_filter*dims;

		FLOATd new_scale = cast_uintd_to_floatd(filter)*scale/cast_uintd_to_floatd(factors) + cast_uintd_to_floatd(inv_filter)*scale;
/*
		if (new_dims.x == 0 || new_dims.y == 0 || new_dims.z == 0)
		{
			new_dims = dims;
			if (factors.x > 1)
				new_dims.x = new_dims.x*factors.x;	
			if (factors.y > 1)
				new_dims.y = new_dims.y*factors.y;	
			if (factors.z > 1)
				new_dims.z = new_dims.z*factors.z;
		}

		if (factors.x > 1)
			new_scale.x = new_scale.x/((float)factors.x);
		if (factors.y > 1)
			new_scale.y = new_scale.y/((float)factors.y);
		if (factors.z > 1)
			new_scale.z = new_scale.z/((float)factors.z);
*/
		result = new DataVolumeCUDA<TYPE, UINTd, FLOATd>( new_dims, new_scale, origin );

		// Setup texture
		cudaArray *tmp_image_array = prepare_texture<TYPE>(dims, data);
		if( !tmp_image_array ){
			printf("\nImplementation immature! Returning 0x0.\n");
			return 0x0;
		}

		// Check for errors
		cudaError_t err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::upsample': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		dim3 blockDim(512,1,1);
		dim3 gridDim((unsigned int) ceil((double)prod(result->dims)/(double)blockDim.x), 1, 1 );

		upsample_kernel<TYPE, UINTd, FLOATd><<< gridDim, blockDim >>>( result->data, result->dims, factors, dims );

		// Release texture
		bool _impl = release_texture<TYPE>(dims, tmp_image_array );
		if( !_impl ){
			printf("\nImplementation inconsistent! Returning 0x0.\n");
			return 0x0;
		}
		tmp_image_array = 0x0;

		// Check for errors
		err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::upsample': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		return result;
	}
}

template< class TYPE, class UINTd, class FLOATd >
TYPE DataVolumeCUDA<TYPE, UINTd, FLOATd>::vectorMax()
{
	// Temporary storage for magnitudes image

	DataVolumeCUDA<float, UINTd, FLOATd> *_magnitudes = magnitudes();

	//Find the maximum value idx in the array
	int max_idx = cublasIsamax( prod(dims), _magnitudes->data, 1 );

//	uint3 co = idx_to_co( max_idx, dims );
//	printf("\nMax index is: (%d, %d, %d)", co.x, co.y, co.z );

	delete _magnitudes;
	_magnitudes = 0x0;

	// Copy that value back to host memory
	TYPE max_vec;
	cudaMemcpy( &max_vec, data+max_idx-1, sizeof(TYPE), cudaMemcpyDeviceToHost );

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::vectorMax': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	return max_vec;
}


template< class TYPE, class UINTd, class FLOATd >
float DataVolumeCUDA<TYPE, UINTd, FLOATd>::jacobianMin()
{
	// Temporary storage for magnitudes image
	float *_jacobians = jacobians();

	//Find the maximum value idx in the array
	int min_idx = cublasIsamin( prod(dims), _jacobians, 1 );

	// Copy that value back to host memory
	float minimum;
	cudaMemcpy( &minimum, _jacobians+min_idx-1, sizeof(float), cudaMemcpyDeviceToHost );

	cudaFree( _jacobians );
	_jacobians = 0x0;

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::jacobianMin': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
	if (minimum<0.5)
		printf("\nWarning: Jacobian min = %f", minimum );

	return minimum;
}

template< class TYPE, class UINTd, class FLOATd >
DataVolumeCUDA<float, UINTd, FLOATd>* DataVolumeCUDA<TYPE, UINTd, FLOATd>::magnitudes()
{
//	float *_magnitudes;
//	cudaMalloc( (void**) &_magnitudes, prod(dims)*sizeof(float));

	DataVolumeCUDA<float, UINTd, FLOATd> * _magnitudes = new DataVolumeCUDA<float, UINTd, FLOATd>(dims, scale, origin);

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::magnitudes': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	dim3 blockDim(512,1,1);
	dim3 gridDim((unsigned int) ceil((double)prod(dims)/(double)blockDim.x), 1, 1 );

	// Make modulus image
	magnitudes_kernel<TYPE><<< gridDim, blockDim >>>( data, _magnitudes->data, prod(dims) );

	err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::magnitudes': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	return _magnitudes;
}

template< class TYPE, class UINTd, class FLOATd >
void DataVolumeCUDA<TYPE, UINTd, FLOATd>::clear( const TYPE &val )
{

	// Find dimensions of grid/blocks.

	dim3 blockDim(512,1,1);
	dim3 gridDim((unsigned int) ceil((double)prod(dims)/(double)blockDim.x), 1, 1 );

	// Invoke kernel
	clear_kernel<TYPE><<< gridDim, blockDim >>>( val, data, prod(dims) );

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::clear': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
}

template< class TYPE, class UINTd, class FLOATd >
void DataVolumeCUDA<TYPE, UINTd, FLOATd>::setLastComponent( const float val )
{

	// Find dimensions of grid/blocks.

	dim3 blockDim(512,1,1);
	dim3 gridDim((unsigned int) ceil((double)prod(dims)/(double)blockDim.x), 1, 1 );

	// Invoke kernel
	setLastComponent_kernel<TYPE><<< gridDim, blockDim >>>( val, data, prod(dims) );

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::setComponent': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
}

template< class TYPE, class UINTd, class FLOATd >
float* DataVolumeCUDA<TYPE, UINTd, FLOATd>::jacobians()
{
	if( sizeof(TYPE) != sizeof(float4) ){
		printf("\nWARNING: DataVolumeCUDA<TYPE>::jacobians only implemented on the 'float4' type.");
		return 0x0;
	}
	else{

		// Allocate memory for result
		float *_jacobians;
		cudaMalloc( (void**) &_jacobians, prod(dims)*sizeof(float));

		// Bind textures
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();	
		cudaBindTexture( 0, in_deformations_tex, data, channelDesc );

		// Check for errors
		cudaError_t err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::jacobians': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		dim3 blockDim(256,1,1);
		dim3 gridDim((unsigned int) ceil((double)prod(dims)/(double)blockDim.x), 1, 1 );

		// Make modulus image
		calculate_jacobians_kernel<FLOATd, UINTd><<< gridDim, blockDim >>>( _jacobians, scale, dims );

		cudaUnbindTexture( in_deformations_tex );

		err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::jacobians': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		return _jacobians;
	}
}

template< class TYPE, class UINTd, class FLOATd > DataVolumeCUDA<TYPE, UINTd, FLOATd>* 
DataVolumeCUDA<TYPE, UINTd, FLOATd>::applyDeformation( DataVolumeCUDA<FLOATd, UINTd, FLOATd> *in_deformations, bool unit_displacements )
{
	if( sizeof(TYPE) == sizeof(float3) ){
		printf("\nWARNING: DataVolumeCUDA<TYPE>::applyDeformation not implemented for the 'float3' type.");
		return 0x0;
	}
	else{

		// Allocate memory for result
		DataVolumeCUDA<TYPE, UINTd, FLOATd> *result = new DataVolumeCUDA<TYPE, UINTd, FLOATd>( dims, scale, origin );

		// Setup texture
		cudaArray *tmp_image_array = prepare_texture<TYPE>(dims, data);
		if( !tmp_image_array ){
			printf("\nImplementation immature! Returning 0x0.\n");
			return 0x0;
		}

		// Check for errors
		cudaError_t err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::applyDeformation': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		FLOATd ones; make_scale_vec( 1.0f, ones );

		dim3 blockDim(512,1,1);
		dim3 gridDim((unsigned int) ceil((double)prod(dims)/(double)blockDim.x), 1, 1 );

		if (unit_displacements)
			apply_deformations_kernel<TYPE, UINTd, FLOATd><<< gridDim, blockDim >>>( in_deformations->data, result->data, ones, dims );
		else
			apply_deformations_kernel<TYPE, UINTd, FLOATd><<< gridDim, blockDim >>>( in_deformations->data, result->data, scale, dims );

		// Release texture
		bool _impl = release_texture<TYPE>(dims, tmp_image_array );
		if( !_impl ){
			printf("\nImplementation inconsistent! Returning 0x0.\n");
			return 0x0;
		}
		tmp_image_array = 0x0;

		// Check for errors
		err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::applyDeformation': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		return result;
	}
}


template< class TYPE, class UINTd, class FLOATd >
DataVolumeCUDA<TYPE, UINTd, FLOATd>* DataVolumeCUDA<TYPE, UINTd, FLOATd>::concatenateDeformations( DataVolumeCUDA<TYPE, UINTd, FLOATd> *in_deformations )
{
	if( !((sizeof(TYPE)==sizeof(float4))||(sizeof(TYPE)==sizeof(float2))) ){
		printf("\nWARNING: DataVolumeCUDA<TYPE>::concatenateDeformations only implemented for the 'float2' and 'float4' types.");
		return 0x0;
	}
	else{

		// Allocate memory for result
		DataVolumeCUDA<TYPE, UINTd, FLOATd> *result = new DataVolumeCUDA<TYPE, UINTd, FLOATd>( dims, scale, origin );

		cudaArray *tmp_image_array = prepare_texture<TYPE>(dims, data);

		// Check for errors
		cudaError_t err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::concatenateDeformations': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		dim3 blockDim(384,1,1);
		dim3 gridDim((unsigned int) ceil((double)prod(dims)/(double)blockDim.x), 1, 1 );

		// Make modulus image
		concatenate_deformation_kernel<TYPE, UINTd, FLOATd><<< gridDim, blockDim >>>( in_deformations->data, result->data, scale, dims );

		release_texture<TYPE>(dims, tmp_image_array );
		tmp_image_array = 0x0;

		err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::concatenateDeformations': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		return result;
	}
}

template< class TYPE, class UINTd, class FLOATd >
DataVolumeCUDA<TYPE, UINTd, FLOATd>* DataVolumeCUDA<TYPE, UINTd, FLOATd>::concatenateDisplacementFields( DataVolumeCUDA<TYPE, UINTd, FLOATd> *in_field )
{
	if( !((sizeof(TYPE)==sizeof(float4))||(sizeof(TYPE)==sizeof(float2))) ){
		printf("\nWARNING: DataVolumeCUDA<TYPE>::concatenateDisplacementFields only implemented for the 'float2' and 'float4' types.");
		return 0x0;
	}
	else{

		// Allocate memory for result
		DataVolumeCUDA<TYPE, UINTd, FLOATd> *result = new DataVolumeCUDA<TYPE, UINTd, FLOATd>( dims, scale, origin );

		cudaArray *tmp_image_array = prepare_texture<TYPE>(dims, in_field->data);

		// Check for errors
		cudaError_t err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::concatenateDisplacementFields': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		dim3 blockDim(384,1,1);
		dim3 gridDim((unsigned int) ceil((double)prod(dims)/(double)blockDim.x), 1, 1 );

		// Make modulus image
		concatenate_deformation_kernel<TYPE, UINTd, FLOATd><<< gridDim, blockDim >>>( data, result->data, scale, dims );

		release_texture<TYPE>(dims, tmp_image_array );
		tmp_image_array = 0x0;

		err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::concatenateDisplacementFields': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		return result;
	}
}

template< class TYPE, class UINTd, class FLOATd >
void DataVolumeCUDA<TYPE, UINTd, FLOATd>::offset( TYPE scale, TYPE *offsets )
{

	// Find dimensions of grid/blocks.

	dim3 blockDim(512,1,1);
	dim3 gridDim((unsigned int) ceil((double)prod(dims)/(double)blockDim.x), 1, 1 );

	// Invoke kernel
	offset_kernel<TYPE, UINTd><<< gridDim, blockDim >>>( data, scale, offsets, dims );

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::offset': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
}

template< class TYPE, class UINTd, class FLOATd >
void DataVolumeCUDA<TYPE, UINTd, FLOATd>::cropDeformationToVolume()
{
	if( sizeof(TYPE) != sizeof(float4) ){
		printf("\nWARNING: DataVolumeCUDA<TYPE>::cropDeformationToVolume only implemented for the 'float4' type.");
		return;
	}
	else{

		// Find dimensions of grid/blocks.

		dim3 blockDim(512,1,1);
		dim3 gridDim((unsigned int) ceil((double)prod(dims)/(double)blockDim.x), 1, 1 );

		// Invoke kernel
		crop_deformation_kernel<TYPE, UINTd, FLOATd><<< gridDim, blockDim >>>( data, scale, dims );

		cudaError_t err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::cropDeformationToVolume': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}	
	}
}

template< class TYPE, class UINTd, class FLOATd >
void DataVolumeCUDA<TYPE, UINTd, FLOATd>::makeBorder( const TYPE &val )
{
	// Find dimensions of grid/blocks.

	dim3 blockDim(512,1,1);
	dim3 gridDim((unsigned int) ceil((double)prod(dims)/(double)blockDim.x), 1, 1 );

	// Invoke kernel
	make_border_kernel<TYPE, UINTd><<< gridDim, blockDim >>>( val, data, dims );

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::makeBorder': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
}

template< class TYPE, class UINTd, class FLOATd >
void DataVolumeCUDA<TYPE, UINTd, FLOATd>::multiply( const TYPE &val )
{
	// Find dimensions of grid/blocks.

	dim3 blockDim(512,1,1);
	dim3 gridDim((unsigned int) ceil((double)prod(dims)/(double)blockDim.x), 1, 1 );

	// Invoke kernel
	multiply_kernel<TYPE, UINTd><<< gridDim, blockDim >>>( val, data, dims );

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::multiply': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
}

template< class TYPE, class UINTd, class FLOATd >
void DataVolumeCUDA<TYPE, UINTd, FLOATd>::divide( const TYPE &_val )
{
	// Find dimensions of grid/blocks.

	dim3 blockDim(512,1,1);
	dim3 gridDim((unsigned int) ceil((double)prod(dims)/(double)blockDim.x), 1, 1 );

	TYPE ones; make_scale_vec( 1.0f, ones );
	TYPE val = ones / _val;

	// Invoke kernel
	multiply_kernel<TYPE, UINTd><<< gridDim, blockDim >>>( val, data, dims );

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::divide': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
}

template< class TYPE, class UINTd, class FLOATd >
void DataVolumeCUDA<TYPE, UINTd, FLOATd>::add( const TYPE &val )
{
	// Find dimensions of grid/blocks.

	dim3 blockDim(512,1,1);
	dim3 gridDim((unsigned int) ceil((double)prod(dims)/(double)blockDim.x), 1, 1 );

	// Invoke kernel
	add_kernel<TYPE, UINTd><<< gridDim, blockDim >>>( val, data, dims );

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::add': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
}

template< class TYPE, class UINTd, class FLOATd > void 
DataVolumeCUDA<TYPE, UINTd, FLOATd>::add( DataVolumeCUDA<TYPE, UINTd, FLOATd> *volume )
{
	// Find dimensions of grid/blocks.

	dim3 blockDim(512,1,1);
	dim3 gridDim((unsigned int) ceil((double)prod(dims)/(double)blockDim.x), 1, 1 );

	// Invoke kernel
	add_volume_kernel<TYPE, UINTd><<< gridDim, blockDim >>>( volume->data, data, dims );

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::add': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}
}

template< class TYPE, class UINTd, class FLOATd >
TYPE* DataVolumeCUDA<TYPE, UINTd, FLOATd>::evaluateInterpolated( FLOATd *points, unsigned int num_points )
{
	TYPE *result;
	
	cudaMalloc((void**) &result, num_points*sizeof(TYPE));

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::evaluateInterpolated': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	dim3 blockDim(512,1,1);
	dim3 gridDim((unsigned int) ceil((double)num_points/(double)blockDim.x), 1, 1 );

	// Invoke kernel
	evaluate_linear_kernel<TYPE, UINTd, FLOATd><<< gridDim, blockDim >>>( points, num_points, data, dims, scale, origin, result );

	err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::evaluateInterpolated': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	return result;
}

/*
template< class TYPE, class uint4, class float4 >
TYPE* DataVolumeCUDA<TYPE, uint4, float4>::evaluateInterpolated( float4 *points, unsigned int num_points )
{
	TYPE * result;
	
	cudaMalloc((void**) &result, num_points*sizeof(TYPE));

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::evaluateInterpolated': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	dim3 blockDim(512,1,1);
	dim3 gridDim((unsigned int) ceil((double)num_points/(double)blockDim.x), 1, 1 );

	// Invoke kernel
	evaluate_linear_kernel_float4<TYPE><<< gridDim, blockDim >>>( points, num_points, data, dims, scale, origin, result );

	err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'DataVolumeCUDA::evaluateInterpolated': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	return result;
}
*/


template< class TYPE, class UINTd, class FLOATd >
TYPE DataVolumeCUDA<TYPE, UINTd, FLOATd>::max()
{
	if( sizeof(TYPE) != sizeof(float) ){
		printf("\nWARNING: DataVolumeCUDA<TYPE>::max not implemented on this type.");
		TYPE ret;
		zero(&ret);
		return ret;
	}
	else{

		//Find the maximum value idx in the array
		int max_idx = cublasIsamax( prod(dims), (float*)data, 1 );

		// Copy that value back to host memory
		TYPE max;
		cudaMemcpy( &max, data+max_idx-1, sizeof(TYPE), cudaMemcpyDeviceToHost );

		cudaError_t err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::max': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		return max;
	}
}

template< class TYPE, class UINTd, class FLOATd >
TYPE DataVolumeCUDA<TYPE, UINTd, FLOATd>::min()
{
	if( sizeof(TYPE) != sizeof(float) ){
		printf("\nWARNING: DataVolumeCUDA<TYPE>::min not implemented on this type.");
		TYPE ret;
		zero(&ret);
		return ret;
	}
	else{

		//Find the maximum value idx in the array
		int min_idx = cublasIsamin( prod(dims), (float*)data, 1 );

		// Copy that value back to host memory
		TYPE min;
		cudaMemcpy( &min, data+min_idx-1, sizeof(TYPE), cudaMemcpyDeviceToHost );

		cudaError_t err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::min': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		return min;
	}
}


template< class TYPE, class UINTd, class FLOATd >
void DataVolumeCUDA<TYPE, UINTd, FLOATd>::normalize(TYPE new_min, TYPE new_max, TYPE prev_min, TYPE prev_max)
{
	if( sizeof(TYPE) != sizeof(float) ){
		printf("\nWARNING: DataVolumeCUDA<TYPE>::normalize not implemented on this type.");
		return;
	}
	else{

//		TYPE prev_min = min();
//		TYPE prev_max = max();

//		printf("max: %f", *((float *)&prev_max));
//		printf("min: %f", *((float *)&prev_min));

		TYPE zer;
		zero(&zer);
		add(zer-prev_min);

		TYPE on;
		one(&on);
		multiply((on/(prev_max-prev_min))*new_max);

		add(new_min);
	}
}

template< class TYPE, class UINTd, class FLOATd >
DataVolumeCUDA<TYPE, UINTd, FLOATd>* DataVolumeCUDA<TYPE, UINTd, FLOATd>::convolve(DataVolumeCUDA<TYPE, UINTd, FLOATd>* kernel)
{
	UINTd kdims = kernel->dims;

	if ( !odd(kdims) )
	{
		printf("\nERROR: DataVolumeCUDA::convolve needs odd kernel sizes");
		return NULL;
	} 

	if( sizeof(TYPE) == sizeof(float3) )
	{
		printf("\nWARNING: DataVolumeCUDA<TYPE,.,.>::convolve not implemented on the 'float3' type.");
		return 0x0;
	}
	else{

		// Allocate memory for result
		DataVolumeCUDA<TYPE, UINTd, FLOATd>* result = new DataVolumeCUDA<TYPE, UINTd, FLOATd>( dims, scale, origin );

		cudaChannelFormatDesc channelDesc;
		cudaChannelFormatDesc channelDesc2;

		// Copy to array and bind input texture

		switch(sizeof(TYPE))
		{

		case sizeof(float):
			channelDesc = cudaCreateChannelDesc<float>();	
			cudaBindTexture( 0, f1_1d_tex, data, channelDesc );
			channelDesc2 = cudaCreateChannelDesc<float>();	
			cudaBindTexture( 0, f1_1d_tex2, kernel->data, channelDesc2 );
			break;

		case sizeof(float2):
			channelDesc = cudaCreateChannelDesc<float2>();	
			cudaBindTexture( 0, f2_1d_tex, data, channelDesc );
			channelDesc2 = cudaCreateChannelDesc<float2>();	
			cudaBindTexture( 0, f2_1d_tex2, kernel->data, channelDesc2 );
			break;

		case sizeof(float4):
			channelDesc = cudaCreateChannelDesc<float4>();	
			cudaBindTexture( 0, f4_1d_tex, data, channelDesc );
			channelDesc2 = cudaCreateChannelDesc<float4>();	
			cudaBindTexture( 0, f4_1d_tex2, kernel->data, channelDesc2 );
			break;
		}

		// Check for errors
		cudaError_t err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::convolve': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		dim3 blockDim(512,1,1);
		dim3 gridDim((unsigned int)ceil((double)prod(dims)/(double)blockDim.x), 1, 1 );

		convolution_kernel<TYPE, UINTd><<< gridDim, blockDim >>>( result->data, dims, kernel->dims);

		// Unbind texture
		switch(sizeof(TYPE)){

		case sizeof(float):
			cudaUnbindTexture( f1_1d_tex );
			cudaUnbindTexture( f1_1d_tex2 );
			break;
		case sizeof(float2):
			cudaUnbindTexture( f2_1d_tex );
			cudaUnbindTexture( f2_1d_tex2 );
			break;
		case sizeof(float4):
			cudaUnbindTexture( f4_1d_tex );
			cudaUnbindTexture( f4_1d_tex2 );
			break;
		}

		err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::convolve': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		return result;
	}
}

template< class TYPE, class UINTd, class FLOATd >
void DataVolumeCUDA<TYPE, UINTd, FLOATd>::swapOut()
{
//	assert(!in_host_memory && host_data == 0x0);

	host_data = (TYPE *) malloc(prod(dims)*sizeof(TYPE));

	if( host_data == 0x0){
		printf("\nError detected in 'DataVolumeCUDA::swapOut': Not enough memory. Quitting.\n" ); fflush(stdout);
		exit(1);
	}

	cudaMemcpy(	host_data, data, prod(dims)*sizeof(TYPE), cudaMemcpyDeviceToHost );

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		data = 0x0; // Just in case we remove the 'exit' one day...
		printf("\nCuda error detected in 'DataVolumeCUDA::swapOut': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	in_host_memory = true;
	cudaFree(data);
	data = 0x0;
}

template< class TYPE, class UINTd, class FLOATd >
void DataVolumeCUDA<TYPE, UINTd, FLOATd>::swapIn()
{
//	assert(in_host_memory && data == 0x0);

	cudaMalloc( (void**) &data, prod(dims)*sizeof(TYPE));
	cudaMemcpy(	data, host_data, prod(dims)*sizeof(TYPE), cudaMemcpyHostToDevice );

	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		data = 0x0; // Just in case we remove the 'exit' one day...
		printf("\nCuda error detected in 'DataVolumeCUDA::swapIn': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	in_host_memory = false;
	free(host_data);
	host_data = 0x0;
}

template< class TYPE > DataVolumeCUDA<TYPE, uint3, float3>* 
applyDeformation( DataVolumeCUDA<TYPE, uint3, float3> *image, DataVolumeCUDA<float4, uint3, float3> *deformations, bool unit_displacements )
{
	if( sizeof(TYPE) == sizeof(float3) ){
		printf("\nWARNING: DataVolumeCUDA<TYPE>::applyDeformation not implemented for the 'float3' type.");
		return 0x0;
	}
	else{

		// Allocate memory for result
		DataVolumeCUDA<TYPE, uint3, float3> *result = new DataVolumeCUDA<TYPE, uint3, float3>( image->dims, image->scale, image->origin );

		// Setup texture
		cudaArray *tmp_image_array = prepare_texture<TYPE>(image->dims, image->data);
		if( !tmp_image_array ){
			printf("\nImplementation immature! Returning 0x0.\n");
			return 0x0;
		}

		// Check for errors
		cudaError_t err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::applyDeformation': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		float3 ones; make_scale_vec( 1.0f, ones );

		dim3 blockDim(512,1,1);
		dim3 gridDim((unsigned int) ceil((double)prod(image->dims)/(double)blockDim.x), 1, 1 );

		if (unit_displacements)
			func_apply_deformations_kernel<TYPE><<< gridDim, blockDim >>>( deformations->data, result->data, ones, image->dims );
		else
			func_apply_deformations_kernel<TYPE><<< gridDim, blockDim >>>( deformations->data, result->data, image->scale, image->dims );

		// Release texture
		bool _impl = release_texture<TYPE>(image->dims, tmp_image_array );
		if( !_impl ){
			printf("\nImplementation inconsistent! Returning 0x0.\n");
			return 0x0;
		}
		tmp_image_array = 0x0;

		// Check for errors
		err = cudaGetLastError();
		if( err != cudaSuccess ){
			printf("\nCuda error detected in 'DataVolumeCUDA::applyDeformation': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		return result;
	}
}


// Template instantiation

template class DataVolumeCUDA<float, unsigned int, float>;
template class DataVolumeCUDA<float2, unsigned int, float>;
//template class DataVolumeCUDA<float3, unsigned int, float>;
template class DataVolumeCUDA<float4, unsigned int, float>;

template class DataVolumeCUDA<float, uint2, float2>;
template class DataVolumeCUDA<float2, uint2, float2>;
//template class DataVolumeCUDA<float3, uint2, float2>;
template class DataVolumeCUDA<float4, uint2, float2>;

template class DataVolumeCUDA<float, uint3, float3>;
template class DataVolumeCUDA<float2, uint3, float3>;
//template class DataVolumeCUDA<float3, uint3, float3>;
template class DataVolumeCUDA<float4, uint3, float3>;

template class DataVolumeCUDA<float, uint4, float4>;
template class DataVolumeCUDA<float2, uint4, float4>;
//template class DataVolumeCUDA<float3, uint4, float4>;
template class DataVolumeCUDA<float4, uint4, float4>;

template DataVolumeCUDA<float, uint3, float3>* applyDeformation( DataVolumeCUDA<float, uint3, float3>*, DataVolumeCUDA<float4, uint3, float3>*, bool );
template DataVolumeCUDA<float2, uint3, float3>* applyDeformation( DataVolumeCUDA<float2, uint3, float3>*, DataVolumeCUDA<float4, uint3, float3>*, bool );
//template DataVolumeCUDA<float3, uint3, float3>* applyDeformation( DataVolumeCUDA<float3, uint3, float3>*, DataVolumeCUDA<float4, uint3, float3>*, bool );
template DataVolumeCUDA<float4, uint3, float3>* applyDeformation( DataVolumeCUDA<float4, uint3, float3>*, DataVolumeCUDA<float4, uint3, float3>*, bool );

