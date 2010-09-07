
/*
	File containing the kernels for texture volume
*/

#include "uint_utils.h"
#include "uint_utils_device.h"
#include "float_utils.h"
#include "float_utils_device.h"


#define NORMALIZED_TC 1

// 1D textures with linear interpolation
texture<float,  1, cudaReadModeElementType> source_f1_1_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeClamp );
texture<float2, 1, cudaReadModeElementType> source_f2_1_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeClamp );
texture<float4, 1, cudaReadModeElementType> source_f4_1_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeClamp );

// 2D textures with linear interpolation
texture<float,  3, cudaReadModeElementType> source_f1_2_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeClamp );
texture<float2, 3, cudaReadModeElementType> source_f2_2_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeClamp );
texture<float4, 3, cudaReadModeElementType> source_f4_2_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeClamp );

// 3D textures with linear interpolation
texture<float,  3, cudaReadModeElementType> source_f1_3_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeClamp );
texture<float2, 3, cudaReadModeElementType> source_f2_3_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeClamp );
texture<float4, 3, cudaReadModeElementType> source_f4_3_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeClamp );

// "Simulated 4D" textures with linear interpolation (except 'w' direction)
texture<float,  3, cudaReadModeElementType> source_f1_4_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeClamp );
texture<float2, 3, cudaReadModeElementType> source_f2_4_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeClamp );
texture<float4, 3, cudaReadModeElementType> source_f4_4_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeClamp );

// 1D textures from linear memory
texture<float,  1, cudaReadModeElementType> f1_1d_tex;
texture<float2, 1, cudaReadModeElementType> f2_1d_tex;
texture<float4, 1, cudaReadModeElementType> f4_1d_tex;

texture<float,  1, cudaReadModeElementType> f1_1d_tex2;
texture<float2, 1, cudaReadModeElementType> f2_1d_tex2;
texture<float4, 1, cudaReadModeElementType> f4_1d_tex2;

texture<float4, 1, cudaReadModeElementType> in_deformations_tex;


template< class TYPE > __global__ void 
magnitudes_kernel( TYPE *devPtr_in, float *devPtr_out, unsigned int number_of_elements )
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  
  if( idx<number_of_elements ){
	  TYPE in = devPtr_in[idx];
	  devPtr_out[idx] = length(in);
  }
}

template< class TYPE > __global__ void
clear_kernel( TYPE val, TYPE *imageDevPtr, unsigned int num_elements )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < num_elements ){
		imageDevPtr[idx] = val;
	}
}

template< class TYPE > __global__ void
setLastComponent_kernel( float val, TYPE *imageDevPtr, unsigned int num_elements )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < num_elements ){
		TYPE in = imageDevPtr[idx];
		set_last_dim( val, in );
		imageDevPtr[idx] = in;
	}
}

__device__ float _jacobian( unsigned int idx, unsigned int co, float scale, unsigned int dims )
{
	// FIXME
	return 0.0f;
}

__device__ float _jacobian( unsigned int idx, uint2 co, float2 scale, uint2 dims )
{
	// FIXME
	return 0.0f;
}

__device__ float _jacobian( unsigned int idx, uint3 co, float3 scale, uint3 dims )
{
	const uint3 D = make_uint3( 1, dims.x, dims.x*dims.y);
	const float3 co_f = uintd_to_floatd( co );

	const float3 Ppnn =	scale * (co_f + make_float3(1,0,0)) + crop_last_dim(tex1Dfetch( in_deformations_tex, idx + D.x ));
	const float3 Pmnn = scale * (co_f - make_float3(1,0,0)) + crop_last_dim(tex1Dfetch( in_deformations_tex, idx - D.x ));
	const float3 Pnpn = scale * (co_f + make_float3(0,1,0)) + crop_last_dim(tex1Dfetch( in_deformations_tex, idx + D.y ));
	const float3 Pnmn = scale * (co_f - make_float3(0,1,0)) + crop_last_dim(tex1Dfetch( in_deformations_tex, idx - D.y ));
	const float3 Pnnp = scale * (co_f + make_float3(0,0,1)) + crop_last_dim(tex1Dfetch( in_deformations_tex, idx + D.z ));
	const float3 Pnnm = scale * (co_f - make_float3(0,0,1)) + crop_last_dim(tex1Dfetch( in_deformations_tex, idx - D.z ));

	const float3 factors = make_float3( 1.0f/(2.0f*scale.x), 1.0f/(2.0f*scale.y), 1.0f/(2.0f*scale.z) );

	const float3 dPdx = factors.x * (Ppnn-Pmnn);
	const float3 dPdy = factors.y * (Pnpn-Pnmn);
	const float3 dPdz = factors.z * (Pnnp-Pnnm);

#define a11 dPdx.x
#define a21 dPdx.y
#define a31 dPdx.x

#define a12 dPdy.x
#define a22 dPdy.y
#define a32 dPdy.z

#define a13 dPdz.x
#define a23 dPdz.y
#define a33 dPdz.z

	return a11*(a22*a33-a32*a23) + a12*(a23*a31-a33*a21) + a13*(a21*a32-a31*a22);
}

__device__ float _jacobian( unsigned int idx, uint4 co, float4 scale, uint4 dims )
{
	// FIXME
	return 0.0f;
}

template< class FLOATd, class UINTd > __global__ void 
calculate_jacobians_kernel( float *out_jacobians, FLOATd scale, UINTd dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims) ){

		const UINTd co = idx_to_co( idx, dims );
		UINTd zeros; make_scale_vec( 0, zeros );
		UINTd ones; make_scale_vec( 1, ones );

		if( weak_equal(co, zeros) || weak_equal(co, dims-ones) ){
//		if( (co.x == 0) || (co.x == dims.x-1) || (co.y == 0) || (co.y == dims.y-1) || (co.z == 0) || (co.z == dims.z-1) ){
			out_jacobians[idx] = 1.0f;
			return;
		}

		float J = _jacobian( idx, co, scale, dims);

		out_jacobians[idx] = J;
	}
}

template<class TYPE> inline __device__ TYPE 
__get_image_val( float tc )
{
	TYPE val;

	switch(sizeof(TYPE)){
		case sizeof(float):
			{
				float tmp = tex1D( source_f1_1_tex, tc );
				val = *((TYPE*) &tmp);
			}
			break;
		case sizeof(float2):
			{
				float2 tmp = tex1D( source_f2_1_tex, tc );
				val = *((TYPE*) &tmp);
			}
			break;
		case sizeof(float4):
			{
				float4 tmp = tex1D( source_f4_1_tex, tc );
				val = *((TYPE*) &tmp);
			}
			break;
		default:
			// Not supported
			break;
	}

	return val;
}

template<class TYPE> inline __device__ TYPE 
__get_image_val( float2 _tc )
{
	TYPE val;
	float3 tc = make_float3( _tc.x, _tc.y, 0.5f );

	switch(sizeof(TYPE)){
		case sizeof(float):
			{
				float tmp = tex3D( source_f1_2_tex, tc.x, tc.y, tc.z );
				val = *((TYPE*) &tmp);
			}
			break;
		case sizeof(float2):
			{
				float2 tmp = tex3D( source_f2_2_tex, tc.x, tc.y, tc.z );
				val = *((TYPE*) &tmp);
			}
			break;
		case sizeof(float4):
			{
				float4 tmp = tex3D( source_f4_2_tex, tc.x, tc.y, tc.z );
				val = *((TYPE*) &tmp);
			}
			break;
		default:
			// Not supported
			break;
	}

	return val;
}

template<class TYPE> inline __device__ TYPE 
__get_image_val( float3 tc )
{
	TYPE val;

	switch(sizeof(TYPE)){
		case sizeof(float):
			{
				float tmp = tex3D( source_f1_3_tex, tc.x, tc.y, tc.z );
				val = *((TYPE*) &tmp);
			}
			break;
		case sizeof(float2):
			{
				float2 tmp = tex3D( source_f2_3_tex, tc.x, tc.y, tc.z );
				val = *((TYPE*) &tmp);
			}
			break;
		case sizeof(float4):
			{
				float4 tmp = tex3D( source_f4_3_tex, tc.x, tc.y, tc.z );
				val = *((TYPE*) &tmp);
			}
			break;
		default:
			// Not supported
			break;
	}

	return val;
}

template<class TYPE> inline __device__ TYPE 
__get_image_val( float4 tc )
{
	TYPE val;

	switch(sizeof(TYPE)){
		case sizeof(float):
			{
				float tmp = tex3D( source_f1_4_tex, tc.x, tc.y, tc.z );
				val = *((TYPE*) &tmp);
			}
			break;
		case sizeof(float2):
			{
				float2 tmp = tex3D( source_f2_4_tex, tc.x, tc.y, tc.z );
				val = *((TYPE*) &tmp);
			}
			break;
		case sizeof(float4):
			{
				float4 tmp = tex3D( source_f4_4_tex, tc.x, tc.y, tc.z );
				val = *((TYPE*) &tmp);
			}
			break;
		default:
			// Not supported
			break;
	}

	return val;
}

/*
inline __device__ void __get_image_val( float &val, float tc )
{
	val = tex1D( source_f1_1_tex, tc );
}

inline __device__ void __get_image_val( float2 &val, float tc )
{
	val = tex1D( source_f2_1_tex, tc );
}

inline __device__ void __get_image_val( float3 &val, float tc )
{
	// Not used
}

inline __device__ void __get_image_val( float4 &val, float tc )
{
	val = tex1D( source_f4_1_tex, tc );
}

inline __device__ void __get_image_val( float &val, float2 tc )
{
	val = tex3D( source_f1_2_tex, tc.x, tc.y, 0.5f );
}

inline __device__ void __get_image_val( float2 &val, float2 tc )
{
	val = tex3D( source_f2_2_tex, tc.x, tc.y, 0.5f );
}

inline __device__ void __get_image_val( float3 &val, float2 tc )
{
	// Not used
}

inline __device__ void __get_image_val( float4 &val, float2 tc )
{
	val = tex3D( source_f4_2_tex, tc.x, tc.y, 0.5f );
}

inline __device__ void __get_image_val( float &val, float3 tc )
{
	val = tex3D( source_f1_3_tex, tc.x, tc.y, tc.z );
}

inline __device__ void __get_image_val( float2 &val, float3 tc )
{
	val = tex3D( source_f2_3_tex, tc.x, tc.y, tc.z );
}

inline __device__ void __get_image_val( float3 &val, float3 tc )
{
	// Not used
}

inline __device__ void __get_image_val( float4 &val, float3 tc )
{
	val = tex3D( source_f4_3_tex, tc.x, tc.y, tc.z );
}

inline __device__ void __get_image_val( float &val, float4 tc )
{
	val = tex3D( source_f1_4_tex, tc.x, tc.y, tc.z );
}

inline __device__ void __get_image_val( float2 &val, float4 tc )
{
	val = tex3D( source_f2_4_tex, tc.x, tc.y, tc.z );
}

inline __device__ void __get_image_val( float3 &val, float4 tc )
{
	// Not used
}

inline __device__ void __get_image_val( float4 &val, float4 tc )
{
	val = tex3D( source_f4_4_tex, tc.x, tc.y, tc.z );
}
*/

inline __device__ void __get_val_1d( float *val, int index )
{
	*val = tex1Dfetch(  f1_1d_tex, index );
}

inline __device__ void __get_val_1d( float2 *val, int index )
{
	*val = tex1Dfetch(  f2_1d_tex, index );
}

inline __device__ void __get_val_1d( float3 *val, int index )
{
	// Not used
}

inline __device__ void __get_val_1d( float4 *val, int index )
{
	*val = tex1Dfetch(  f4_1d_tex, index );
}

inline __device__ void __get_val_1d2( float *val, int index )
{
	*val = tex1Dfetch(  f1_1d_tex2, index );
}

inline __device__ void __get_val_1d2( float2 *val, int index )
{
	*val = tex1Dfetch(  f2_1d_tex2, index );
}

inline __device__ void __get_val_1d2( float3 *val, int index )
{
	// Not used
}

inline __device__ void __get_val_1d2( float4 *val, int index )
{
	*val = tex1Dfetch(  f4_1d_tex2, index );
}

template< class TYPE > inline __device__ void 
_apply_deformations_kernel( float *in_deformations, TYPE *out_image, float scale, unsigned int dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims) ){

		float half_vec = 0.5f;
		float one_vec = 1.0f;

		const unsigned int co = idx_to_co( idx, dims );
		const float co_f =  half_vec + uintd_to_floatd( co );

		const float mm_deformation = in_deformations[idx];
		const float voxel_deformation = mm_deformation / scale;

		const float D = one_vec / uintd_to_floatd( dims );
		const float tc = co_f + voxel_deformation;
		const float tc_n = tc * D;

		// Texture lookup
		TYPE val = __get_image_val<TYPE>( tc_n );
		
		// Output
		out_image[idx] = val;

	}
}

template< class TYPE > inline __device__ void 
_apply_deformations_kernel( float2 *in_deformations, TYPE *out_image, float2 scale, uint2 dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims) ){

		float2 half_vec; make_scale_vec(0.5f, half_vec);
		float2 one_vec; make_scale_vec(1.0f, one_vec);

		const uint2 co = idx_to_co( idx, dims );
		const float2 co_f =  half_vec + uintd_to_floatd( co );

		const float2 mm_deformation = in_deformations[idx];
		const float2 voxel_deformation = mm_deformation / scale;

		float2 D = one_vec / uintd_to_floatd( dims );
		const float2 tc = co_f + voxel_deformation;
		const float2 tc_n = tc * D;

		// Texture lookup
		TYPE val =__get_image_val<TYPE>( tc_n );
		
		// Output
		out_image[idx] = val;
	}
}

template< class TYPE > inline __device__ void 
_apply_deformations_kernel( float3 *in_deformations, TYPE *out_image, float3 scale, uint3 dims )
{
}

template< class TYPE > inline __device__ void 
_apply_deformations_kernel( float4 *in_deformations, TYPE *out_image, float4 scale, uint4 dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims) ){

		float4 half_vec; make_scale_vec(0.5f, half_vec);
		float4 one_vec; make_scale_vec(1.0f, one_vec);

		const uint4 co = idx_to_co( idx, dims );
		const float4 co_f = uintd_to_floatd( co );

		const float4 mm_deformation = in_deformations[idx];
		const float4 voxel_deformation = mm_deformation / scale;

		float4 tc = co_f + voxel_deformation; tc.z += (tc.w*uint2float(dims.z)); tc += half_vec;
		float4 D = one_vec / make_float4(uint2float(dims.x),uint2float(dims.y),uint2float(dims.z*dims.w),1.0f ); D.w = 0.0f;
		const float4 tc_n = tc * D;

		// Texture lookup
		TYPE val = __get_image_val<TYPE>( tc_n );
		
		// Output
		out_image[idx] = val;
	}
}

template<class TYPE, class UINTd, class FLOATd> 
__global__ void apply_deformations_kernel( FLOATd *in_deformations, TYPE *out_image, FLOATd scale, UINTd dims )
{
	// CUDA does not support float3 textures
	if(sizeof(TYPE)==sizeof(float3)) return;

	_apply_deformations_kernel<TYPE>( in_deformations, out_image, scale, dims );
}

template< class TYPE >
__global__ void func_apply_deformations_kernel( float4 *in_deformations, TYPE *out_image, float3 scale, uint3 dims )
{
	// CUDA does not support float3 textures
	if(sizeof(TYPE)==sizeof(float3)) return;

	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims) ){

		float3 half_vec; make_scale_vec(0.5f, half_vec);
		float3 one_vec; make_scale_vec(1.0f, one_vec);

		const uint3 co = idx_to_co( idx, dims );
		const float3 co_f =  half_vec + uintd_to_floatd( co );

		const float3 mm_deformation = crop_last_dim(in_deformations[idx]);
		const float3 voxel_deformation = mm_deformation / scale;

		float3 D = one_vec / uintd_to_floatd( dims );
		const float3 tc = co_f + voxel_deformation;
		const float3 tc_n = tc * D;

		// Texture lookup
		TYPE val = __get_image_val<TYPE>( tc_n );
		
		// Output
		out_image[idx] = val;
	}
}

__device__ void 
_concatenate_deformation_kernel( float *in_deformations, float *out_image, float scale, unsigned int dims )
{
	// Fill me in
}

__device__ void 
_concatenate_deformation_kernel( float *in_deformations, float *out_image, float2 scale, uint2 dims )
{
	// Ill defined
}

__device__ void 
_concatenate_deformation_kernel( float *in_deformations, float *out_image, float3 scale, uint3 dims )
{
	// Ill defined
}

__device__ void 
_concatenate_deformation_kernel( float *in_deformations, float *out_image, float4 scale, uint4 dims )
{
	// Ill defined
}

__device__ void 
_concatenate_deformation_kernel( float2 *in_deformations, float2 *out_image, float scale, unsigned int dims )
{
	// Ill defined
}

__device__ void 
_concatenate_deformation_kernel( float2 *in_deformations, float2 *out_image, float2 scale, uint2 dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims) ){

		float2 half_vec; make_scale_vec(0.5f, half_vec);
		float2 one_vec; make_scale_vec(1.0f, one_vec);

		const uint2 co = idx_to_co(idx, dims);
		const float2 co_f =  half_vec + uintd_to_floatd(co);

		const float2 mm_deformation = in_deformations[idx];
		const float2 voxel_deformation = mm_deformation / scale;

		const float2 D = one_vec / uintd_to_floatd(dims);
		const float2 tc = co_f + voxel_deformation;
		const float2 tc_n = tc * D;

		// Texture lookup
		float2 val = __get_image_val<float2>( tc_n );

		// Output
		out_image[idx] = val + mm_deformation;
	}
}

__device__ void 
_concatenate_deformation_kernel( float2 *in_deformations, float2 *out_image, float3 scale, uint3 dims )
{
	// Ill defined
}

__device__ void 
_concatenate_deformation_kernel( float2 *in_deformations, float2 *out_image, float4 scale, uint4 dims )
{
	// Ill defined
}

__device__ void 
_concatenate_deformation_kernel( float3 *in_deformations, float3 *out_image, float scale, unsigned int dims )
{
	// Ill defined
}

__device__ void 
_concatenate_deformation_kernel( float3 *in_deformations, float3 *out_image, float2 scale, uint2 dims )
{
	// Ill defined
}

__device__ void 
_concatenate_deformation_kernel( float3 *in_deformations, float3 *out_image, float3 scale, uint3 dims )
{
	// Not used - no float3 texture support
}

__device__ void 
_concatenate_deformation_kernel( float3 *in_deformations, float3 *out_image, float4 scale, uint4 dims )
{
	// Ill defined
}

__device__ void 
_concatenate_deformation_kernel( float4 *in_deformations, float4 *out_image, float scale, unsigned int dims )
{
	// Ill defined
}

__device__ void 
_concatenate_deformation_kernel( float4 *in_deformations, float4 *out_image, float2 scale, uint2 dims )
{
	// Ill defined
}

__device__ void 
_concatenate_deformation_kernel( float4 *in_deformations, float4 *out_image, float3 scale, uint3 dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims) ){

		float3 half_vec; make_scale_vec(0.5f, half_vec);
		float3 one_vec; make_scale_vec(1.0f, one_vec);

		const uint3 co = idx_to_co(idx, dims);
		const float3 co_f =  half_vec + uintd_to_floatd(co);

		const float4 mm_deformation = in_deformations[idx];
		const float3 voxel_deformation = crop_last_dim(mm_deformation) / scale;

		float3 D = one_vec / uintd_to_floatd(dims);
		const float3 tc = co_f + voxel_deformation;
		const float3 tc_n = tc * D;

		// Texture lookup
		float4 val = __get_image_val<float4>( tc_n );

		// Output
		out_image[idx] = val + mm_deformation;
	}
}

__device__ void 
_concatenate_deformation_kernel( float4 *in_deformations, float4 *out_image, float4 scale, uint4 dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims) ){

		float4 half_vec; make_scale_vec(0.5f, half_vec);
		float4 one_vec; make_scale_vec(1.0f, one_vec);

		const uint4 co = idx_to_co(idx, dims);
		const float4 co_f = uintd_to_floatd(co);

		const float4 mm_deformation = in_deformations[idx];
		const float4 voxel_deformation = mm_deformation / scale;

		float4 tc = co_f + voxel_deformation; tc.z += (tc.w*uint2float(dims.z)); tc += half_vec;
		float4 D = one_vec / make_float4(uint2float(dims.x),uint2float(dims.y),uint2float(dims.z*dims.w),1.0f ); D.w = 0.0f;
		float4 tc_n = tc * D;

		// Texture lookup
		float4 val = __get_image_val<float4>( tc_n );

		// Output
		out_image[idx] = val + mm_deformation;
	}
}

template<class TYPE, class UINTd, class FLOATd> 
__global__ void concatenate_deformation_kernel( TYPE *in_deformations, TYPE *out_image, FLOATd scale, UINTd dims )
{
	// CUDA does not support float3 textures
	if(sizeof(TYPE)==sizeof(float3)) return;

	_concatenate_deformation_kernel( in_deformations, out_image, scale, dims );
}

template< class TYPE > inline __device__ void 
_upsample_kernel( TYPE* out_result, unsigned int dims_upsampled, unsigned int factors, unsigned int orig_image_dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims_upsampled) ){

		const unsigned int co = idx_to_co( idx, dims_upsampled );
		float half_vec; make_scale_vec(0.5f, half_vec);
		float one_vec; make_scale_vec(1.0f, one_vec);
		const float co_in_orig = uintd_to_floatd( co )/uintd_to_floatd(factors)+half_vec;
		const float D = one_vec / uintd_to_floatd( orig_image_dims );
		const float tc_n = co_in_orig * D;

		// Texture lookup
		TYPE val = __get_image_val<TYPE>( tc_n );
		
		// Output
		out_result[idx] = val;
	}
}

template< class TYPE > inline __device__ void 
_upsample_kernel( TYPE* out_result, uint2 dims_upsampled, uint2 factors, uint2 orig_image_dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims_upsampled) ){

		const uint2 co = idx_to_co( idx, dims_upsampled );
		float2 half_vec; make_scale_vec(0.5f, half_vec);
		float2 one_vec; make_scale_vec(1.0f, one_vec);
		const float2 co_in_orig = uintd_to_floatd( co )/uintd_to_floatd(factors)+half_vec;
		const float2 D = one_vec / uintd_to_floatd( orig_image_dims );
		const float2 tc_n = co_in_orig * D;

		// Texture lookup
		TYPE val = __get_image_val<TYPE>( tc_n );
		
		// Output
		out_result[idx] = val;
	}
}

template< class TYPE > inline __device__ void 
_upsample_kernel( TYPE* out_result, uint3 dims_upsampled, uint3 factors, uint3 orig_image_dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims_upsampled) ){

		const uint3 co = idx_to_co( idx, dims_upsampled );
		float3 half_vec; make_scale_vec(0.5f, half_vec);
		float3 one_vec; make_scale_vec(1.0f, one_vec);
		const float3 co_in_orig = uintd_to_floatd( co )/uintd_to_floatd(factors);
		const float3 D = one_vec / uintd_to_floatd( orig_image_dims );
		const float3 tc_n = co_in_orig * D;

		// Texture lookup
		TYPE val = __get_image_val<TYPE>( tc_n );
		
		// Output
		out_result[idx] = val;
	}
}

template< class TYPE > inline __device__ void 
_upsample_kernel( TYPE* out_result, uint4 dims_upsampled, uint4 factors, uint4 orig_image_dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims_upsampled) ){

		const uint4 co = idx_to_co( idx, dims_upsampled );
		float4 half_vec; make_scale_vec(0.5f, half_vec);
		float4 one_vec; make_scale_vec(1.0f, one_vec);
		const float4 co_in_orig = uintd_to_floatd( co )/uintd_to_floatd(factors);

		float4 D = one_vec / make_float4(uint2float(orig_image_dims.x),uint2float(orig_image_dims.y),uint2float(orig_image_dims.z*orig_image_dims.w),1.0f ); D.w = 0.0f;
		float4 tc = co_in_orig; tc.z += (tc.w*uint2float(orig_image_dims.z)); tc += half_vec;
		float4 tc_n = tc * D;

		// Texture lookup
		TYPE val = __get_image_val<TYPE>( tc_n );
		
		// Output
		out_result[idx] = val;
	}
}

template <class TYPE, class UINTd, class FLOATd> __global__ void 
upsample_kernel( TYPE* out_result, UINTd dims_upsampled, UINTd factors, UINTd orig_image_dims )
{
	// CUDA does not support float3 textures
	if(sizeof(TYPE)==sizeof(float3)) return;

	_upsample_kernel( out_result, dims_upsampled, factors, orig_image_dims );
}

inline __device__ void __add_image_val( float *val, unsigned int idx )
{
	*val += tex1Dfetch( f1_1d_tex, idx );
}

inline __device__ void __add_image_val( float2 *val, unsigned int idx )
{
	*val += tex1Dfetch( f2_1d_tex, idx );
}

inline __device__ void __add_image_val( float3 *val, unsigned int idx )
{
	// Not used
}

inline __device__ void __add_image_val( float4 *val, unsigned int idx )
{
	*val += tex1Dfetch( f4_1d_tex, idx );
}

template <class TYPE> inline __device__
TYPE _downsample( unsigned int idx_orig, unsigned int co_orig, unsigned int dims_orig, unsigned int factors )
{
	TYPE val; zero(&val);
	unsigned int count = 1;

	const unsigned int DX = 1;

	for (int i = 0; i<factors; i++)
	{ 
		if ( (co_orig+i)<dims_orig )
		{
			__add_image_val( &val, idx_orig + i*DX );
			count++;
		}
	}

	val /= uint2float(count);
	return val;
}

template <class TYPE> inline __device__
TYPE _downsample( unsigned int idx_orig, uint2 co_orig, uint2 dims_orig, uint2 factors )
{
	TYPE val; zero(&val);
	unsigned int count = 1;

	const uint2 DX = make_uint2(1, dims_orig.x);

	for (int j = 0; j<factors.y; j++){
		for (int i = 0; i<factors.x; i++)
		{ 
			if ( (co_orig.x+i)<dims_orig.x && (co_orig.y+j)<dims_orig.y )
			{
				__add_image_val( &val, idx_orig + i*DX.x + j*DX.y );
				count++;
			}
		}
	}

	val /= uint2float(count);
	return val;
}

template <class TYPE> inline __device__
TYPE _downsample( unsigned int idx_orig, uint3 co_orig, uint3 dims_orig, uint3 factors )
{
	TYPE val; zero(&val);
	unsigned int count = 1;

	const uint3 DX = make_uint3(1, dims_orig.x, dims_orig.x*dims_orig.y);

	for (int k = 0; k<factors.z; k++){
		for (int j = 0; j<factors.y; j++)
			for (int i = 0; i<factors.x; i++)
			{ 
				if ( (co_orig.x+i)<dims_orig.x && (co_orig.y+j)<dims_orig.y  && (co_orig.z+k)<dims_orig.z )
				{
					__add_image_val( &val, idx_orig + i*DX.x + j*DX.y + k*DX.z);
					count++;
				}
			}
	}

	val /= uint2float(count);
	return val;
}

template <class TYPE> inline __device__
TYPE _downsample( unsigned int idx_orig, uint4 co_orig, uint4 dims_orig, uint4 factors )
{
	TYPE val; zero(&val);
	unsigned int count = 1;

	const uint4 DX = make_uint4(1, dims_orig.x, dims_orig.x*dims_orig.y, dims_orig.x*dims_orig.y*dims_orig.z);

	for (int l = 0; l<factors.w; l++){
		for (int k = 0; k<factors.z; k++)
			for (int j = 0; j<factors.y; j++)
				for (int i = 0; i<factors.x; i++)
				{ 
					if ( (co_orig.x+i)<dims_orig.x && (co_orig.y+j)<dims_orig.y  && (co_orig.z+k)<dims_orig.z && (co_orig.w+l)<dims_orig.w )
					{
						__add_image_val( &val, idx_orig + i*DX.x + j*DX.y + k*DX.z + l*DX.w);
						count++;
					}
				}
	}

	val /= uint2float(count);
	return val;
}

template <class TYPE, class UINTd, class FLOATd> __global__ void 
downsample_kernel( TYPE* out_result, UINTd dims_orig, UINTd dims_downsampled, UINTd factors)
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims_downsampled) ){

		const UINTd co = idx_to_co( idx, dims_downsampled );
		UINTd co_orig = co*factors; 
		unsigned int idx_orig = co_to_idx( co_orig, dims_orig );
		
		// Output
		out_result[idx] = _downsample<TYPE>( idx_orig, co_orig, dims_orig, factors );
	}
}

template <class TYPE, class UINTd> __global__ void 
offset_kernel( TYPE *in_out_displacements, TYPE scale, TYPE *in_offsets, UINTd dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims) ){

		TYPE source = in_out_displacements[idx];
		TYPE offset = in_offsets[idx];

		TYPE val = source + scale * offset;
		
		in_out_displacements[idx] = val;
	}
}


__device__ void 
_crop_deformation_kernel( float *in_out_deformations, float scale, unsigned int dims )
{
	// Fill me in
}

__device__ void 
_crop_deformation_kernel( float *in_out_deformations, float2 scale, uint2 dims )
{
	// Ill defined
}

__device__ void 
_crop_deformation_kernel( float *in_out_deformations, float3 scale, uint3 dims )
{
	// Ill defined
}

__device__ void 
_crop_deformation_kernel( float *in_out_deformations, float4 scale, uint4 dims )
{
	// Ill defined
}

__device__ void 
_crop_deformation_kernel( float2 *in_out_deformations, float scale, unsigned int dims )
{
	// Ill defined
}

__device__ void 
_crop_deformation_kernel( float2 *in_out_deformations, float2 scale, uint2 dims )
{
	// Fill me in
}

__device__ void 
_crop_deformation_kernel( float2 *in_out_deformations, float3 scale, uint3 dims )
{
	// Ill defined
}

__device__ void 
_crop_deformation_kernel( float2 *in_out_deformations, float4 scale, uint4 dims )
{
	// Ill defined
}

__device__ void 
_crop_deformation_kernel( float3 *in_out_deformations, float scale, unsigned int dims )
{
	// Ill defined
}

__device__ void 
_crop_deformation_kernel( float3 *in_out_deformations, float2 scale, uint2 dims )
{
	// Ill defined
}

__device__ void 
_crop_deformation_kernel( float3 *in_out_deformations, float3 scale, uint3 dims )
{
	// Not used - no float3 texture support
}

__device__ void 
_crop_deformation_kernel( float3 *in_out_deformations, float4 scale, uint4 dims )
{
	// Ill defined
}

__device__ void 
_crop_deformation_kernel( float4 *in_out_deformations, float scale, unsigned int dims )
{
	// Ill defined
}

__device__ void 
_crop_deformation_kernel( float4 *in_out_deformations, float2 scale, uint2 dims )
{
	// Ill defined
}

__device__ void 
_crop_deformation_kernel( float4 *in_out_deformations, float3 scale, uint3 dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims) ){

		const uint3 co = idx_to_co( idx, dims );
		const float3 co_f = uintd_to_floatd( co );

		float3 def_mm = crop_last_dim(in_out_deformations[idx]);
		float3 def_vox = def_mm/scale;

		float3 def_pos = co_f + def_vox;

		// Crop
		crop_to_volume( dims, def_pos );

		// Get cropped deformation in mm's
		def_vox = def_pos-co_f;
		def_mm = def_vox*scale;

		in_out_deformations[idx] = make_float4( def_mm.x, def_mm.y, def_mm.z, 0.0f );
	}
}

__device__ void 
_crop_deformation_kernel( float4 *in_out_deformations, float4 scale, uint4 dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims) ){

		const uint4 co = idx_to_co( idx, dims );
		const float4 co_f = uintd_to_floatd( co );

		float4 def_mm = in_out_deformations[idx];
		float4 def_vox = def_mm/scale;

		float4 def_pos = co_f + def_vox;

		// Crop
		crop_to_volume( dims, def_pos );

		// Get cropped deformation in mm's
		def_vox = def_pos-co_f;
		def_mm = def_vox*scale;

		in_out_deformations[idx] = def_mm;
	}
}

template<class TYPE, class UINTd, class FLOATd> 
__global__ void crop_deformation_kernel( TYPE *in_out_deformations, FLOATd scale, UINTd dims )
{
	// CUDA does not support float3 textures
	if(sizeof(TYPE)==sizeof(float3)) return;

	_crop_deformation_kernel( in_out_deformations, scale, dims );
}


template <class TYPE, class UINTd> __global__ void 
make_border_kernel( TYPE val, TYPE *in_out_image, UINTd dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims) ){

		const UINTd co = idx_to_co( idx, dims );

		UINTd zeros; make_scale_vec( 0, zeros );
		UINTd ones; make_scale_vec( 1, ones );

		if( weak_equal(co, zeros) || weak_equal(co, dims-ones) )
//		if( (co.x == 0) || (co.x == dims.x-1) || (co.y == 0) || (co.y == dims.y-1) || (co.z == 0) || (co.z == dims.z-1) )
			in_out_image[idx] = val;
	}
}


template< class TYPE, class UINTd > __global__ void
multiply_kernel( TYPE val, TYPE *in_out_image, UINTd dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims) )
		in_out_image[idx] = in_out_image[idx] * val;
}

template< class TYPE, class UINTd > __global__ void
add_kernel( TYPE val, TYPE *in_out_image, UINTd dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims) )
		in_out_image[idx] = in_out_image[idx] +  val;
}

template< class TYPE, class UINTd > __global__ void
add_volume_kernel( TYPE *in_image, TYPE *in_out_image, UINTd dims )
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx < prod(dims) )
		in_out_image[idx] = in_out_image[idx] + in_image[idx];
}

template< class TYPE > 
inline __device__ TYPE _interpolate(unsigned int dim, unsigned int index, float delta, TYPE *data)
{
	// FIXME
	TYPE val;
	zero(&val);
	return val;
}

template< class TYPE > 
inline __device__ TYPE _interpolate(uint2 dim, uint2 index, float2 delta, TYPE *data)
{
	// FIXME
	TYPE val;
	zero(&val);
	return val;
}

template< class TYPE > 
inline __device__ TYPE _interpolate(uint3 dim, uint3 index, float3 delta, TYPE *data)
{
	int offsetDOWN = dim.x*dim.y*(index.z);
	TYPE valueDOWN = (1-delta.x)*(1-delta.y)*data[index.y*dim.x+index.x+offsetDOWN]+
		delta.x*(1-delta.y)*data[index.y*dim.x+index.x+1+offsetDOWN]+
		delta.x*delta.y*data[(index.y+1)*dim.x+index.x+1+offsetDOWN]+
		(1-delta.x)*delta.y*data[(index.y+1)*dim.x+index.x+offsetDOWN];

	int offsetUP= dim.x*dim.y*(index.z+1);
	TYPE valueUP = (1-delta.x)*(1-delta.y)*data[index.y*dim.x+index.x+offsetUP]+
		delta.x*(1-delta.y)*data[index.y*dim.x+index.x+1+offsetUP]+
		delta.x*delta.y*data[(index.y+1)*dim.x+index.x+1+offsetUP]+
		(1-delta.x)*delta.y*data[(index.y+1)*dim.x+index.x+offsetUP];

	return (1-delta.z)*valueDOWN + delta.z*valueUP;
}

template< class TYPE > 
inline __device__ TYPE _interpolate(uint4 dim, uint4 index, float4 delta, TYPE *data)
{
	// FIXME
	TYPE val;
	zero(&val);
	return val;
}

template< class TYPE, class UINTd, class FLOATd >
inline __device__ TYPE interpolate(TYPE *data, FLOATd pos, UINTd dim)
{
	TYPE value;
	const UINTd index = floatd_to_uintd(pos);
	const FLOATd delta = pos - uintd_to_floatd(index);

	FLOATd zeros; make_scale_vec( 0.0f, zeros );
	UINTd ones; make_scale_vec( 1, ones );

	if( weak_greater_equal(pos, zeros) || weak_greater_equal(pos, uintd_to_floatd(dim-ones)) )
//  if (pos.x>=0 && pos.x<dim.x-1 && pos.y>=0 && pos.y<dim.y-1 && pos.z>=0 && pos.z<dim.z-1 )
		value = _interpolate<TYPE>(dim, index, delta, data);
	else 
		zero(&value); //set to a default zero value

	return value;
}
/*
__device__ float interpolate(float * data, float3 pos, uint3 dim);
__device__ float2 interpolate(float2 * data, float3 pos, uint3 dim);
__device__ float3 interpolate(float3 * data, float3 pos, uint3 dim);
__device__ float4 interpolate(float4 * data, float3 pos, uint3 dim);
*/

template< class TYPE, class UINTd, class FLOATd > __global__ void 
evaluate_linear_kernel( FLOATd *points, unsigned int num_points, TYPE* data, UINTd dims, FLOATd scale, FLOATd origin, TYPE *result_out )
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  
  if( idx<num_points ){

	  const FLOATd pos = points[idx];

	  const FLOATd indexpos = (pos-origin)/scale; 

	  const TYPE value = interpolate(data, indexpos, dims);

	  result_out[idx] = value;
  }
}
/*
template< class TYPE > __global__ void 
evaluate_linear_kernel_float4( float4 *points, unsigned int num_points, TYPE* data, uint3 dims, float3 scale, float3 origin, TYPE *result_out )
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  
  if( idx<num_points ){

	  const float3 pos = crop_last_dim(points[idx]);

	  const float3 indexpos = (pos-origin)/scale; 

	  const TYPE value = interpolate(data, indexpos, dims);

	  result_out[idx] = value;
  }
}
*/

template< class TYPE > 
inline __device__ TYPE _convolve( unsigned int co, unsigned int dims, unsigned int kdims, unsigned int start, unsigned int end, unsigned int center )
{
	// FIXME
	TYPE val;
	zero(&val);
	return val;
}

template< class TYPE > 
inline __device__ TYPE _convolve( uint2 co, uint2 dims, uint2 kdims, uint2 start, uint2 end, uint2 center )
{
	// FIXME
	TYPE val;
	zero(&val);
	return val;
}

template< class TYPE > 
inline __device__ TYPE _convolve( uint3 co, uint3 dims, uint3 kdims, uint3 start, uint3 end, uint3 center )
{
	TYPE sum;
	zero(&sum);

	for (int k = -start.z; k<=end.z; k++)
		for (int j = -start.y; j<=end.y; j++)
			for (int i = -start.x; i<=end.x; i++)
			{
				TYPE imageval;
				TYPE kernelval;

				uint3 image_pos = make_uint3(co.x + i, co.y + j, co.z + k);

				__get_val_1d(&imageval,  co_to_idx( image_pos, dims ) );

				uint3 kernel_pos = make_uint3(center.x + i, center.y + j, center.z + k);

				__get_val_1d2(&kernelval,  co_to_idx( kernel_pos, kdims ) );

				sum = sum + imageval*kernelval;
			}
	return sum;
}

template< class TYPE > 
inline __device__ TYPE _convolve( uint4 co, uint4 dims, uint4 kdims, uint4 start, uint4 end, uint4 center )
{
	// FIXME
	TYPE val;
	zero(&val);
	return val;
}

template< class TYPE, class UINTd > __global__ void 
convolution_kernel( TYPE *output, UINTd dims, UINTd kdims)
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  
  if( idx >= prod(dims) )
	  return;

  UINTd ones; make_scale_vec(1, ones);
  UINTd center = (kdims - ones)/2;
  UINTd start = center; UINTd end = center;

  const UINTd co = idx_to_co( idx, dims );
  const UINTd rest = dims - co;

  UINTd filter = dot_less( co, start );
  UINTd inv_filter = dot_greater_eq( co, start );
  start = filter*co + inv_filter*(kdims - ones)/2; 

  filter = dot_less( rest, end );
  inv_filter = dot_greater_eq( rest, end );
  end = filter*rest + inv_filter*(kdims - ones)/2;

 /*
  if (co.x < start.x) start.x = co.x;
  if (co.y < start.y) start.y = co.y;
  if (co.z < start.z) start.z = co.z;

  if (rest.x < end.x) end.x = rest.x;
  if (rest.y < end.y) end.y = rest.y;
  if (rest.z < end.z) end.z = rest.z;
*/

  output[idx] = _convolve<TYPE>( co, dims, kdims, start, end, center );
}