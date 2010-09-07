/**********************************************************
 *
 *   Utility functions for uint vector types - DEVICE only
 *   uint2, uint3, uint4
 *
 **********************************************************/

#ifndef _UINT_UTIL_DEVICE_
#define _UINT_UTIL_DEVICE_


#include <device_functions.h>


inline __device__ uint4 floatd_to_uintd( float4 a )
{
	return make_uint4( float2uint(a.x), float2uint(a.y), float2uint(a.z), float2uint(a.w) );
}

inline __device__ uint3 floatd_to_uintd( float3 a )
{
	return make_uint3( float2uint(a.x), float2uint(a.y), float2uint(a.z) );
}

inline __device__ uint2 floatd_to_uintd( float2 a )
{
	return make_uint2( float2uint(a.x), float2uint(a.y) );
}

inline __device__ unsigned int floatd_to_uintd( float a )
{
	return float2uint(a);
}

#endif
