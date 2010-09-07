#pragma once

#include <vector_functions.h>
#include <device_functions.h>

inline __device__ float uintd_to_floatd( unsigned int a )
{
	return (float) a;
}

inline __device__ float2 uintd_to_floatd( uint2 a )
{
	return make_float2( uint2float(a.x), uint2float(a.y) );
}

inline __device__ float3 uintd_to_floatd( uint3 a )
{
	return make_float3( uint2float(a.x), uint2float(a.y), uint2float(a.z) );
}

inline __device__ float4 uintd_to_floatd( uint4 a )
{
	return make_float4( uint2float(a.x), uint2float(a.y), uint2float(a.z), uint2float(a.w) );
}
