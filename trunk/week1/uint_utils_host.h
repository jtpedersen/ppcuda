/**********************************************************
 *
 *   Utility functions for uint vector types - DEVICE only
 *   uint2, uint3, uint4
 *
 **********************************************************/

#pragma once

#include <vector_functions.h>

inline __host__ uint4 cast_floatd_to_uintd( float4 a )
{
	return make_uint4( (unsigned int)a.x, (unsigned int)a.y, (unsigned int)a.z, (unsigned int)a.w );
}

inline __host__ uint3 cast_floatd_to_uintd( float3 a )
{
	return make_uint3( (unsigned int)a.x, (unsigned int)a.y, (unsigned int)a.z );
}

inline __host__ uint2 cast_floatd_to_uintd( float2 a )
{
	return make_uint2( (unsigned int)a.x, (unsigned int)a.y );
}

inline __host__ unsigned int cast_floatd_to_uintd( float a )
{
	return (unsigned int)a;
}
