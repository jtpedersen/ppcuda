#pragma once

/*******************************************************
 *
 *   Utility functions for float vector types
 *   float2, float3, float4
 *
 ******************************************************/
#include <cmath>
#include <vector_functions.h>

/*  OPERATORS */

inline __host__ __device__ float2 operator *(float2 a, float2 b)
{
	return make_float2(a.x*b.x, a.y*b.y);
}

inline __host__ __device__ float3 operator *(float3 a, float3 b)
{
	return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline __host__ __device__ float4 operator *(float4 a, float4 b)
{
	return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

inline __host__ __device__ float2 operator *(float f, float2 v)
{
	return make_float2(v.x*f, v.y*f);
}

inline __host__ __device__ float3 operator *(float f, float3 v)
{
	return make_float3(v.x*f, v.y*f, v.z*f);
}

inline __host__ __device__ float4 operator *(float f, float4 v)
{
	return make_float4(v.x*f, v.y*f, v.z*f, v.w*f);
}

inline __host__ __device__ float2 operator *(float2 v, float f)
{
	return make_float2(v.x*f, v.y*f);
}

inline __host__ __device__ float3 operator *(float3 v, float f)
{
	return make_float3(v.x*f, v.y*f, v.z*f);
}

inline __host__ __device__ float4 operator *(float4 v, float f)
{
	return make_float4(v.x*f, v.y*f, v.z*f, v.w*f);
}

inline __host__ __device__ float2 operator +(float2 a, float2 b)
{
	return make_float2(a.x+b.x, a.y+b.y);
}

inline __host__ __device__ float3 operator +(float3 a, float3 b)
{
	return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __host__ __device__ float4 operator +(float4 a, float4 b)
{
	return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

inline __host__ __device__ void operator +=(float2 &b, float2 a)
{
	b.x += a.x;
	b.y += a.y;
}

inline __host__ __device__ void operator +=(float3 &b, float3 a)
{
	b.x += a.x;
	b.y += a.y;
	b.z += a.z;
}

inline __host__ __device__ void operator +=(float4 &b, float4 a)
{
	b.x += a.x;
	b.y += a.y;
	b.z += a.z;
	b.w += a.w;
}

inline __host__ __device__ float2 operator -(float2 a, float2 b)
{
	return make_float2(a.x-b.x, a.y-b.y);
}

inline __host__ __device__ float3 operator -(float3 a, float3 b)
{
	return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __host__ __device__ float4 operator -(float4 a, float4 b)
{
	return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

inline __host__ __device__ void operator -=(float2 & b, float2 a)
{
	b.x -= a.x;
	b.y -= a.y;
}

inline __host__ __device__ void operator -=(float3 & b, float3 a)
{
	b.x -= a.x;
	b.y -= a.y;
	b.z -= a.z;
}

inline __host__ __device__ void operator -=(float4 & b, float4 a)
{
	b.x -= a.x;
	b.y -= a.y;
	b.z -= a.z;
	b.w -= a.w;
}

inline __host__ __device__ float2 operator /(float2 a, float2 b)
{
	return make_float2(a.x/b.x, a.y/b.y);
}

inline __host__ __device__ float3 operator /(float3 a, float3 b)
{
	return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);
}

inline __host__ __device__ float4 operator /(float4 a, float4 b)
{
	return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}

inline __host__ __device__ float2 operator /(float2 a, float f)
{
	return make_float2(a.x/f, a.y/f);
}

inline __host__ __device__ float3 operator /(float3 a, float f)
{
	return make_float3(a.x/f, a.y/f, a.z/f);
}

inline __host__ __device__ float4 operator /(float4 a, float f)
{
	return make_float4(a.x/f, a.y/f, a.z/f, a.w/f);
}

inline __host__ __device__ void operator /=(float2 &b, float f)
{
	b.x /= f;
	b.y /= f;
}

inline __host__ __device__ void operator /=(float3 &b, float f)
{
	b.x /= f;
	b.y /= f;
	b.z /= f;
}

inline __host__ __device__ void operator /=(float4 &b, float f)
{
	b.x /= f;
	b.y /= f;
	b.z /= f;
	b.w /= f;
}

inline __host__ __device__ void operator /=(float2 &b, float2 a)
{
	b.x /= a.x;
	b.y /= a.y;
}

inline __host__ __device__ void operator /=(float3 &b, float3 a)
{
	b.x /= a.x;
	b.y /= a.y;
	b.z /= a.z;
}

inline __host__ __device__ void operator /=(float4 &b, float4 a)
{
	b.x /= a.x;
	b.y /= a.y;
	b.z /= a.z;
	b.w /= a.w;
}

//inline __host__ __device__ float operator[](float2 &v, unsigned int i )
//{
//	return ((float*)&v)[i]; 
//}


/* OPERATORS END */



inline __host__ __device__ float prod(float2 f)
{
  return f.x*f.y;
}

inline __host__ __device__ float prod(float3 f)
{
  return f.x*f.y*f.z;
}

inline __host__ __device__ float prod(float4 f)
{
  return f.x*f.y*f.z*f.w;
}

inline __host__ __device__ float2 float4_to_float2(float4 f)
{
  float2 val;
  val.x = f.x;
  val.y = f.y;
  return val;
}

inline __host__ __device__ float3 float4_to_float3(float4 f)
{
  float3 val;
  val.x = f.x;
  val.y = f.y;
  val.z = f.z;
  return val;
}

inline __host__ __device__ float dot( float2 a, float2 b )
{
	return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ float dot( float3 a, float3 b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float dot( float4 a, float4 b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ float2 normalize( float2 v )
{
	return v * (1.0f/sqrtf(v.x*v.x + v.y*v.y ));
}

inline __host__ __device__ float3 normalize( float3 v )
{
	return v * (1.0f/sqrtf(v.x*v.x + v.y*v.y + v.z*v.z ));
}

inline __host__ __device__ float4 normalize( float4 v )
{
	return v * (1.0f/sqrtf(v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w));
}

inline __host__ __device__ float length( float v )
{
	return fabs(v);
}

inline __host__ __device__ float length( float2 v )
{
	return sqrtf( v.x*v.x + v.y*v.y );
}

inline __host__ __device__ float length( float3 v )
{
	return sqrtf( v.x*v.x + v.y*v.y + v.z*v.z );
}

inline __host__ __device__ float length( float4 v )
{
	return sqrtf( v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w );
}

inline __host__ __device__ float squared_length( float2 v )
{
	return v.x*v.x + v.y*v.y;
}

inline __host__ __device__ float squared_length( float3 v )
{
	return v.x*v.x + v.y*v.y + v.z*v.z;
}

inline __host__ __device__ float squared_length( float4 v )
{
	return v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
}

inline __host__ __device__ float4 float2_to_float4_with_ones(float2 ui)
{
  float4 val;
  val.x = ui.x;
  val.y = ui.y;
  val.z = 1;
  val.w = 1;
  return val;
}

inline __host__ __device__ float3 float2_to_float3_with_ones(float2 ui)
{
  float3 val;
  val.x = ui.x;
  val.y = ui.y;
  val.z = 1;
  return val;
}

inline __host__ __device__ void set_last_dim(float val, float &ui)
{
	ui = val;
}

inline __host__ __device__ void set_last_dim(float val, float2 &ui)
{
	ui.y = val;
}

inline __host__ __device__ void set_last_dim(float val, float3 &ui)
{
	ui.z = val;
}

inline __host__ __device__ void set_last_dim(float val, float4 &ui)
{
	ui.w = val;
}

inline __host__ __device__ float get_last_dim(float ui)
{
  return ui;
}

inline __host__ __device__ float get_last_dim(float2 ui)
{
  return ui.y;
}

inline __host__ __device__ float get_last_dim(float3 ui)
{
  return ui.z;
}

inline __host__ __device__ float get_last_dim(float4 ui)
{
  return ui.w;
}

inline __host__ __device__ float crop_last_dim(float2 ui)
{
  return ui.x;
}

inline __host__ __device__ float2 crop_last_dim(float3 ui)
{
  return make_float2( ui.x, ui.y );
}

inline __host__ __device__ float3 crop_last_dim(float4 ui)
{
  return make_float3( ui.x, ui.y, ui.z );
}

inline __host__ __device__ void zero(float *z)
{
  *z = 0.0f;
}

inline __host__ __device__ void zero(float2 *z)
{
  z->x = 0.0f;
  z->y = 0.0f;
}

inline __host__ __device__ void zero(float3 *z)
{
  z->x = 0.0f;
  z->y = 0.0f;
  z->z = 0.0f;
}

inline __host__ __device__ void zero(float4 *z)
{
  z->x = 0.0f;
  z->y = 0.0f;
  z->z = 0.0f;
  z->w = 0.0f;
}

inline __host__ __device__ void one(float *z)
{
  *z = 1.0f;
}

inline __host__ __device__ void one(float2 *z)
{
  z->x = 1.0f;
  z->y = 1.0f;
}

inline __host__ __device__ void one(float3 *z)
{
  z->x = 1.0f;
  z->y = 1.0f;
  z->z = 1.0f;
}

inline __host__ __device__ void one(float4 *z)
{
  z->x = 1.0f;
  z->y = 1.0f;
  z->z = 1.0f;
  z->w = 1.0f;
}

inline __host__ __device__ void make_scale_vec( float val, float &tgt )
{
  tgt = val;
}

inline __host__ __device__ void make_scale_vec( float val, float2 &tgt )
{
  tgt = make_float2( val, val );
}

inline __host__ __device__ void make_scale_vec( float val, float3 &tgt )
{
  tgt = make_float3( val, val, val );
}

inline __host__ __device__ void make_scale_vec( float val, float4 &tgt )
{
  tgt = make_float4( val, val, val, val );
}

inline __host__ __device__ void crop_to_volume( unsigned int dims, float &vol )
{
	if( vol < 0 ) vol = 0.0f;
	if( vol > (float)(dims-1) ) vol = (float)(dims-1);
}

inline __host__ __device__ void crop_to_volume( uint2 dims, float2 &vol )
{
	if( vol.x < 0 ) vol.x = 0.0f;
	if( vol.x > (float)(dims.x-1) ) vol.x = (float)(dims.x-1);
	if( vol.y < 0 ) vol.y = 0.0f;
	if( vol.y > (float)(dims.y-1) ) vol.y = (float)(dims.y-1);
}

inline __host__ __device__ void crop_to_volume( uint3 dims, float3 &vol )
{
	if( vol.x < 0 ) vol.x = 0.0f;
	if( vol.x > (float)(dims.x-1) ) vol.x = (float)(dims.x-1);
	if( vol.y < 0 ) vol.y = 0.0f;
	if( vol.y > (float)(dims.y-1) ) vol.y = (float)(dims.y-1);
	if( vol.z < 0 ) vol.z = 0.0f;
	if( vol.z > (float)(dims.z-1) ) vol.z = (float)(dims.z-1);
}

inline __host__ __device__ void crop_to_volume( uint4 dims, float4 &vol )
{
	if( vol.x < 0 ) vol.x = 0.0f;
	if( vol.x > (float)(dims.x-1) ) vol.x = (float)(dims.x-1);
	if( vol.y < 0 ) vol.y = 0.0f;
	if( vol.y > (float)(dims.y-1) ) vol.y = (float)(dims.y-1);
	if( vol.z < 0 ) vol.z = 0.0f;
	if( vol.z > (float)(dims.z-1) ) vol.z = (float)(dims.z-1);
	if( vol.w < 0 ) vol.w = 0.0f;
	if( vol.w > (float)(dims.w-1) ) vol.w = (float)(dims.w-1);
}

// operators <, <=, >, >= are "strong" in the sense they require all components to fullfil the scalar operator.
// Here are the corresponding "weak" alternatives

inline __host__ __device__ bool weak_less(float2 ui1, float2 ui2)
{
  return ui1.x<ui2.x || ui1.y<ui2.y;
}

inline __host__ __device__ bool weak_less(float3 ui1, float3 ui2)
{
  return ui1.x<ui2.x || ui1.y<ui2.y || ui1.z<ui2.z;
}

inline __host__ __device__ bool weak_less(float4 ui1, float4 ui2)
{
  return ui1.x<ui2.x || ui1.y<ui2.y || ui1.z<ui2.z || ui1.w<ui2.w;
}

inline __host__ __device__ bool weak_equal(float ui1, float ui2)
{
  return ui1==ui2;
}

inline __host__ __device__ bool weak_equal(float2 ui1, float2 ui2)
{
  return ui1.x==ui2.x || ui1.y==ui2.y;
}

inline __host__ __device__ bool weak_equal(float3 ui1, float3 ui2)
{
  return ui1.x==ui2.x || ui1.y==ui2.y || ui1.z==ui2.z;
}

inline __host__ __device__ bool weak_equal(float4 ui1, float4 ui2)
{
  return ui1.x==ui2.x || ui1.y==ui2.y || ui1.z==ui2.z || ui1.w==ui2.w;
}

inline __host__ __device__ bool weak_less_equal(float2 ui1, float2 ui2)
{
  return ui1.x<=ui2.x || ui1.y<=ui2.y;
}

inline __host__ __device__ bool weak_less_equal(float3 ui1, float3 ui2)
{
  return ui1.x<=ui2.x || ui1.y<=ui2.y || ui1.z<=ui2.z;
}

inline __host__ __device__ bool weak_less_equal(float4 ui1, float4 ui2)
{
  return ui1.x<=ui2.x || ui1.y<=ui2.y || ui1.z<=ui2.z || ui1.w<=ui2.w;
}

inline __host__ __device__ bool weak_greater(float2 ui1, float2 ui2)
{
  return ui1.x>ui2.x || ui1.y>ui2.y;
}

inline __host__ __device__ bool weak_greater(float3 ui1, float3 ui2)
{
  return ui1.x>ui2.x || ui1.y>ui2.y || ui1.z>ui2.z;
}

inline __host__ __device__ bool weak_greater(float4 ui1, float4 ui2)
{
  return ui1.x>ui2.x || ui1.y>ui2.y || ui1.z>ui2.z || ui1.w>ui2.w;
}

inline __host__ __device__ bool weak_greater_equal(float ui1, float ui2)
{
  return ui1>=ui2;
}

inline __host__ __device__ bool weak_greater_equal(float2 ui1, float2 ui2)
{
  return ui1.x>=ui2.x || ui1.y>=ui2.y;
}

inline __host__ __device__ bool weak_greater_equal(float3 ui1, float3 ui2)
{
  return ui1.x>=ui2.x || ui1.y>=ui2.y || ui1.z>=ui2.z;
}

inline __host__ __device__ bool weak_greater_equal(float4 ui1, float4 ui2)
{
  return ui1.x>=ui2.x || ui1.y>=ui2.y || ui1.z>=ui2.z || ui1.w>=ui2.w;
}
