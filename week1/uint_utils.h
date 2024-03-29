/*******************************************************
 *
 *   Utility functions for uint vector types
 *   uint2, uint3, uint4
 *
 ******************************************************/

#pragma once

#include <vector_functions.h>

/*  OPERATORS */

inline __host__ __device__ uint2 operator *(uint2 a, uint2 b)
{
	return make_uint2(a.x*b.x, a.y*b.y);
}

inline __host__ __device__ uint3 operator *(uint3 a, uint3 b)
{
	return make_uint3(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline __host__ __device__ uint4 operator *(uint4 a, uint4 b)
{
	return make_uint4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

inline __host__ __device__ uint2 operator *(unsigned int f, uint2 v)
{
	return make_uint2(v.x*f, v.y*f);
}

inline __host__ __device__ uint3 operator *(unsigned int f, uint3 v)
{
	return make_uint3(v.x*f, v.y*f, v.z*f);
}

inline __host__ __device__ uint4 operator *(unsigned int f, uint4 v)
{
	return make_uint4(v.x*f, v.y*f, v.z*f, v.w*f);
}

inline __host__ __device__ uint2 operator *(uint2 v, unsigned int f)
{
	return make_uint2(v.x*f, v.y*f);
}

inline __host__ __device__ uint3 operator *(uint3 v, unsigned int f)
{
	return make_uint3(v.x*f, v.y*f, v.z*f);
}

inline __host__ __device__ uint4 operator *(uint4 v, unsigned int f)
{
	return make_uint4(v.x*f, v.y*f, v.z*f, v.w*f);
}

inline __host__ __device__ uint2 operator +(uint2 a, uint2 b)
{
	return make_uint2(a.x+b.x, a.y+b.y);
}

inline __host__ __device__ uint3 operator +(uint3 a, uint3 b)
{
	return make_uint3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __host__ __device__ uint4 operator +(uint4 a, uint4 b)
{
	return make_uint4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

inline __host__ __device__ void operator +=(uint2 &b, uint2 a)
{
	b.x += a.x;
	b.y += a.y;
}

inline __host__ __device__ void operator +=(uint3 &b, uint3 a)
{
	b.x += a.x;
	b.y += a.y;
	b.z += a.z;
}

inline __host__ __device__ void operator +=(uint4 &b, uint4 a)
{
	b.x += a.x;
	b.y += a.y;
	b.z += a.z;
	b.w += a.w;
}

inline __host__ __device__ uint2 operator -(uint2 a, uint2 b)
{
	return make_uint2(a.x-b.x, a.y-b.y);
}

inline __host__ __device__ uint3 operator -(uint3 a, uint3 b)
{
	return make_uint3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __host__ __device__ uint4 operator -(uint4 a, uint4 b)
{
	return make_uint4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

inline __host__ __device__ void operator -=(uint2 & b, uint2 a)
{
	b.x -= a.x;
	b.y -= a.y;
}

inline __host__ __device__ void operator -=(uint3 & b, uint3 a)
{
	b.x -= a.x;
	b.y -= a.y;
	b.z -= a.z;
}

inline __host__ __device__ void operator -=(uint4 & b, uint4 a)
{
	b.x -= a.x;
	b.y -= a.y;
	b.z -= a.z;
	b.w -= a.w;
}

inline __host__ __device__ uint2 operator /(uint2 a, uint2 b)
{
	return make_uint2(a.x/b.x, a.y/b.y);
}

inline __host__ __device__ uint3 operator /(uint3 a, uint3 b)
{
	return make_uint3(a.x/b.x, a.y/b.y, a.z/b.z);
}

inline __host__ __device__ uint4 operator /(uint4 a, uint4 b)
{
	return make_uint4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}

inline __host__ __device__ uint2 operator /(uint2 a, unsigned int f)
{
	return make_uint2(a.x/f, a.y/f);
}

inline __host__ __device__ uint3 operator /(uint3 a, unsigned int f)
{
	return make_uint3(a.x/f, a.y/f, a.z/f);
}

inline __host__ __device__ uint4 operator /(uint4 a, unsigned int f)
{
	return make_uint4(a.x/f, a.y/f, a.z/f, a.w/f);
}

inline __host__ __device__ void operator /=(uint2 &b, unsigned int f)
{
	b.x /= f;
	b.y /= f;
}

inline __host__ __device__ void operator /=(uint3 &b, unsigned int f)
{
	b.x /= f;
	b.y /= f;
	b.z /= f;
}

inline __host__ __device__ void operator /=(uint4 &b, unsigned int f)
{
	b.x /= f;
	b.y /= f;
	b.z /= f;
	b.w /= f;
}

inline __host__ __device__ uint2 operator >>(uint2 a, unsigned int b)
{
	return make_uint2(a.x>>b, a.y>>b);
}

inline __host__ __device__ uint3 operator >>(uint3 a, unsigned int b)
{
	return make_uint3(a.x>>b, a.y>>b, a.z>>b);
}

inline __host__ __device__ uint4 operator >>(uint4 a, unsigned int b)
{
	return make_uint4(a.x>>b, a.y>>b, a.z>>b, a.w>>b );
}

inline __host__ __device__ uint2 operator <<(uint2 a, unsigned int b)
{
	return make_uint2(a.x<<b, a.y<<b);
}

inline __host__ __device__ uint3 operator <<(uint3 a, unsigned int b)
{
	return make_uint3(a.x<<b, a.y<<b, a.z<<b);
}

inline __host__ __device__ uint4 operator <<(uint4 a, unsigned int b)
{
	return make_uint4(a.x<<b, a.y<<b, a.z<<b, a.w<<b);
}

inline __host__ __device__ uint2 operator %(uint2 ui1, unsigned int ui2)
{
  return make_uint2(ui1.x%ui2, ui1.y%ui2);
}

inline __host__ __device__ uint3 operator %(uint3 ui1, unsigned int ui2)
{
  return make_uint3(ui1.x%ui2, ui1.y%ui2, ui1.z%ui2);
}

inline __host__ __device__ uint4 operator %(uint4 ui1, unsigned int ui2)
{
  return make_uint4(ui1.x%ui2, ui1.y%ui2, ui1.z%ui2, ui1.w%ui2);
}

inline __host__ __device__ uint2 operator %(uint2 ui1, uint2 ui2)
{
  return make_uint2(ui1.x%ui2.x, ui1.y%ui2.y);
}

inline __host__ __device__ uint3 operator %(uint3 ui1, uint3 ui2)
{
  return make_uint3(ui1.x%ui2.x, ui1.y%ui2.y, ui1.z%ui2.z);
}

inline __host__ __device__ uint4 operator %(uint4 ui1, uint4 ui2)
{
  return make_uint4(ui1.x%ui2.x, ui1.y%ui2.y, ui1.z%ui2.z, ui1.w%ui2.w);
}

inline __host__ __device__ bool operator <(uint2 ui1, uint2 ui2)
{
  return ui1.x<ui2.x && ui1.y<ui2.y;
}

inline __host__ __device__ bool operator <(uint3 ui1, uint3 ui2)
{
  return ui1.x<ui2.x && ui1.y<ui2.y && ui1.z<ui2.z;
}

inline __host__ __device__ bool operator <(uint4 ui1, uint4 ui2)
{
  return ui1.x<ui2.x && ui1.y<ui2.y && ui1.z<ui2.z && ui1.w<ui2.w;
}

inline __host__ __device__ bool operator <=(uint2 ui1, uint2 ui2)
{
  return ui1.x<=ui2.x && ui1.y<=ui2.y;
}

inline __host__ __device__ bool operator <=(uint3 ui1, uint3 ui2)
{
  return ui1.x<=ui2.x && ui1.y<=ui2.y && ui1.z<=ui2.z;
}

inline __host__ __device__ bool operator <=(uint4 ui1, uint4 ui2)
{
  return ui1.x<=ui2.x && ui1.y<=ui2.y && ui1.z<=ui2.z && ui1.w<=ui2.w;
}

inline __host__ __device__ bool operator >(uint2 ui1, uint2 ui2)
{
  return ui1.x>ui2.x && ui1.y>ui2.y;
}

inline __host__ __device__ bool operator >(uint3 ui1, uint3 ui2)
{
  return ui1.x>ui2.x && ui1.y>ui2.y && ui1.z>ui2.z;
}

inline __host__ __device__ bool operator >(uint4 ui1, uint4 ui2)
{
  return ui1.x>ui2.x && ui1.y>ui2.y && ui1.z>ui2.z && ui1.w>ui2.w;
}

inline __host__ __device__ bool operator >=(uint2 ui1, uint2 ui2)
{
  return ui1.x>=ui2.x && ui1.y>=ui2.y;
}

inline __host__ __device__ bool operator >=(uint3 ui1, uint3 ui2)
{
  return ui1.x>=ui2.x && ui1.y>=ui2.y && ui1.z>=ui2.z;
}

inline __host__ __device__ bool operator >=(uint4 ui1, uint4 ui2)
{
  return ui1.x>=ui2.x && ui1.y>=ui2.y && ui1.z>=ui2.z && ui1.w>=ui2.w;
}

inline __host__ __device__ bool operator ==(uint2 a, uint2 b)
{
	return (a.x==b.x && a.y==b.y);
}

inline __host__ __device__ bool operator ==(uint3 a, uint3 b)
{
	return (a.x==b.x && a.y==b.y && a.z==b.z );
}

inline __host__ __device__ bool operator ==(uint4 a, uint4 b)
{
	return (a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w );
}

/* OPERATORS END */

// operators <, <=, >, >= are "strong" in the sense they require all components to fullfil the scalar operator.
// Here are the corresponding "weak" alternatives

inline __host__ __device__ bool weak_less(uint2 ui1, uint2 ui2)
{
  return ui1.x<ui2.x || ui1.y<ui2.y;
}

inline __host__ __device__ bool weak_less(uint3 ui1, uint3 ui2)
{
  return ui1.x<ui2.x || ui1.y<ui2.y || ui1.z<ui2.z;
}

inline __host__ __device__ bool weak_less(uint4 ui1, uint4 ui2)
{
  return ui1.x<ui2.x || ui1.y<ui2.y || ui1.z<ui2.z || ui1.w<ui2.w;
}

inline __host__ __device__ bool weak_equal(unsigned int ui1, unsigned int ui2)
{
  return ui1==ui2;
}

inline __host__ __device__ bool weak_equal(uint2 ui1, uint2 ui2)
{
  return ui1.x==ui2.x || ui1.y==ui2.y;
}

inline __host__ __device__ bool weak_equal(uint3 ui1, uint3 ui2)
{
  return ui1.x==ui2.x || ui1.y==ui2.y || ui1.z==ui2.z;
}

inline __host__ __device__ bool weak_equal(uint4 ui1, uint4 ui2)
{
  return ui1.x==ui2.x || ui1.y==ui2.y || ui1.z==ui2.z || ui1.w==ui2.w;
}

inline __host__ __device__ bool weak_less_equal(uint2 ui1, uint2 ui2)
{
  return ui1.x<=ui2.x || ui1.y<=ui2.y;
}

inline __host__ __device__ bool weak_less_equal(uint3 ui1, uint3 ui2)
{
  return ui1.x<=ui2.x || ui1.y<=ui2.y || ui1.z<=ui2.z;
}

inline __host__ __device__ bool weak_less_equal(uint4 ui1, uint4 ui2)
{
  return ui1.x<=ui2.x || ui1.y<=ui2.y || ui1.z<=ui2.z || ui1.w<=ui2.w;
}

inline __host__ __device__ bool weak_greater(uint2 ui1, uint2 ui2)
{
  return ui1.x>ui2.x || ui1.y>ui2.y;
}

inline __host__ __device__ bool weak_greater(uint3 ui1, uint3 ui2)
{
  return ui1.x>ui2.x || ui1.y>ui2.y || ui1.z>ui2.z;
}

inline __host__ __device__ bool weak_greater(uint4 ui1, uint4 ui2)
{
  return ui1.x>ui2.x || ui1.y>ui2.y || ui1.z>ui2.z || ui1.w>ui2.w;
}

inline __host__ __device__ bool weak_greater_equal(unsigned int ui1, unsigned int ui2)
{
  return ui1>=ui2;
}

inline __host__ __device__ bool weak_greater_equal(uint2 ui1, uint2 ui2)
{
  return ui1.x>=ui2.x || ui1.y>=ui2.y;
}

inline __host__ __device__ bool weak_greater_equal(uint3 ui1, uint3 ui2)
{
  return ui1.x>=ui2.x || ui1.y>=ui2.y || ui1.z>=ui2.z;
}

inline __host__ __device__ bool weak_greater_equal(uint4 ui1, uint4 ui2)
{
  return ui1.x>=ui2.x || ui1.y>=ui2.y || ui1.z>=ui2.z || ui1.w>=ui2.w;
}

// Element-wise less/greater
inline __host__ __device__ unsigned int dot_less(unsigned int ui1, unsigned int ui2)
{
  return ui1<ui2;
}

inline __host__ __device__ uint2 dot_less(uint2 ui1, uint2 ui2)
{
  return make_uint2(ui1.x<ui2.x, ui1.y<ui2.y);
}

inline __host__ __device__ uint3 dot_less(uint3 ui1, uint3 ui2)
{
  return make_uint3(ui1.x<ui2.x, ui1.y<ui2.y, ui1.z<ui2.z);
}

inline __host__ __device__ uint4 dot_less(uint4 ui1, uint4 ui2)
{
  return make_uint4(ui1.x<ui2.x, ui1.y<ui2.y, ui1.z<ui2.z, ui1.w<ui2.w);
}

inline __host__ __device__ unsigned int dot_greater(unsigned int ui1, unsigned int ui2)
{
  return ui1>ui2;
}

inline __host__ __device__ uint2 dot_greater(uint2 ui1, uint2 ui2)
{
  return make_uint2(ui1.x>ui2.x, ui1.y>ui2.y);
}

inline __host__ __device__ uint3 dot_greater(uint3 ui1, uint3 ui2)
{
  return make_uint3(ui1.x>ui2.x, ui1.y>ui2.y, ui1.z>ui2.z);
}

inline __host__ __device__ uint4 dot_greater(uint4 ui1, uint4 ui2)
{
  return make_uint4(ui1.x>ui2.x, ui1.y>ui2.y, ui1.z>ui2.z, ui1.w>ui2.w);
}

inline __host__ __device__ unsigned int dot_greater_eq(unsigned int ui1, unsigned int ui2)
{
  return ui1>=ui2;
}

inline __host__ __device__ uint2 dot_greater_eq(uint2 ui1, uint2 ui2)
{
  return make_uint2(ui1.x>=ui2.x, ui1.y>=ui2.y);
}

inline __host__ __device__ uint3 dot_greater_eq(uint3 ui1, uint3 ui2)
{
  return make_uint3(ui1.x>=ui2.x, ui1.y>=ui2.y, ui1.z>=ui2.z);
}

inline __host__ __device__ uint4 dot_greater_eq(uint4 ui1, uint4 ui2)
{
  return make_uint4(ui1.x>=ui2.x, ui1.y>=ui2.y, ui1.z>=ui2.z, ui1.w>=ui2.w);
}

inline __host__ __device__ unsigned int dot_less_eq(unsigned int ui1, unsigned int ui2)
{
  return ui1<=ui2;
}

inline __host__ __device__ uint2 dot_less_eq(uint2 ui1, uint2 ui2)
{
  return make_uint2(ui1.x<=ui2.x, ui1.y<=ui2.y);
}

inline __host__ __device__ uint3 dot_less_eq(uint3 ui1, uint3 ui2)
{
  return make_uint3(ui1.x<=ui2.x, ui1.y<=ui2.y, ui1.z<=ui2.z);
}

inline __host__ __device__ uint4 dot_less_eq(uint4 ui1, uint4 ui2)
{
  return make_uint4(ui1.x<=ui2.x, ui1.y<=ui2.y, ui1.z<=ui2.z, ui1.w<=ui2.w);
}

inline __host__ __device__ unsigned int dot_equal(unsigned int ui1, unsigned int ui2)
{
  return ui1==ui2;
}

inline __host__ __device__ uint2 dot_equal(uint2 ui1, uint2 ui2)
{
  return make_uint2(ui1.x==ui2.x, ui1.y==ui2.y);
}

inline __host__ __device__ uint3 dot_equal(uint3 ui1, uint3 ui2)
{
  return make_uint3(ui1.x==ui2.x, ui1.y==ui2.y, ui1.z==ui2.z);
}

inline __host__ __device__ uint4 dot_equal(uint4 ui1, uint4 ui2)
{
  return make_uint4(ui1.x==ui2.x, ui1.y==ui2.y, ui1.z==ui2.z, ui1.w==ui2.w);
}

/* Coordinate transformation functions */

inline __host__ __device__ unsigned int co_to_idx(unsigned int co, unsigned int dim)
{
  return co;
}

inline __host__ __device__ unsigned int co_to_idx(uint2 co, uint2 dim)
{
  return co.y*dim.x + co.x;
}

inline __host__ __device__ unsigned int co_to_idx(uint3 co, uint3 dim)
{
  return co.z*dim.x*dim.y + co.y*dim.x + co.x;
}

inline __host__ __device__ unsigned int co_to_idx(uint4 co, uint4 dim)
{
  return co.w*dim.x*dim.y*dim.z + co.z*dim.x*dim.y + co.y*dim.x + co.x;
}

inline __host__ __device__ unsigned int idx_to_co(unsigned int idx, unsigned int dim)
{ 
  return idx;
}

inline __host__ __device__ uint2 idx_to_co(unsigned int idx, uint2 dim)
{
  uint2 co;
  unsigned int temp = idx;
  co.x = temp%dim.x;temp -= co.x;
  co.y = temp/(dim.x);
  
  return co;
}

inline __host__ __device__ uint3 idx_to_co(unsigned int idx, uint3 dim)
{
  uint3 co;
  unsigned int temp = idx;
  co.x = temp%dim.x;temp -= co.x;
  co.y = (temp/(dim.x))%dim.y; temp -= co.y*dim.x;
  co.z = temp/(dim.x*dim.y);
  return co;
}

inline __host__ __device__ uint4 idx_to_co(unsigned int idx, uint4 dim)
{
  uint4 co;
  unsigned int temp = idx;
  co.x = temp%dim.x;temp -= co.x;
  co.y = (temp/(dim.x))%dim.y; temp -= co.y*dim.x;
  co.z = (temp/(dim.x*dim.y))%dim.z; temp -= co.z*dim.x*dim.y;
  co.w = temp/(dim.x*dim.y*dim.z);
  
  return co;
}

inline __host__ __device__ unsigned int prod(uint2 ui)
{
  return ui.x*ui.y;
}

inline __host__ __device__ unsigned int prod(uint3 ui)
{
  return ui.x*ui.y*ui.z;
}

inline __host__ __device__ unsigned int prod(uint4 ui)
{
  return ui.x*ui.y*ui.z*ui.w;
}

inline __host__ __device__ unsigned int prod(dim3 ui)
{
  return ui.x*ui.y*ui.z;
}

inline __host__ __device__ unsigned int sum(uint2 dir)
{
	return dir.x+dir.y;
}

inline __host__ __device__ unsigned int sum( uint3 dir )
{
	return dir.x+dir.y+dir.z;
}

inline __host__ __device__ unsigned int sum( uint4 dir )
{
	return dir.x+dir.y+dir.z+dir.w;
}

inline __host__ __device__ uint2 uint4_to_uint2(uint4 ui)
{
  uint2 val;
  val.x = ui.x;
  val.y = ui.y;
  return val;
}

inline __host__ __device__ uint3 uint4_to_uint3(uint4 ui)
{
  uint3 val;
  val.x = ui.x;
  val.y = ui.y;
  val.z = ui.z;
  return val;
}

inline __host__ __device__ uint2 uint3_to_uint2(uint3 ui)
{
  uint2 val;
  val.x = ui.x;
  val.y = ui.y;
  return val;
}

inline __host__ __device__ uint2 uintd_to_uint2(uint2 ui)
{
  return ui;
}

inline __host__ __device__ uint2 uintd_to_uint2(uint3 ui)
{
  uint2 val;
  val.x = ui.x;
  val.y = ui.y;
  return val;
}

inline __host__ __device__ uint2 uintd_to_uint2(uint4 ui)
{
  uint2 val;
  val.x = ui.x;
  val.y = ui.y;
  return val;
}

inline __host__ __device__ uint3 uintd_to_uint3(uint2 ui)
{
  uint3 val;
  val.x = ui.x;
  val.y = ui.y;
  val.z = 0;
  return val;
}

inline __host__ __device__ uint3 uintd_to_uint3(uint3 ui)
{
  return ui;
}

inline __host__ __device__ uint3 uintd_to_uint3(uint4 ui)
{
  uint3 val;
  val.x = ui.x;
  val.y = ui.y;
  val.z = ui.z;
  return val;
}

inline __host__ __device__ uint4 uintd_to_uint4(uint2 ui)
{
  uint4 val;
  val.x = ui.x;
  val.y = ui.y;
  val.z = 0;
  val.w = 0;
  return val;
}

inline __host__ __device__ uint4 uintd_to_uint4(uint3 ui)
{
  uint4 val;
  val.x = ui.x;
  val.y = ui.y;
  val.z = ui.z;
  val.w = 0;
  return val;
}

inline __host__ __device__ uint4 uintd_to_uint4(uint4 ui)
{
  return ui;
}

inline __host__ __device__ uint4 uintd_to_uint4_with_ones(uint2 ui)
{
  uint4 val;
  val.x = ui.x;
  val.y = ui.y;
  val.z = 1;
  val.w = 1;
  return val;
}

inline __host__ __device__ uint4 uintd_to_uint4_with_ones(uint3 ui)
{
  uint4 val;
  val.x = ui.x;
  val.y = ui.y;
  val.z = ui.z;
  val.w = 1;
  return val;
}

inline __host__ __device__ uint4 uintd_to_uint4_with_ones(uint4 ui)
{
  return ui;
}

inline __host__ __device__ dim3 uintd_to_dim3(uint2 ui)
{
  dim3 val;
  val.x = ui.x;
  val.y = ui.y;
  val.z = 1;
  return val;
}

inline __host__ __device__ dim3 uintd_to_dim3(uint3 ui)
{
  return dim3(ui);
}

inline __host__ __device__ dim3 uintd_to_dim3(uint4 ui)
{
  dim3 val;
  val.x = ui.x;
  val.y = ui.y;
  val.z = ui.z;
  return val;
}

inline __host__ __device__ void set_last_dim(unsigned int i, unsigned int &ui)
{
  ui=i;
}

inline __host__ __device__ void set_last_dim(unsigned int i, uint2 &ui)
{
  ui.y=i;
}

inline __host__ __device__ void set_last_dim(unsigned int i, uint3 &ui)
{
  ui.z=i;
}

inline __host__ __device__ void set_last_dim(unsigned int i, uint4 &ui)
{
  ui.w=i;
}

inline __host__ __device__ unsigned int get_last_dim(unsigned int ui)
{
  return ui;
}

inline __host__ __device__ unsigned int get_last_dim(uint2 ui)
{
  return ui.y;
}

inline __host__ __device__ unsigned int get_last_dim(uint3 ui)
{
  return ui.z;
}

inline __host__ __device__ unsigned int get_last_dim(uint4 ui)
{
  return ui.w;
}

inline __host__ __device__ unsigned int crop_last_dim(uint2 ui)
{
  return ui.x;
}

inline __host__ __device__ uint2 crop_last_dim(uint3 ui)
{
  return make_uint2( ui.x, ui.y );
}

inline __host__ __device__ uint3 crop_last_dim(uint4 ui)
{
  return make_uint3( ui.x, ui.y, ui.z );
}

inline __host__ __device__ uint2 shift_down(uint2 ui)
{
  return make_uint2(ui.y, ui.x);
}

inline __host__ __device__ uint3 shift_down(uint3 ui)
{
  return make_uint3(ui.y, ui.z, ui.x);
}

inline __host__ __device__ uint4 shift_down(uint4 ui)
{
  return make_uint4(ui.y, ui.z, ui.w, ui.x);
}

template<class T> inline __host__ __device__ T shift_down(T ui, unsigned int steps)
{
   T temp = ui;
   for (unsigned int i = 0; i < steps; i++)
   {
      temp = shift_down(temp);
   }   
   return temp;
}

inline __host__ __device__ uint2 shift_up(uint2 ui)
{
  return make_uint2(ui.y, ui.x);
}

inline __host__ __device__ uint3 shift_up(uint3 ui)
{
  return make_uint3(ui.z, ui.x, ui.y);
}

inline __host__ __device__ uint4 shift_up(uint4 ui)
{
  return make_uint4(ui.w, ui.x, ui.y, ui.z);
}

template<class T> inline __host__ __device__ T shift_up(T ui, unsigned int steps)
{
   T temp = ui;
   for (unsigned int i = 0; i < steps; i++)
   {
      temp = shift_up(temp);
   }   
   return temp;
}

inline __device__ unsigned int dot( uint2 a, uint2 b )
{
	return a.x * b.x + a.y * b.y;
}

inline __device__ unsigned int dot( uint3 a, uint3 b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ unsigned int dot( uint4 a, uint4 b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ uint2 int_to_uint(int2 i)
{
  return make_uint2(i.x, i.y);
}

inline __host__ __device__ uint3 int_to_uint(int3 i)
{
  return make_uint3(i.x, i.y, i.z);
}

inline __host__ __device__ uint4 int_to_uint(int4 i)
{
  return make_uint4(i.x, i.y, i.z, i.w);
}

inline __host__ __device__ void make_scale_vec( unsigned int val, unsigned int &tgt )
{
  tgt = val;
}

inline __host__ __device__ void make_scale_vec( unsigned int val, uint2 &tgt )
{
  tgt = make_uint2( val, val );
}

inline __host__ __device__ void make_scale_vec( unsigned int val, uint3 &tgt )
{
  tgt = make_uint3( val, val, val );
}

inline __host__ __device__ void make_scale_vec( unsigned int val, uint4 &tgt )
{
  tgt = make_uint4( val, val, val, val );
}

inline __host__ __device__ bool even( unsigned int a )
{
	return (a%2==0);
}

inline __host__ __device__ bool even( uint2 a )
{
	return ((a.x%2)==0 && (a.y%2)==0);
}

inline __host__ __device__ bool even( uint3 a )
{
	return ((a.x%2)==0 && (a.y%2)==0 && (a.z%2)==0);
}

inline __host__ __device__ bool even( uint4 a )
{
	return ((a.x%2)==0 && (a.y%2)==0 && (a.z%2)==0 && (a.w%2)==0);
}

inline __host__ __device__ bool odd( unsigned int a )
{
	return (a%2==1);
}

inline __host__ __device__ bool odd( uint2 a )
{
	return ((a.x%2)==1 && (a.y%2)==1);
}

inline __host__ __device__ bool odd( uint3 a )
{
	return ((a.x%2)==1 && (a.y%2)==1 && (a.z%2)==1);
}

inline __host__ __device__ bool odd( uint4 a )
{
	return ((a.x%2)==1 && (a.y%2)==1 && (a.z%2)==1 && (a.w%2)==1);
}

inline __host__ __device__ unsigned int max_element( unsigned int a )
{
	return a;
}

inline __host__ __device__ unsigned int max_element( uint2 a )
{
	return (a.x>a.y) ? a.x : a.y;
}

inline __host__ __device__ unsigned int max_element( uint3 a )
{
	const unsigned int _tmp = (a.x>a.y) ? a.x : a.y;
	return (_tmp>a.z) ? _tmp : a.z;
}

inline __host__ __device__ unsigned int max_element( uint4 a )
{
	const unsigned int _tmp1 = (a.x>a.y) ? a.x : a.y;
	const unsigned int _tmp2 = (a.z>a.w) ? a.z : a.w;
	return (_tmp1>_tmp2) ? _tmp1 : _tmp2;
}

inline __host__ __device__ unsigned int min_element( unsigned int a )
{
	return a;
}

inline __host__ __device__ unsigned int min_element( uint2 a )
{
	return (a.x<a.y) ? a.x : a.y;
}

inline __host__ __device__ unsigned int min_element( uint3 a )
{
	const unsigned int _tmp = (a.x<a.y) ? a.x : a.y;
	return (_tmp<a.z) ? _tmp : a.z;
}

inline __host__ __device__ unsigned int min_element( uint4 a )
{
	const unsigned int _tmp1 = (a.x<a.y) ? a.x : a.y;
	const unsigned int _tmp2 = (a.z<a.w) ? a.z : a.w;
	return (_tmp1<_tmp2) ? _tmp1 : _tmp2;
}
