#pragma once

#include <builtin_types.h>
#include "uint_utils.h"

//! a class wrapping a 3D volume of data in device memory
template< class TYPE, class UINTd, class FLOATd > class DataVolumeCUDA
{
public:

	DataVolumeCUDA<TYPE, UINTd, FLOATd>( UINTd dims, FLOATd scale, FLOATd origin );
	~DataVolumeCUDA<TYPE, UINTd, FLOATd>();

	/*! downsample volume to half resolution
	*/
	DataVolumeCUDA<TYPE, UINTd, FLOATd> *downsample(UINTd factors /*= scaled_ones<UINTd>(2)*/);

	/*! upsample volume to double resolution
	* default value of (0,0,0) means "calculate from input image" 
	*/
	DataVolumeCUDA<TYPE, UINTd, FLOATd> *upsample(UINTd factors /*= scaled_ones<UINTd>(2)*/, UINTd new_dims /*= scaled_ones<UINTd>(0)*/);

	/*! returns the index corresponding to the grid point co.
	*/ 
//	unsigned int getIndex( const UINTd &co );

	/*!returns buffer of the data vector lengths.
	*/
	DataVolumeCUDA<float, UINTd, FLOATd>* magnitudes();

	/*!returns buffer of Jacobians.
	*/
	float* jacobians();

	/*! forwards (normal) concatenation of vector fields
	*/
	DataVolumeCUDA<TYPE, UINTd, FLOATd>* concatenateDeformations( DataVolumeCUDA<TYPE, UINTd, FLOATd> *deformations );

	/*! backwards concatenation of vector fields
	*/
	DataVolumeCUDA<TYPE, UINTd, FLOATd>* concatenateDisplacementFields( DataVolumeCUDA<TYPE, UINTd, FLOATd> *field );

	/*! returns deformed image.
	*/
	DataVolumeCUDA<TYPE, UINTd, FLOATd>* applyDeformation( DataVolumeCUDA<FLOATd, UINTd, FLOATd> *deformations, bool unit_displacements = false );

	/*! Offset data (typically vectorfield).
	*/
	void offset( TYPE scale, TYPE *offsets );

	/*! Constrain deformation to the volume
	*/
	void cropDeformationToVolume();

	/*! Border the volume with val
	*/
	void makeBorder( const TYPE &val );

	/*! perform pointwise multiplication with a constant
	*/
	void multiply( const TYPE &val );

	/*! perform pointwise division with a constant
	*/
	void divide( const TYPE &val );

	/*! perform pointwise addition with a constant
	*/
	void add( const TYPE &val );

	/*! perform pointwise addition with volume
	*/
	void add( DataVolumeCUDA<TYPE, UINTd, FLOATd> *volume );

	/*! returns the largest vector (Euclidian norm).
	*/
	TYPE vectorMax();

	/*! returns the largest vector (Euclidian norm).
	*/
	float jacobianMin();

	/*! clears volume.
	*/
	void clear( const TYPE &val );

	/*! sets the last component of each entry
	*/
	void setLastComponent( const float val );

	/*!returns the largest value in the image.
	* FIXME: currently only implemented for float images
	*/
	TYPE max();

	/*!returns the smallest value in the image.
	* FIXME: currently only implemented for float images
	*/
	TYPE min();

	/*! evaluate positions by linear interpolation
	*/
	TYPE* evaluateInterpolated( FLOATd *points, unsigned int num_points );

	/*! normalize image so all values are between min and max
	* FIXME: currently only implemented for float images
	*/
	void normalize(TYPE min, TYPE max, TYPE old_min, TYPE old_max);

	/*! return the do a linear convolution with the a kernel
	* the kernel must have odd size in all 3 dimensions
	*/
	DataVolumeCUDA<TYPE, UINTd, FLOATd>* convolve(DataVolumeCUDA<TYPE, UINTd, FLOATd>* kernel);

	/*! copy the data from device to host memory and free device memory
	*/
	void swapOut();

	/*! copy the data from host to device memory and free host memory
	*/
	void swapIn();

public:
	TYPE *data;	//! volume data
	UINTd dims;		//! volume dimensions
	FLOATd scale;	//! grid-distances (mm).
	FLOATd origin;	//! physical position of the image origin

	TYPE *host_data;	//! volume data when in host memory

	bool in_host_memory;
};

template< class TYPE > DataVolumeCUDA<TYPE, uint3, float3>* 
applyDeformation( DataVolumeCUDA<TYPE, uint3, float3> *image, DataVolumeCUDA<float4, uint3, float3> *deformations, bool unit_displacements = false );
