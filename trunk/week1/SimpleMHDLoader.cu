#include "DataVolumeCUDA.h"
#include "float_utils.h"
#include "uint_utils.h"

#include <stdio.h>

//
// This is a reader/writer which assumes three-dimensional input volumes
//

template<class FilePixelType, class InternalPixelType> 
DataVolumeCUDA<InternalPixelType, uint3, float3>* readImageFromMetaImage(const char* filename, const char* filenameRAW)
{
	FILE * pFile;
	pFile = fopen(filename,"r");

	char tmp[512];

	float3 origin;
	float3 scale;
	uint3 dim;

	for (int i=0; i<28; i++)
		fscanf (pFile, "%s\n", &tmp);

	fscanf (pFile, "%f %f %f", &(origin.x), &(origin.y), &(origin.z));
	printf("origin: %f %f %f\n", (origin.x), (origin.y), (origin.z));

	for (int i=0; i<10; i++)
		fscanf (pFile, "%s", &tmp);

	fscanf (pFile, "%f %f %f", &(scale.x), &(scale.y), &(scale.z));
	printf("scale: %f %f %f\n", (scale.x), (scale.y), (scale.z));

	for (int i=0; i<2; i++)
		fscanf (pFile, "%s", &tmp);

	fscanf (pFile, "%i %i %i", &(dim.x), &(dim.y), &(dim.z));
	printf("dim: %i %i %i\n", (dim.x), (dim.y), (dim.z));

	for (int i=0; i<6; i++)
		fscanf (pFile, "%s", &tmp);

	fclose (pFile);

	// create a new DataVolumeCUDA object of the correct size
	DataVolumeCUDA<InternalPixelType, uint3, float3> *image = new DataVolumeCUDA<InternalPixelType, uint3, float3>(dim, scale, origin); 

	pFile = fopen (filenameRAW, "rb");

	FilePixelType * tmpdata = new FilePixelType[dim.x*dim.y*dim.z];
	InternalPixelType * tmpdata2 = new InternalPixelType[dim.x*dim.y*dim.z];

	fread (tmpdata,2,dim.x*dim.y*dim.z,pFile);

	fclose (pFile);

	for(unsigned int i=0;i<dim.x*dim.y*dim.z;i++)
		tmpdata2[i] = (InternalPixelType)tmpdata[i];

	delete [] tmpdata;

	// transfer pixel data from the temporary buffer in host memory to device memory allocated by 
	// the constructor for DataVolumeCUDA
	cudaMemcpy( image->data, tmpdata2, prod(image->dims)*sizeof(InternalPixelType), cudaMemcpyHostToDevice );

	delete [] tmpdata2;

	// Return loaded image
	return image;
}

template<class InternalPixelType> 
void saveFloatMetaImage(DataVolumeCUDA<InternalPixelType, uint3, float3>* image, const char* filename, const char* filenameRAW)
{
	FILE * pFile;

	pFile = fopen(filename,"w");
	fprintf (pFile, "ObjectType = Image\n");
	fprintf (pFile, "NDims = 3\n");
	fprintf (pFile, "BinaryData = True\n");
	fprintf (pFile, "BinaryDataByteOrderMSB = False\n");
	fprintf (pFile, "CompressedData = False\n");
	fprintf (pFile, "TransformMatrix = 1 0 0 0 1 0 0 0 1\n");
	fprintf (pFile, "Offset = %f %f %f\n", image->origin.x, image->origin.y, image->origin.z);
	fprintf (pFile, "CenterOfRotation = 0 0 0\n");
	fprintf (pFile, "AnatomicalOrientation = RPI\n");
	fprintf (pFile, "ElementSpacing = %f %f %f\n", image->scale.x, image->scale.y, image->scale.z);
	fprintf (pFile, "DimSize = %i %i %i\n", image->dims.x, image->dims.y, image->dims.z);
	fprintf (pFile, "ElementType = MET_FLOAT\n");
	fprintf (pFile, "ElementDataFile = %s\n", filenameRAW);
	fclose(pFile);

    InternalPixelType * tmpdata = new InternalPixelType[prod(image->dims)];
	// transfer pixel data from device memory to a temporary buffer in host memory
	cudaMemcpy(tmpdata, image->data, prod(image->dims)*sizeof(InternalPixelType), cudaMemcpyDeviceToHost );

    float * tmpdata2 = new float[prod(image->dims)];
	//relying on a type cast when converting from our pixel type to float
	for (int i=0; i<prod(image->dims); i++)
		tmpdata2[i] = (float)tmpdata[i];

	pFile = fopen(filenameRAW,"wb");
	fwrite (tmpdata2, sizeof(float),  image->dims.x*image->dims.y*image->dims.z , pFile );
	fclose(pFile);

	delete tmpdata;
}


template<class InternalPixelType> 
void saveShortMetaImage(DataVolumeCUDA<InternalPixelType, uint3, float3>* image, const char* filename, const char* filenameRAW)
{
	FILE * pFile;

	pFile = fopen(filename,"w");
	fprintf (pFile, "ObjectType = Image\n");
	fprintf (pFile, "NDims = 3\n");
	fprintf (pFile, "BinaryData = True\n");
	fprintf (pFile, "BinaryDataByteOrderMSB = False\n");
	fprintf (pFile, "CompressedData = False\n");
	fprintf (pFile, "TransformMatrix = 1 0 0 0 1 0 0 0 1\n");
	fprintf (pFile, "Offset = %f %f %f\n", image->origin.x, image->origin.y, image->origin.z);
	fprintf (pFile, "CenterOfRotation = 0 0 0\n");
	fprintf (pFile, "AnatomicalOrientation = RPI\n");
	fprintf (pFile, "ElementSpacing = %f %f %f\n", image->scale.x, image->scale.y, image->scale.z);
	fprintf (pFile, "DimSize = %i %i %i\n", image->dims.x, image->dims.y, image->dims.z);
	fprintf (pFile, "ElementType = MET_SHORT\n");
	fprintf (pFile, "ElementDataFile = %s\n", filenameRAW);
	fclose(pFile);

    InternalPixelType * tmpdata = new InternalPixelType[prod(image->dims)];
	// transfer pixel data from device memory to a temporary buffer in host memory
	cudaMemcpy(tmpdata, image->data, prod(image->dims)*sizeof(InternalPixelType), cudaMemcpyDeviceToHost );

    short * tmpdata2 = new short[prod(image->dims)];
	//relying on a type cast when converting from our pixel type to short int
	for (int i=0; i<prod(image->dims); i++)
		tmpdata2[i] = (short)tmpdata[i];

	pFile = fopen(filenameRAW,"wb");
	fwrite (tmpdata2, sizeof(short),  image->dims.x*image->dims.y*image->dims.z , pFile );
	fclose(pFile);

	delete tmpdata;
}




template DataVolumeCUDA<float, uint3, float3>* readImageFromMetaImage<short, float> (const char* filename, const char* filenameRAW);
template void saveFloatMetaImage(DataVolumeCUDA<float, uint3, float3>* image, const char* filename, const char* filenameRAW);
template void saveShortMetaImage(DataVolumeCUDA<float, uint3, float3>* image, const char* filename, const char* filenameRAW);
