#include "DataVolumeCUDA.h"

template<class FilePixelType, class InternalPixelType> 
DataVolumeCUDA<InternalPixelType, uint3, float3> * readImageFromMetaImage(const char* filename, const char* filenameRAW);

template<class InternalPixelType> 
void saveFloatMetaImage(DataVolumeCUDA<InternalPixelType, uint3, float3> * image, const char* filename, const char* filenameRAW);

template<class InternalPixelType> 
void saveShortMetaImage(DataVolumeCUDA<InternalPixelType, uint3, float3> * image, const char* filename, const char* filenameRAW);

