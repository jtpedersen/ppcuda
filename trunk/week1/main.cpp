#include "DataVolumeCUDA.h"
#include "SimpleMHDLoader.h"
#include "Thresholding.h"

int main(int argc, char** argv)
{
	// load a 3D image containing a thorax CT acquisition
	DataVolumeCUDA<float, uint3, float3> * image = readImageFromMetaImage<short,float>("data/20-P.mhd", "data/20-P.raw");

	// do a thresholding of the image in which voxels with an intensity in Hounsfield Units (HU)
	// of less than 400 are replaced by '0' and '255' otherwise
	// This should segment out bone morphology
	binary_threshold( image, 250.0f, 0.0f, 255.0f );

	saveShortMetaImage(image,"thresholdedOutput.mhd", "thresholdedOutput.raw");
}
