template< class TYPEd > __global__ void
binary_threshold_kernel( TYPEd *in_out_image, uint3 dims, TYPEd threshold, TYPEd low_val, TYPEd high_val )
{
	//calculate a unique index into image (device) memory for each kernel invocation 
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if( idx < prod(dims) ) // make sure we are inside the image domain
	{
		TYPEd pixel_value = in_out_image[idx];

		if (pixel_value > threshold)
			in_out_image[idx] = high_val;
		else
			in_out_image[idx] = low_val;

	}
}
