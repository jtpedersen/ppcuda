


// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}

// get pixel from 2D image, with clamping to border

__device__ int getPixel(int *data, int x, int y, int width, int height)
{
    x = clamp(x, 0, width-1);
    y = clamp(y, 0, height-1);

    return data[y*width+x];
}

__global__ void
cudaProcess(int* g_data, int* g_odata, int imgw, int imgh, float * device_stencil_data, int stencil_width, int stencil_height)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x*bw + tx;
    int y = blockIdx.y*bh + ty;

    // perform convolution
    float rsum = 0.0;
    float gsum = 0.0;
    float bsum = 0.0;

	for(int dy=0; dy<stencil_height; dy++) {
		for(int dx=0; dx<stencil_width; dx++) {

			int pixel = getPixel(g_data, x+dx-(stencil_width-1)/2, y+dy-(stencil_height-1)/2, imgw, imgh);

			float r = float(pixel&0xff);
			float g = float((pixel>>8)&0xff);
			float b = float((pixel>>16)&0xff);

			float stencil_value = device_stencil_data[dx+dy*stencil_width];

			rsum += r*stencil_value;
			gsum += g*stencil_value;
			bsum += b*stencil_value;
		}
	}

    g_odata[y*imgw+x] = rgbToInt(rsum, gsum, bsum);
}

