#ifndef DWTIMG_H_
#define DWTIMG_H_

void test_img(const char* file, int);


void from_grayscale_floats_to_ppm(const char *filename, float *d_odata, int img_w, int img_h);

__global__
void simple_copy_kernel(int *in, int *out);


__global__
void to_grayscale_floats(int *in, float *out, int size);
__global__
void from_grayscale_floats(float *in, int* out, int size);

__global__ void 
dwtHaar2D( float* id, float* od, float* approx_final, 
          const unsigned int dlevels,
          const unsigned int slength_step_half,
	   const int bdim );





#endif /* !DWTIMG_H_ */
