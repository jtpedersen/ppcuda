#ifndef DWTIMG_H_
#define DWTIMG_H_

void test_img(const char* file, float);


void simple_decompose(float * d_idata, float *d_odata, int);
void simple_recompose(float * d_idata, float *d_odata, int);

void optimized_decompose(float *d_idata, float *d_odata, int levels, int img_w, int img_h);

void from_grayscale_floats_to_ppm(const char *filename, float *d_odata, int img_w, int img_h);

__global__
void simple_copy_kernel(int *in, int *out);


__global__
void to_grayscale_floats(int *in, float *out, int size);
__global__
void from_grayscale_floats(float *in, int* out, int size);




__global__ void 
dwtHaar2D_row( float* id, float* od,
          const unsigned int dlevels,
          const unsigned int slength_step_half,
	   const int bdim );



__global__ void 
dwtHaar2D_col( float* id, float* od, 
          const unsigned int dlevels,
          const unsigned int slength_step_half,
	   const int bdim );

__global__
void twdHaar2D_row( float* id, float* od,
	   const unsigned int dlevels,
	   const unsigned int slength_step_half,
		    const int bdim ) ;

__global__
void twdHaar2D_col( float* id, float* od,
	   const unsigned int dlevels,
	   const unsigned int slength_step_half,
		    const int bdim ) ;


__device__
void recomposeStep(float *D, float *out, int half_step);
__global__
void recompose(float *D, float * out, int levels);
__global__
void decomposition(float *C, float * out, int levels);
__device__
void decompositionStep(float *C, float *out, int half_step);






__global__
void clamp_kernel(float *in, float *out, float clamp);

void clamp_function(float *data, int size, float clamp);
void reconstruct(float *d_idata, float *d_odata, int levels, int img_w, int img_h);




#endif /* !DWTIMG_H_ */
