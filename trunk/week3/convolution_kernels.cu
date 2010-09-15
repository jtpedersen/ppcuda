
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

__global__ void
cudaProcessEx3(int* g_data, int* g_odata, int imgw, int imgh, float * device_stencil_data)
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

  /* sharemem for stencil */
  __shared__ float s_stencil[STENCIL_HEIGHT][STENCIL_WIDTH];


  if (tx < STENCIL_WIDTH && ty < STENCIL_HEIGHT )
    s_stencil[ty][tx] = device_stencil_data[ty*STENCIL_WIDTH + tx];
  __syncthreads();

  for(int dy=0; dy<STENCIL_HEIGHT; dy++) {
    for(int dx=0; dx<STENCIL_WIDTH; dx++) {

      int pixel = getPixel(g_data, x+dx-(STENCIL_WIDTH-1)/2, y+dy-(STENCIL_HEIGHT-1)/2, imgw, imgh);

      float r = float(pixel&0xff);
      float g = float((pixel>>8)&0xff);
      float b = float((pixel>>16)&0xff);

      float stencil_value = s_stencil[dy][dx];

      rsum += r*stencil_value;
      gsum += g*stencil_value;
      bsum += b*stencil_value;
    }
  }

  g_odata[y*imgw+x] = rgbToInt(rsum, gsum, bsum);
}


/* from NVIDEA sdk examples cudaprocessex */

/*
    2D convolution using shared memory
    - operates on 8-bit RGB data stored in 32-bit int
    - assumes kernel radius is less than or equal to block size
    - not optimized for performance
     _____________
    |   :     :   |
    |_ _:_____:_ _|
    |   |     |   |
    |   |     |   |
    |_ _|_____|_ _|
  r |   :     :   |
    |___:_____:___|
      r    bw   r
    <----tilew---->
*/



#define CLAMP(X, MIN, MAX) ( (X > MAX) ? MAX : (X < MIN) ? MIN : X )
#define DATA(X,Y) g_data[ CLAMP(Y, 0, imgw) *imgw + CLAMP(X,0,imgh)]

__global__ void
cudaProcessEx4(int* g_data, int* g_odata, int imgw, int imgh, float * device_stencil_data)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x*BLOCK_WIDTH + tx;
  int y = blockIdx.y*BLOCK_HEIGHT + ty;

  // perform convolution
  float rsum = 0.0;
  float gsum = 0.0;
  float bsum = 0.0;

  /* sharemem for stencil */
  __shared__ float s_stencil[STENCIL_HEIGHT][STENCIL_WIDTH];
  __shared__ int tile[(2+BLOCK_HEIGHT)][(2+BLOCK_WIDTH)];

  if (tx < STENCIL_WIDTH && ty < STENCIL_HEIGHT )
    s_stencil[ty][tx] = device_stencil_data[ty*STENCIL_WIDTH + tx];
  /* read ze tile */

  tile[ty+1][tx+1] = DATA(x, y);

  //borders
  if(tx == 0) {
    // left
    tile[ty+1][0] = DATA( x-1, y);
    // right
    tile[ty+1][BLOCK_WIDTH+1] = DATA(x+BLOCK_WIDTH+1, y);
  }
  if (ty == 0) {
    // top
    tile[ty][tx+1] = DATA(x, y-1);
    // bottom
    tile[BLOCK_HEIGHT+1][tx+1] = DATA(x, y+BLOCK_HEIGHT+1);
  }
  
  // load corners
  if ((tx == 0 ) && (ty == 0)) {
    // tl
    tile[0][0] = DATA(x-1, y-1);
    // bl
    tile[BLOCK_HEIGHT+1][0] = DATA(x-1, y+BLOCK_HEIGHT+1);
    // tr
    tile[0][BLOCK_WIDTH+1] = DATA(x+BLOCK_WIDTH+1, y-1);
    // br
    tile[BLOCK_HEIGHT+1][BLOCK_WIDTH+1] = DATA(x+BLOCK_WIDTH+1, y+BLOCK_HEIGHT+1);
  }
  

  __syncthreads();

  for(int dy=-1; dy <= 1; dy++) {
    for(int dx=-1; dx <= 1 ; dx++) {

      int pixel = tile[1+ty+dy][1+tx+dx];

      float r = float(pixel&0xff);
      float g = float((pixel>>8)&0xff);
      float b = float((pixel>>16)&0xff);

      float stencil_value =  s_stencil[1+dy][1+dx];

      rsum += r*stencil_value;
      gsum += g*stencil_value;
      bsum += b*stencil_value;
    }
  }

  g_odata[y*imgw+x] = rgbToInt(rsum, gsum, bsum);
}
