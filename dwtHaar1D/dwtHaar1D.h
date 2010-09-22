// includes, project
//#include <cutil_inline.h>


// constants which are used in host and device code
#define             INV_SQRT_2      0.70710678118654752440f;
const unsigned int  LOG_NUM_BANKS = 4;
const unsigned int  NUM_BANKS     = 16;
// declaration, forward
void runTest( int argc, char** argv);
CUTBoolean  getLevels( unsigned int len, unsigned int* levels);
