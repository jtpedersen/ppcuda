#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil_inline.h>

#include "dwtHaar1D.h"
#include "dwtImg.h"



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
  if (argc != 2) 
    printf("give a filename\n");

    test_img(argv[1]);

  //    runTest( argc, argv);

    // cutilExit(argc, argv);

    return 0;
}
