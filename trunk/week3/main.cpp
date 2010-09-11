/* 
    This example demonstrates the use of CUDA/OpenGL interoperability
    to post-process an image of a 3D scene generated in OpenGL.

    The basic steps are:
    1 - render the scene to the framebuffer
    2 - copy the image to a PBO (pixel buffer object)
    3 - map this PBO so that its memory is accessible from CUDA
    4 - run CUDA to process the image, writing to memory mapped from a second PBO
    6 - copy from result PBO to a texture
    7 - display the texture

    Press space to toggle the CUDA processing on/off.
    Press 'a' to toggle animation.
*/

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#pragma warning(disable:4996)
#endif

#define MAX(a,b) ((a > b) ? a : b)

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cuda_runtime_api.h"

// includes, GL
#include <GL/glew.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cutil.h>
#include <cutil_gl_error.h>

////////////////////////////////////////////////////////////////////////////////
// constants / global variables
unsigned int window_width = 512;
unsigned int window_height = 512;
unsigned int image_width = 512;
unsigned int image_height = 512;

// pbo variables
GLuint pbo_source = 0;
GLuint pbo_dest = 0;
unsigned int size_tex_data = 0;
unsigned int num_texels = 0;
unsigned int num_values = 0;

// (offscreen) render target
GLuint tex_screen = 0;

float rotate[3];

int fpsCount = 0;        // FPS count for averaging
unsigned int frameCount = 0;
extern unsigned int timer;
int fpsLimit = 1;        // FPS limit for sampling

bool enable_cuda = true;
bool animate = true;

////////////////////////////////////////////////////////////////////////////////
extern "C" void initCuda(int argc, char **argv);
extern "C" void process(int pbo_in, int pbo_out, int width, int height, float * host_stencil_data, int kernel_width, int kernel_height);
extern "C" void pboRegister(int pbo );
extern "C" void pboUnregister(int pbo );

// declaration, forward
void runTest(int argc, char** argv);

// GL functionality
CUTBoolean initGL();
void createPBO(GLuint* pbo);
void deletePBO(GLuint* pbo);
void createTexture(GLuint* tex_name, unsigned int size_x, unsigned int size_y);
void deleteTexture(GLuint* tex);

// rendering callbacks
void display();
void idle();
void keyboard(unsigned char key, int x, int y);
void reshape(int w, int h);
void mainMenu(int i);
void runTest(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    initCuda(argc, argv);

    CUT_SAFE_CALL( cutCreateTimer( &timer));

    // Create GL context
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("CUDA OpenGL post-processing");

    // initialize GL
    if( CUTFalse == initGL()) {
        return -1;
    }

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    // create menu
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Toggle CUDA processing [ ]", ' ');
    glutAddMenuEntry("Toggle animation [a]", 'a');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    // create pbo
    createPBO(&pbo_source);
    createPBO(&pbo_dest);

    // create texture for blitting onto the screen
    createTexture(&tex_screen, image_width, image_height);

    // start rendering mainloop

	glutMainLoop();

	CUT_SAFE_CALL( cutDeleteTimer( timer));

	return EXIT_SUCCESS;
}

void computeFPS()
{
  //  frameCount++;
    fpsCount++;
//    if (fpsCount == fpsLimit-1) {
//        g_Verify = true;
//    }
    if (fpsCount == fpsLimit) {
        char fps[256];
        float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
        sprintf(fps, "CUDA convolution: %3.1f fps", ifps);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 

		 fpsLimit = (int)MAX(ifps, 1.f);

    }
}
////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
initGL()
{
    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported(
        "GL_VERSION_2_0 " 
        "GL_ARB_pixel_buffer_object "
        "GL_EXT_framebuffer_object "
		)) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return CUTFalse;
    }

    // default initialization
    glClearColor(0.5, 0.5, 0.5, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_LIGHT0);
    float red[] = { 1.0, 0.1, 0.1, 1.0 };
    float white[] = { 1.0, 1.0, 1.0, 1.0 };
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0);

    CUT_CHECK_ERROR_GL();

    return CUTTrue;
}

////////////////////////////////////////////////////////////////////////////////
//! Create PBO
////////////////////////////////////////////////////////////////////////////////
void
createPBO(GLuint* pbo)
{
    // set up vertex data parameter
    num_texels = image_width * image_height;
    num_values = num_texels * 4;
    size_tex_data = sizeof(GLubyte) * num_values;
    void *data = malloc(size_tex_data);

    // create buffer object
    glGenBuffers(1, pbo);
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);

    // buffer data
    glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
    free(data);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // attach this Buffer Object to CUDA
    pboRegister(*pbo);

    CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete PBO
////////////////////////////////////////////////////////////////////////////////
void
deletePBO(GLuint* pbo)
{
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);
    CUT_CHECK_ERROR_GL();

    *pbo = 0;
}

// render a simple 3D scene
void renderScene()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, -3.0);
    glRotatef(rotate[0], 1.0, 0.0, 0.0);
    glRotatef(rotate[1], 0.0, 1.0, 0.0);
    glRotatef(rotate[2], 0.0, 0.0, 1.0);

    glViewport(0, 0, 512, 512);

    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glutSolidTeapot(1.0);

    CUT_CHECK_ERROR_GL();
}

// copy image and process using CUDA
void processImage()
{
	float stencil[] = { -1, -1, -1,
						-1,  8, -1,
						-1, -1, -1 };
/*	float stencil[] = {0, 0, 0,
						0, 1, 0,
						0, 0, 0 };
*/
    // tell cuda we are going to get into these buffers
    pboUnregister(pbo_source);

    // activate destination buffer
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, pbo_source);

    // read data into pbo. note: use BGRA format for optimal performance
    glReadPixels(0, 0, image_width, image_height, GL_BGRA, GL_UNSIGNED_BYTE, NULL); 

    // Done : re-register for Cuda
    pboRegister(pbo_source);

    // run the Cuda kernel
    process(pbo_source, pbo_dest, image_width, image_height, stencil, 3, 3);

    // blit convolved texture onto the screen

    // download texture from PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);
    glBindTexture(GL_TEXTURE_2D, tex_screen);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 
                    image_width, image_height, 
                    GL_BGRA, GL_UNSIGNED_BYTE, NULL);

    CUT_CHECK_ERROR_GL();
}

void displayImage()
{
    // render a screen sized quad
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode( GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, window_width, window_height);

    glBegin(GL_QUADS);

    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, 0.5);

    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, 0.5);

    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, 0.5);

    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, 0.5);

    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    CUT_CHECK_ERROR_GL();

}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void
display()
{
	renderScene();

	if (enable_cuda) {
		processImage();
		displayImage();
	}

	glutSwapBuffers();

	computeFPS();
}

void idle()
{
    if (animate) {
        rotate[0] += 0.2;
        rotate[1] += 0.6;
        rotate[2] += 1.0;
    }
    glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void
keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
    case(27) :
        deletePBO(&pbo_source);
        deletePBO(&pbo_dest);
        deleteTexture(&tex_screen);
        exit( 0);
    case ' ':
        enable_cuda ^= 1;
        break;
    case 'a':
        animate ^= 1;
        break;
    }
}

void reshape(int w, int h)
{
    window_width = w;
    window_height = h;
}

void mainMenu(int i)
{
    keyboard((unsigned char) i, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
void
deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    CUT_CHECK_ERROR_GL();

    *tex = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
void
createTexture( GLuint* tex_name, unsigned int size_x, unsigned int size_y)
{
    // create a texture
    glGenTextures(1, tex_name);
    glBindTexture(GL_TEXTURE_2D, *tex_name);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // buffer data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    CUT_CHECK_ERROR_GL();
}
