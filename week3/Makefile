CUDA_INSTALL_PATH ?= /usr/local/cuda

CXX := g++
CC := gcc
LINK := g++
NVCC := nvcc -ccbin /usr/bin 


ifeq ($(debug),1) 
 COMMONFLAGS += -g
 NVCCFLAGS += -G
endif

ifeq ($(release),1) 
 COMMONFLAGS += -O2
 COMMONFLAGS += -DNDEBUG
else 
  COMMONFLAGS += -Wall
endif

ifeq ($(info), 1)
   NVCCFLAGS += --ptxas-options="-v"
endif

BUILDDIR = build
BINDIR = bin
SRCDIR = .

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include
INCLUDES += -I../common/inc #cutil

# Common flags
COMMONFLAGS += $(INCLUDES)
COMMONFLAGS += -g
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib -lcudart -lcublas
LIB_CUDA += -L../lib -lcutil_i386 #cutil
LIB_CUDA += -lGL -lGLU -lX11 -lGLEW -LXi -lglut #glut mm

OBJS =  convolution.cu.o main.cpp.o
BUILD_OBJS =$(addprefix $(BUILDDIR)/, $(OBJS))




LINKLINE = $(LINK) $(BUILD_OBJS) $(LIB_CUDA)

all: week3


.SUFFIXES: .c .cpp .cu .o

$(BUILDDIR)/%.c.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.cu.o:$(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILDDIR)/%.cpp.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

week3: $(BUILD_OBJS) Makefile
	$(LINKLINE) -o$(BINDIR)/w3

clean:
	rm $(BUILDDIR)/*.o $(BINDIR)/w3