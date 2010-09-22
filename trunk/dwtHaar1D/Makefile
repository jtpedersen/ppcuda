CUDA_INSTALL_PATH ?= /usr/local/cuda

CXX := g++
CC := gcc
LINK := g++
NVCC := nvcc -ccbin /usr/bin 


ifeq ($(debug),1) 
 COMMONFLAGS += -g
 NVCCFLAGS += -G
endif


ifeq ($(emu),1) 
 NVCCFLAGS += -deviceemu
endif


ifeq ($(release),1) 
 COMMONFLAGS += -O2
 COMMONFLAGS += -DNDEBUG
else 
  CXXFLAGS += -Wall
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
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib -lcudart #cuda api runtime
LIB_CUDA += -L../lib -lcutil_i386    #cutil
LIB_CUDA += -lGL -lGLU -lGLEW -lglut #glut mm

OBJS =  dwtHaar1D.cu dwt1d_main.cpp dwtImg.cu

BUILD_OBJS =$(addprefix $(BUILDDIR)/, $(addsuffix .o, $(OBJS)))
LINKLINE = $(LINK) $(BUILD_OBJS) $(LIB_CUDA)

all: dwt1d

.SUFFIXES: .c .cpp .cu .o

$(BUILDDIR)/%.c.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.cu.o:$(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILDDIR)/%.cpp.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

dwt1d: clean $(BUILD_OBJS) 
	$(LINKLINE) -o$(BINDIR)/dwt

clean:
	rm -f $(BUILDDIR)/*.o $(BINDIR)/dwt *~