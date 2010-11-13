################################################################################
# source files
TARGETS		:= \
  tests/pyramid/test \
  tests/S1-layer/callS1 \
  tests/C1-layer/callC1 \
  tests/patch_extract/patch_extract \
  tests/weight_patches/weight_patches \
	tests/S2-layer/callS2

	#tests/svmlearn/svmlearn \

# Cuda source files (compiled with cudacc)
CUFILES		:= \
	layers/base.cu 
	#multisvm/multisvm.cu

# cuda sources included by base.cu
CUIFILES        := \
  layers/general.cu \
  layers/pyramid.cu \
  layers/C1.cu \
  layers/S1.cu \
  layers/S2.cu \
  layers/pyramid.h \
  layers/patches.cu \
  layers/learn.cu \
  layers/pyramid_resize.cu

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= util/io_tga.cc util/io_largetga.cc util/io_matlab.cc #util/io_png.cc


# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \

CXXFLAGS += $(CXXWARN_FLAGS)

CFLAGS += $(CWARN_FLAGS)

NVCCFLAGS += -arch sm_11


################################################################################
# Rules and targets

.SUFFIXES : .cu 

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Compilers
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc 
CXX        := g++
CC         := gcc
LINK       := g++ -fPIC

# Includes
INCLUDES  += -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_DIR)/common/inc
CXXFLAGS += $(INCLUDES)
CFLAGS += $(INCLUDES)
NVCCFLAGS += $(INCLUDES)


ifeq ($(dbg),1)
	COMMONFLAGS += -g
	#NVCCFLAGS   += -D_DEBUG -deviceemu
else
	COMMONFLAGS += -O2 
	NVCCFLAGS   += --compiler-options -fno-strict-aliasing
endif

CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)
NVCCFLAGS += $(COMMONFLAGS)


LDFLAGS += -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUDA_SDK_DIR)/lib -L$(CUDA_SDK_DIR)/common/lib/$(OSLOWER) -lcudart -lcuda -lcutil 
#-lcutil -lcuda
#-lgd -lpng -lz -ljpeg -lfreetype -lm


OBJS := $(patsubst %.cc,%.o,$(CCFILES))
OBJS += $(patsubst %.c,%.o,$(CFILES))
OBJS += $(patsubst %.cu,%.o,$(CUFILES))

all: $(TARGETS)

%.o : %.cu $(CUIFILES)
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<
	$(NVCC) $(NVCCFLAGS) -o $@.ptx -ptx $<

$(TARGETS) : % : %.c $(OBJS)
	$(LINK) -o $@ $^ $(LDFLAGS) 

clean :
	rm -f $(OBJS)
	rm -f $(TARGET)
