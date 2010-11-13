# (c) Ferdinand Deger, 2009
#
# This is file is for customize, local paths. Do not add to svn repository
#
#



# cutil lib has different names -> check this in the cuda SDK
# Path on MacOS: /Developer/GPUComputing/C/lib
CUTIL_LIBNAME = 'cutil_i386'

# only set Cuda paths,when autodetection fails!
CUDA_TOOLKIT_PATH = "/usr/local/cuda"
CUDA_SDK_PATH = "/Developer/GPUComputing/C"

# Force a 32bit compilation / caution 64bit is not tested
m32 = "-m32"

# Compile webcam
COMPILE_WEBCAM = True

# Path to Open CV directory -> necessary for webcam app only
OCV_PATH="/usr/local"

# Path to libsvm
SVM_PATH = "3rdparty/libsvm-2.9"

# Debug informations 
DEBUG = False

# Time measurement using CLOCK_MONOTONIC
DTIME = False

# using gettimeofday 
MACTIME = False


