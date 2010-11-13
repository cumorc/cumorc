# (c) Ferdinand Deger, 2009, 2010
#
# This is the SConstruct, the equivalence to a make-file, all global dependencis should be in this 
# file. The SConstruct is synchronized with the repository
#


import os
import sys

var = {}
try:
  execfile('site_scons/custom.py')
except IOError:
	sys.exit("Can't locate a 'site_scons/custom.py'\n ->Please generate this file first. Examples are given")


env = Environment()

#Check for CUDA, Cuda Compiler is assigened to *.cu files
env_GCC = env.Clone()

# Append 32bit flag, if assigned so in the custom.py
env.Append(CCFLAGS = [m32])
env.Append(LDFLAGS = [m32])
env.Append(LINKFLAGS = [m32])

#Save variable persitently in environment
env['CUDA_TOOLKIT_PATH'] = CUDA_TOOLKIT_PATH
env['CUDA_SDK_PATH'] = CUDA_SDK_PATH
env['OCV_PATH'] = OCV_PATH
env['SVM_PATH'] = SVM_PATH
env['COMPILE_WEBCAM'] = COMPILE_WEBCAM
env['DEBUG'] = DEBUG
env['DTIME'] = DTIME
env['MACTIME'] = MACTIME


env.Tool('cuda')
env['NVCCFLAGS'] = m32+" --host-compilation 'C++'"
env.Append(LIBS=[CUTIL_LIBNAME, 'cudart'])

#Compiling everything with the webcam flag
if COMPILE_WEBCAM:
  env.Append(NVCCINC = ' -DWEBCAM_GUARD')


Export('var')
Export('env')
Export('env_GCC')

# call core SConscript
SConscript('cuda/SConscript')

#call webcam app SConscript
if COMPILE_WEBCAM:
    SConscript('webcam_application/SConscript')

