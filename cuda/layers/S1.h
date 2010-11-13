// ==========================================================================
// Version 1.0
// ==========================================================================
// (C)opyright: 2010
//
//   Ulm University
//
// Creator: Ferdinand Deger, <Firstname>.<Lastname>@gmail.com
// Creator: Helmut Sedding, <Firstname>@<Lastname>.net
// ==========================================================================
// header for C1.cu
// ==========================================================================

#include "pyramid.h"

#ifndef _layer_s1_h_
#define _layer_s1_h_

int calculateS1(float*** gpu_pyramid, t_pyrprops* pyramidProperties, char* inputfile, const char* gaborfilters);
int calculateS1(float*** gpu_pyramid, t_pyrprops* pyramidProperties, float** gpuImagepyramid, const char* gaborfilters);
int outputS1(float** s1pyramid, t_pyrprops* pyramidProperties, const char* out);
int clearS1(float** s1pyramid, t_pyrprops* pyramidProperties);


#endif

