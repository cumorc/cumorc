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
// header for S2.cu
// ==========================================================================

#include "pyramid.h"

#ifndef _layer_s2_h_
#define _layer_s2_h_

int S2_gpu2gpu(float** gpu_pointer, t_patches* patches, float** c1_pyramid, t_pyrprops* pyramidProperties);
int S2_file2gpu(float** gpu_pointer, t_patches* patches, char* inputfile, char* gaborfilters);
int S2_gpu2mem(float** mem_pointer, float* gpu_result, t_pyrprops* c1_pyrProps, int numberOfPatches);
int outputS2(float* result, t_pyrprops*  pyramidProperties,  t_patches* patches);

#endif
