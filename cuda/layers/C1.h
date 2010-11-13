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


#ifndef _layer_c1_h_
#define _layer_c1_h_

#include "pyramid.h"

void calcC1_file2file(char* _infile, char* _gabor, char* _outfile);

void calcC1_file2gpu(float*** gpu_img, t_pyrprops* pyrprops, char* _filename, char* _gaborfile);

#endif

