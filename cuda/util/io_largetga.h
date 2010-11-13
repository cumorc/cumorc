#ifndef _io_largetga_h_
#define _io_largetga_h_

float* loadLargeTGAImage(const char* filename, unsigned int* resx, unsigned int* resy, unsigned int* numchannels);
int saveLargeTGAImage(float* floatdata, const char* filename, unsigned int resx, unsigned int resy, unsigned int numchannels=1);

#endif
