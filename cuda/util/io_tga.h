#ifndef _io_tga_h_
#define _io_tga_h_

float* loadTGAImage(const char* filename, unsigned int* resx, unsigned int* resy, unsigned int* numchannels);
int saveTGAImage(float* floatdata, const char* filename, unsigned int resx, unsigned int resy, unsigned int numchannels=1);

//normalize Values to be min=0 and max=1  => output image e.g. Gabor is becomes more meaningfull
void normalizeForTGAOutput(float* floatdata, unsigned int resx, unsigned int resy, unsigned int numchannels=1);
 
#endif
