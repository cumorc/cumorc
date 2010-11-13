#ifndef _io_matlab_h_
#define _io_matlab_h_

void readMatlabfile(const char* f, int** _width, int** _height, float** _data, int& n);
void pprintMatrix(float* matrix,int width,int height);

#endif