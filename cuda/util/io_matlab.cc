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
// Reads the proprietary binary file created by maltlab(c)
// the structure is: <int> 4 Byt: Nr of arrays
// For each array: 2 <int> 8 Byte: width height
// For each array: n <float>, where n=width*heigh, contains the data
// 
//
// The matlab file looks like this:
// fid = fopen('Gabor4.dat','w');

// find(fSiz==s)
// number = find(fSiz==s);
// fwrite(fid,numberOfGabor,'int');
// for i=1:numberOfGabor
//    fwrite(fid,s,'int');
//    fwrite(fid,s,'int');
//    gabor(i,1:s,1:s) = reshape(filters(1:s*s,number(i)),s,s);   
//    for a=1:s
//        for b=1:s
//            fwrite(fid,gabor(i,a,b),'single');
//        end
//    end
//end
//fclose(fid);

// ==========================================================================


// reading a complete binary file
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>

#include "io_matlab.h"
using namespace std;


void readMatlabfile(const char* f, int** _width, int** _height, float** _data, int& n){
  //n is Numbers of arrays to be read
  int w; //width of the next array
  int h; // height of the next array
  //printf("Reading file: %s\n",f);
  ifstream file;
  if (file.is_open()){
  printf("File is open");
  file.close();
  }
  file.open(f, ios::in|ios::binary|ios::ate); 

  if (file.is_open())
  {   
    //Go to the end of the file and calc global size
    file.seekg (0, ios::end);
    int size = file.tellg();
    file.seekg (0, ios::beg);
    char* t = new char[sizeof(int)];

    file.read (t, sizeof(int));
    n = *((int*)t);

    int number_of_float = (size-((2*n+1)*sizeof(int)))/sizeof(float);

    // init arrays
    *(_width)  = new int[n];
    *(_height) = new int[n];
    *(_data)   = new float[number_of_float];
    float* ptr_data = &(*(_data)[0]);
    
    for(int i=0; i<n; ++i){
      file.read (t, sizeof(int));
      w = *((int*)t);
      file.read (t, sizeof(int));
      h = *((int*)t);
      (*_width)[i] = w;
      (*_height)[i] = h;
      int nBytes = w*h*sizeof(float);
      char* mem = (char*) malloc(nBytes);
      file.read(mem,nBytes);
      memcpy((void*)ptr_data,(const void*)mem, nBytes);
      ptr_data += w*h;
      free(mem);
    }

    file.close();
    //cout << "the complete file content is in memory\n";
    
  }
    else cout << "Unable to open file";
}

void pprintMatrix(float* matrix,int width,int height){
  printf("Matrix %ix%i:\n",width,height);
  for (int i=0; i<height; ++i){
    for(int e=0; e<width; ++e){
      printf("%f, ",matrix[i*width+e]);
    }
    printf(";\n");
  }
}


