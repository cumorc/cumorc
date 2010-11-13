#include <stdio.h>
#include <stdlib.h>
#include "../../layers/learn.h"


int main(int argc, char* argv[]) {
  // parse command line 
  if (argc < 4) { 
    //printf("usage: %s <inputImage> <inputGabor> <output_basename> <npatches> <nimages>", argv[0]); 
    printf("usage: %s <inputGabor> <output_basename> <npatches> <randomseed>", argv[0]); 
    exit(1) ;
  }

  // run
  //patchOfPyr(atoi(argv[4]), argv[1], argv[3]);
  //patchOfS1(atoi(argv[4]), argv[1], argv[2], argv[3]);
  //patchOfC1(atoi(argv[4]), argv[1], argv[2], argv[3]);
  extract_patches(atoi(argv[3]), argv[1], argv[2], atoi(argv[4]));

  return 0;
}
