#include <stdio.h>
#include <stdlib.h>
#include "../../layers/learn.h"


int main(int argc, char* argv[]) {
  // parse command line 
  if (argc <= 5) { 
    printf("usage: %s <gaborfile> <patch_basename> <output_weights_name> <sphering_name> <training: 0 or 1>", argv[0]); 
    exit(1) ;
  }

  // run
  weight_patches(argv[1], argv[2], argv[3], argv[4], atoi(argv[5]));

  return 0;
}
