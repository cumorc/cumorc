#include <stdio.h>
#include <stdlib.h>
#include "../../layers/learn.h"

int main(int argc, char *argv[])
{
	//parse command line
	if (argc < 5) {
		printf("usage: %s <nclasses> <nsamples> <ndimensions> <ntests>", argv[0]);
		exit(1);
	}
	//run
	testSvm(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));

	return 0;
}
