#ifdef WEBCAM_GUARD
#include <svm.h>
#include "pyramid.h"
#include "io_tga.h"


void or_webcamInit(const char* gaborfile, int npatches_maxEveryLevel, int categories);

void extractVector(struct svm_node **sparseVector, t_patches* patches, char* gaborfile, float* image, unsigned int width, unsigned int height);

void webcam_readPatches(t_patches* patches, char* _filename);

void webcam_S2_loadPatches(t_patches* patches);

void webcam_S2_clearPatches();

void webcamToTGA(float* image, char* image_name,unsigned int width, unsigned int height);

void webcam_evalTime( t_patches* patches, char* gaborfile, float* image, unsigned int width, unsigned int height);
#endif //WEBCAM_GUARD
