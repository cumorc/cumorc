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
// Alowing direct access, to the different Kernels
// ==========================================================================



#ifdef WEBCAM_GUARD
#include "object_recognition.h"
#include <sys/time.h>                // for gettimeofday()
#include "svm.h"

int sparsifyVector(svm_node**, float*, int);

void webcam_S2_loadPatches(t_patches* patches){
    S2_loadPatches(patches);    
}

void webcam_readPatches(t_patches* patches, char* _filename){
    readPatches(patches, _filename);
}

void webcam_S2_clearPatches(){
    S2_clearPatches();    
}

void webcam_evalTime( t_patches* patches, char* gaborfile, float* image, unsigned int width, unsigned int height){
    timeval t1, t2;
    double elapsedTime;
    
    float** gpuFirstPyramid;
    float** gpuSecondPyramid;
    float** gpuThirdPyramid;
    float*  gpuForthPyramid;
    float*  hostVector;
    t_pyrprops pyramideProperties1;
    t_pyrprops pyramideProperties2;
    printf("extractVector\n");
    saveTGAImage(image, "Validation.tga", width, height);
    // calc C1
    
    // start timer
    gettimeofday(&t1, NULL);
    pyramidResizeImg2gpuPyr(&gpuFirstPyramid, &pyramideProperties1, image, width, height);
    calculateS1(&gpuSecondPyramid, &pyramideProperties1, gpuFirstPyramid, gaborfile);
    calcC1_gpu2gpu(&gpuThirdPyramid, &pyramideProperties2, gpuSecondPyramid, &pyramideProperties1);
    savePyr_gpu2file(gpuThirdPyramid, &pyramideProperties2, "c1b.tga",false);
    S2_gpu2gpu(&gpuForthPyramid, patches, gpuThirdPyramid, &pyramideProperties2);
    S2_gpu2mem(&hostVector, gpuForthPyramid, &pyramideProperties2, patches->npatchesTotal());
    
    // stop timer
    gettimeofday(&t2, NULL);

    // compute and print the elapsed time in millisec
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    printf("%f ms\n",elapsedTime);// elapsedTime << " ms.\n";

    //sparsifyVector(sparseVector, hostVector, patches->npatchesTotal());
    //clean up
    clearC1(gpuThirdPyramid, &pyramideProperties2);
    cudaFree(gpuForthPyramid);
      
}

//
void extractVector(svm_node **sparseVector, t_patches* patches, char* gaborfile, float* image, unsigned int width, unsigned int height){
    float** gpuFirstPyramid;
    float** gpuSecondPyramid;
    float** gpuThirdPyramid;
    float*  gpuForthPyramid;
    float*  hostVector;
    t_pyrprops pyramideProperties1;
    t_pyrprops pyramideProperties2;
    printf("extractVector\n");
    saveTGAImage(image, "Validation.tga", width, height);
    // calc C1
    pyramidResizeImg2gpuPyr(&gpuFirstPyramid, &pyramideProperties1, image, width, height);
    calculateS1(&gpuSecondPyramid, &pyramideProperties1, gpuFirstPyramid, gaborfile);
    calcC1_gpu2gpu(&gpuThirdPyramid, &pyramideProperties2, gpuSecondPyramid, &pyramideProperties1);
    savePyr_gpu2file(gpuThirdPyramid, &pyramideProperties2, "c1b.tga",false);
    S2_gpu2gpu(&gpuForthPyramid, patches, gpuThirdPyramid, &pyramideProperties2);
    S2_gpu2mem(&hostVector, gpuForthPyramid, &pyramideProperties2, patches->npatchesTotal());

    sparsifyVector(sparseVector, hostVector, patches->npatchesTotal());
    //clean up
    clearC1(gpuThirdPyramid, &pyramideProperties2);
    cudaFree(gpuForthPyramid);
}



int sparsifyVector(svm_node** sparse_out, float* hostVect, int length){
    //printf("#Debug length:%d, sizeof(svm_node):%lu, sizeof(double)%lu, sizeof(int)%lu\n", length,sizeof(svm_node),sizeof(double),sizeof(int));
    (*sparse_out) =  (svm_node*)malloc((length+1) * sizeof(svm_node));
    int node_length = 0;
    for(int i=0; i<length; ++i){
        //printf("node_length:%d %i:%f\n",node_length,i,hostVect[i]);
        if(hostVect[i] != 0.0){
            (*sparse_out)[node_length].index =i;
            (*sparse_out)[node_length].value = (hostVect[i]);
            node_length++;   
        }
    }
    (*sparse_out)[node_length++].index = -1;
    //for(int i=0; i<length+1; ++i){
    //    printf("node_length:%d %f\n",(*sparse_out)[i].index,(*sparse_out)[i].value);
    //}

    //sparse_out = &out;
    return node_length;
}

void webcamToTGA(float* image, char* image_name,unsigned int width, unsigned int height){
    saveTGAImage(image, image_name, width, height);
}



#endif // WEBCAM_GUARD
