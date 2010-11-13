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
// Contains the CUDA kernel for the S1 Layer
// The file also contains the C++ Wrapper code to access the GPU
// ==========================================================================


#include "../util/io_matlab.h"
#include "../util/io_tga.h"
#include "S1.h"

// both threads16 and 8 have equal speed
#define THREADS16 16


#ifdef THREADS16
#define THREADS THREADS16
#else
#define THREADS 8
#endif

#define THREADS2 (THREADS*THREADS)
//#define DD 1

__global__
void convolutionKernel(float* _dst, float* _gab, int _w, int _h, int level, size_t pitch, int _nGab, int _sGab) {

    extern __shared__ float filtermask[];

    int direction = blockIdx.x % _nGab;

    int x = (blockIdx.x / _nGab) * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int posBlock = 0; posBlock < (_sGab*_sGab ); posBlock += THREADS2) {
        int pos = posBlock + threadIdx.y * blockDim.x + threadIdx.x;
        if(pos < _sGab*_sGab) {
            filtermask[pos]  =  _gab[direction*_sGab*_sGab + pos];
        }
    }
    __syncthreads();
    if(x < _w  && y < _h) {
        float value = 0; //filtermask[threadIdx.y * _SGab + threadIdx.x];
        float norm = 0;
        for (int i = 0; i < _sGab; ++i ) {
            for (int e = 0; e < _sGab; ++e) {
                float p = p2ReadP(0, x + _sGab/2 - e, y + _sGab/2 - i);
                value += p * filtermask[i * _sGab + e] ;//_gab[offset_mask + i * _SGab + e]);
                norm += p * p;
            }
        }
        value = abs(value /sqrt(norm));
        p2Write(_dst, pitch, direction, _w, x, y, value);
    }


}

bool global_gabor_set = false;
int global_gabor_npyramids;
float* global_gabor;
int global_gabor_size;

//will calculate a given file -> gabor pyramid
int calculateS1(float*** gpu_pyramid, t_pyrprops* pyramidProperties, char* inputfile, const char* gaborfilters) {
//int calculateS1(char* inputfile, const char* gaborfilters, const char* out){

    // calling the previous layer
    float** gpuImagepyramid;
    pyramidResizeImgFile2gpuPyr(&gpuImagepyramid, pyramidProperties, inputfile) ;
    return  calculateS1(gpu_pyramid, pyramidProperties, gpuImagepyramid, gaborfilters);
}


int calculateS1(float*** gpu_pyramid, t_pyrprops* pyramidProperties, float** gpuImagepyramid, const char* gaborfilters) {
#ifdef DTIME
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
#endif
#ifdef MACTIME
    timeval t1, t2;
    double elapsedTime;
    gettimeofday(&t1, NULL);
#endif

    //get Gaborfilters
    if(!global_gabor_set) {
        global_gabor_set = true;
        int* w_gabTemp,* h_gabTemp;
        readMatlabfile(gaborfilters, &w_gabTemp, &h_gabTemp, &global_gabor, global_gabor_npyramids);
        global_gabor_size = w_gabTemp[0];
        delete[] w_gabTemp;
        delete[] h_gabTemp;
    }
    float* gabor = global_gabor;
    pyramidProperties->npyramids = global_gabor_npyramids;
    int gabSize = global_gabor_size;

    float* gpuGabor;
    //copy gabor to device
    CUDA_SAFE_CALL( cudaMalloc( (void**)&gpuGabor, pyramidProperties->npyramids* gabSize * gabSize * sizeof(float) ) );
    CUDA_SAFE_CALL( cudaMemcpy(gpuGabor, gabor, pyramidProperties->npyramids * gabSize * gabSize * sizeof(float), cudaMemcpyHostToDevice) );



    float** gpu_result = new float*[pyramidProperties->nlevels];
    //initialize result memory


    size_t* old_pitch = (size_t*) malloc(pyramidProperties->nlevels * sizeof(size_t));
    for(int level=0; level < pyramidProperties->nlevels; ++level) {
        old_pitch[level] = (pyramidProperties->pitch)[level];
        cutilSafeCall(cudaMallocPitch((void**)&(gpu_result[level]),
                                      &((pyramidProperties->pitch)[level]),
                                      (pyramidProperties->width[level] * pyramidProperties->npyramids * sizeof(float)),
                                      pyramidProperties->height[level]));
    }
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    size_t offset;
#ifdef DD
    printf("calling S1\n");
    printf("pyramidProperties->npyramids: %i\n",pyramidProperties->npyramids);
    printf("pyramidProperties->nlevels: %i\n",pyramidProperties->nlevels);
    printf("pyramidProperties->size(): %i\n",pyramidProperties->size());


    unsigned int timer;
    for(int ii=10; ii--;) {
        (cutCreateTimer(&timer));
        cutResetTimer(timer);
        cutStartTimer(timer);
#endif
        for(int level=0; level < pyramidProperties->nlevels; ++level) {
            cudaBindTexture2D(&offset,
                              &gpu_pyr_tex_0,
                              gpuImagepyramid[level],
                              &channelDesc,
                              pyramidProperties->width[level],
                              pyramidProperties->height[level],
                              old_pitch[level]);
            gpu_pyr_tex_0.filterMode = cudaFilterModePoint;

            dim3 dimBlock( THREADS, THREADS );
            // create Ngabor_orientations-times more blocks in x direction, to have sufficient
            // threads to work in parallel
            dim3 dimGrid( ((pyramidProperties->width[level]  / THREADS)+1) * pyramidProperties->npyramids ,
                          (pyramidProperties->height[level] / THREADS)+1);


            size_t shmSize = gabSize * gabSize * sizeof(float);
            convolutionKernel<<< dimGrid, dimBlock, shmSize >>>(gpu_result[level],
                    gpuGabor,
                    pyramidProperties->width[level],
                    pyramidProperties->height[level],
                    level,
                    (pyramidProperties->pitch)[level],
                    pyramidProperties->npyramids,
                    gabSize );
            CUT_CHECK_ERROR("Convulution Kernel failed");

            cudaUnbindTexture(&gpu_pyr_tex_0);
        }
        CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifdef DD
        (cutStopTimer(timer));
        printf( "S1 Processing time: %f (ms)\n", cutGetTimerValue(timer));
        (cutDeleteTimer(timer));
    }
#endif


    // Clean-up
    pyramidResizeFreeGpu(gpuImagepyramid, pyramidProperties);

    cutilSafeCall(cudaFree( gpuGabor ));

#ifdef DTIME
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printElapsedTime("S1", ts_start, ts_end);
#endif
#ifdef MACTIME
    gettimeofday(&t2, NULL);
    // compute and print the elapsed time in millisec
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    printf("S1 %f ms\n",elapsedTime);// elapsedTime << " ms.\n";
#endif

    * gpu_pyramid = gpu_result;
    return 0;
}



//takes the pyramid out of gpu memory and prints them to files
int outputS1(float** s1pyramid, t_pyrprops* pyramidProperties, const char* out) {
    // download result
    char* name = new char[255];
    for(int level = 0; level < pyramidProperties->nlevels; ++level) {
#ifdef DD
        printf("Downloading level %i, Size_level %i, Pitch %lu\n",level,
               pyramidProperties->size(level),
               pyramidProperties->pitch[level]);
#endif
        float* test_result = (float*) malloc(pyramidProperties->size(level)*pyramidProperties->npyramids*sizeof(float));
        cutilSafeCall(cudaMemcpy2D( test_result,
                                    pyramidProperties->width[level] * pyramidProperties->npyramids * sizeof(float),
                                    s1pyramid[level],
                                    pyramidProperties->pitch[level],
                                    pyramidProperties->width[level] * pyramidProperties->npyramids * sizeof(float),
                                    pyramidProperties->height[level],
                                    cudaMemcpyDeviceToHost));
        sprintf(name, out, level);
#ifdef DD
        printf("Saving file: %s \n", name);
#endif
        saveTGAImage(test_result,
                     name,
                     pyramidProperties->width[level] * pyramidProperties->npyramids,
                     pyramidProperties->height[level]);
        delete test_result;

    }

    return 0;
}


//clears all references extisting on
int clearS1(float** s1pyramid, t_pyrprops* pyramidProperties) {
    for (int i = 0; i < pyramidProperties->nlevels; ++i) {
        cutilSafeCall(cudaFree(s1pyramid[i]));
    }

    return 0;
}
