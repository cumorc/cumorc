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
// Contains the CUDA kernel for the C1 Layer
// The file also contains the C++ Wrapper code to access the GPU
// ==========================================================================

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cutil_inline.h>
#include <cutil_math.h>
#include "../util/io_tga.h"
#include "../util/io_matlab.h"
#include "pyramid.h"

// lower side length of convolving
#define LOWERLENGTH 10
// higher side length, approx. 10/pyr_factor = 10/1.2
#define HIGHERLENGTH 8
// side length / step size divider = step size
#define STEPSIZEDIVIDER 2

// C1 kernel (MAX-filter)
__global__
void c1Kernel(float* target, size_t pitch, int width, int height, int lvl, int npyramids) {
    // target x, y
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if(tx < width && ty < height) {
        int w1 = gpu_pyr_width[lvl];
        int w2 = gpu_pyr_width[lvl+1];

        // source x, y
        int sx_lower = tx * (LOWERLENGTH / STEPSIZEDIVIDER);
        int sy_lower = ty * (LOWERLENGTH / STEPSIZEDIVIDER);

        // scale sx, sy onto higher layer
        int sx_higher = (float)sx_lower / PYR_FACTOR;
        int sy_higher = (float)sy_lower / PYR_FACTOR;

        for(int pyramid = npyramids; pyramid--; ) {
            // maximum (for dilation)
            float max = 0;

            // lower layer
            // query delta qx, qy
            for(int qx = -LOWERLENGTH / 2; qx < LOWERLENGTH / 2; ++qx) {
                // loop unrolled, for qy=-5; qy<5; qy++
#define MAXL(qy) { max = fmaxf(p2ReadP(pyramid, 0, w1, sx_lower + qx, sy_lower + (qy)), max); }
                MAXL(-5);
                MAXL(-4);
                MAXL(-3);
                MAXL(-2);
                MAXL(-1);
                MAXL(0);
                MAXL(1);
                MAXL(2);
                MAXL(3);
                MAXL(4);
#undef MAXL
            }

            // higher layer
            // query delta qx, qy
            for(int qx = -HIGHERLENGTH / 2; qx < HIGHERLENGTH / 2; ++qx) {
                // loop unrolled, for qy=-4; qy<4; qy++
#define MAXH(qy) { max = fmaxf(p2ReadP(pyramid, 1, w2, sx_higher + qx, sy_higher + (qy)), max); }
                MAXH(-4);
                MAXH(-3);
                MAXH(-2);
                MAXH(-1);
                MAXH(0);
                MAXH(1);
                MAXH(2);
                MAXH(3);
#undef MAXH
            }

            // write out maximum into target picture
            p2Write(target, pitch, pyramid, width, tx, ty, max);
        }
    }
}

// input: gpu pyramid
// output: another gpu pyramid
void calcC1_gpu2gpu(float*** gpu_pyr_c1, t_pyrprops* props2, float** gpu_pyr_s1, t_pyrprops* _pyrprops) {

    unsigned int timer;
    if(DEBUG) {
        if(DEBUG) {
            printf( "Calculate C1 Level\n");
        }
        (cutCreateTimer(&timer));
    }

#ifdef DTIME
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
#endif
#ifdef MACTIME
    timeval t1, t2;
    double elapsedTime;
    gettimeofday(&t1, NULL);
#endif

    // setup new pyrprops
    _pyrprops->clone(props2);
    props2->nlevels -= 1;
    props2->pfxsize[0] = 0;//@ferdi
    for(int lvl = 0; lvl < props2->nlevels; ++lvl) {
        props2->width[lvl] /= LOWERLENGTH / STEPSIZEDIVIDER; // width /= 5
        props2->height[lvl] /= LOWERLENGTH / STEPSIZEDIVIDER;
        props2->pfxsize[lvl+1] = props2->pfxsize[lvl] + props2->width[lvl] * props2->height[lvl];
    }

    if(DEBUG) {
        cutResetTimer(timer);
        (cutStartTimer(timer));
    }

    // allocate gpu mem for output
    float** gpu_img = new float*[props2->nlevels];
    if(DEBUG) printf("allocate mem\n");
    // each level
    for(int a=0; a < props2->nlevels; a++) {
        cutilSafeCall(cudaMallocPitch((void**)&(gpu_img[a]), &(props2->pitch[a]),
                                      _pyrprops->width[a] * _pyrprops->npyramids * sizeof(float),
                                      _pyrprops->height[a]));
        //cutilSafeCall(cudaMemset(gpu_img[a], 0, _pyrprops->height[a] * (_pyrprops->pitch)[a]));
    }
    if(DEBUG) printf("malloc pitch done\n");

    // texture descriptor
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CUT_CHECK_ERROR("C1 Kernel init failed");

    if(DEBUG) {
        (cutStopTimer(timer));
        printf( "C1 Init time: %f (ms)\n", cutGetTimerValue(timer));
    }

    // performance test: repeat C1 layer 10 times.
    // if DEBUG=0, this runs only once
    for(int ii=0; ii<DEBUG*9+1; ii++)
    {
        if(DEBUG) {
            cutResetTimer(timer);
            (cutStartTimer(timer));
        }

        // process each level separately
        for(int lvl=0; lvl < props2->nlevels; ++lvl) {
            dim3 blockSize(8, 8);
            dim3 gridSize(props2->width[lvl] / blockSize.x + 1,
                          props2->height[lvl] / blockSize.y + 1);

            //if(D) printf("c1 16x16, lvl %d, %dx%d, at%d\n", lvl, props2->width[lvl] , props2->height[lvl], props2->pfxsize[lvl]);

            // bind texture of current and texture of next level
            cudaBindTexture2D(0, &gpu_pyr_tex_0, gpu_pyr_s1[lvl], &channelDesc,
                              _pyrprops->width[lvl] * _pyrprops->npyramids, _pyrprops->height[lvl],
                              _pyrprops->pitch[lvl]);
            cudaBindTexture2D(0, &gpu_pyr_tex_1, gpu_pyr_s1[lvl+1], &channelDesc,
                              _pyrprops->width[lvl+1] * _pyrprops->npyramids, _pyrprops->height[lvl+1],
                              _pyrprops->pitch[lvl+1]);
            gpu_pyr_tex_0.filterMode = cudaFilterModePoint;
            gpu_pyr_tex_1.filterMode = cudaFilterModePoint;

            // run kernel
            c1Kernel<<< gridSize, blockSize >>>(gpu_img[lvl], props2->pitch[lvl], props2->width[lvl], props2->height[lvl],
                                                lvl, _pyrprops->npyramids);
            CUT_CHECK_ERROR("C1 Kernel execution failed");

            cudaUnbindTexture(&gpu_pyr_tex_0);
            cudaUnbindTexture(&gpu_pyr_tex_1);
        }

        cutilSafeCall(cudaThreadSynchronize());

        if(DEBUG) {
            (cutStopTimer(timer));
            printf( "C1 Processing time: %f (ms)\n", cutGetTimerValue(timer));
        }
    }
    CUT_CHECK_ERROR("C1 Kernel finalization failed");

    if(DEBUG) {
        // Delete the timer

        (cutDeleteTimer(timer));
    }

    // remove data of previous layer
    clearS1(gpu_pyr_s1, _pyrprops);

    *gpu_pyr_c1 = gpu_img;

#ifdef DTIME
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printElapsedTime("C1", ts_start, ts_end);
#endif
#ifdef MACTIME
    gettimeofday(&t2, NULL);
    // compute and print the elapsed time in millisec
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    printf("C1 %f ms\n",elapsedTime);// elapsedTime << " ms.\n";
#endif
}

// invokes file read, pyramid resize, S1, and C1.
void calcC1_file2gpu(float*** gpu_img, t_pyrprops* pyrprops, char* _filename, char* _gaborfile) {
    float** gpu_pyr;
    t_pyrprops props_pyr;

    //old: skip S1
    //   pyramidResizeImgFile2gpuPyr(&gpu_pyr, &props_pyr, _filename);
    //   if(D) savePyr_gpu2file(gpu_pyr, &props_pyr, "pyr.tga");

    // calc upto S1
    calculateS1(&gpu_pyr,&props_pyr, _filename,  _gaborfile);
    if(DEBUG) savePyr_gpu2file(gpu_pyr, &props_pyr, "s1.tga");

    // do the C1
    calcC1_gpu2gpu(gpu_img, pyrprops, gpu_pyr, &props_pyr);
}

// invokes calcC1_file2gpu but outputs to file, for debugging
void calcC1_file2file(char* _infile, char* _gabor, char* _outfile) {
    float** gpu_img;
    t_pyrprops props;
    calcC1_file2gpu(&gpu_img, &props, _infile, _gabor);
    savePyr_gpu2file(gpu_img, &props, _outfile);
}

// clears all references that C1 created
int clearC1(float** c1pyramid, t_pyrprops* pyramidProperties) {
    for (int i = 0; i < pyramidProperties->nlevels; ++i) {
        cutilSafeCall(cudaFree(c1pyramid[i]));
    }
    return 0;
}
