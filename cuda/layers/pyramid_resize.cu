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
// Contains the CUDA kernel for the interpolated resize to create the pyramid
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

//const float pyr_factor = powf(2, 1/4.0);
// easier...:
#define PYR_FACTOR 1.18920711500272f

/*
== PYRAMID GPU FUNCTIONS
*/

// <<<<<<<<<<<<<<<<<<<

// CUBIC INTERPOLATION, source: nvidia examples

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__host__ __device__
float w0(float a) {
    return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}
__host__ __device__
float w1(float a) {
//    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
    return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}
__host__ __device__
float w2(float a) {
//    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}
__host__ __device__
float w3(float a) {
    return (1.0f/6.0f)*(a*a*a);
}
// g0 and g1 are the two amplitude functions
__device__ float g0(float a) {
    return w0(a) + w1(a);
}
__device__ float g1(float a) {
    return w2(a) + w3(a);
}
// h0 and h1 are the two offset functions
__device__ float h0(float a) {
    // note +0.5 offset to compensate for CUDA linear filtering convention
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}
__device__ float h1(float a) {
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

// filter 4 values using cubic splines
template<class T>
__device__
T cubicFilter(float x, T c0, T c1, T c2, T c3) {
    T r;
    r = c0 * w0(x);
    r += c1 * w1(x);
    r += c2 * w2(x);
    r += c3 * w3(x);
    return r;
}

// slow but precise bicubic lookup using 16 texture lookups
template<class T, class R>  // return type, texture type
__device__
R interp2DBicubic(float x, float y) {
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;
    return cubicFilter<R>(fy,
                          cubicFilter<R>(fx,p2ReadP(0,px-1,py-1),p2ReadP(0,px,py-1),p2ReadP(0,px+1,py-1),p2ReadP(0,px+2,py-1)),
                          cubicFilter<R>(fx,p2ReadP(0,px-1,py),  p2ReadP(0,px,py),  p2ReadP(0,px+1,py),  p2ReadP(0,px+2,py)),
                          cubicFilter<R>(fx,p2ReadP(0,px-1,py+1),p2ReadP(0,px,py+1),p2ReadP(0,px+1,py+1),p2ReadP(0,px+2,py+1)),
                          cubicFilter<R>(fx,p2ReadP(0,px-1,py+2),p2ReadP(0,px,py+2),p2ReadP(0,px+1,py+2),p2ReadP(0,px+2,py+2))
                         );
}

// >>>>>>>>>>>>>>>>>>>


// empty kernel to measure performance accurately. is executed first.
bool pseudoKernel_executed = false;
__global__
void pseudoKernel() {
}

// main pyramid resize kernel
__global__
void resizeKernel(float* img, size_t pitch, int width, int height, int lvl) {
    // coords
    int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if(x < width && y < height) {
        // interpolate color at float-coordinates on lower layer
        float color = interp2DBicubic<float, float>(x * PYR_FACTOR, y * PYR_FACTOR);

        // write color
        p2Write(img, pitch, x, y, color);
    }
}


/*
== PYRAMID RESIZE FUNCTIONS
*/

// main resize method
void pyramidResizeImg2gpuPyr(float*** gpu_pyr, t_pyrprops* _pyrprops, float* _img, uint width, uint height) {

    // run pseudo kernel
    if(!pseudoKernel_executed) {
        pseudoKernel_executed = true;
#ifdef DTIME
        struct timespec ts_start, ts_end;
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
#endif
#ifdef MACTIME
        timeval t1, t2;
        double elapsedTime;
        gettimeofday(&t1, NULL);
#endif

        pseudoKernel<<< dim3(1,1,1), dim3(1,1,1) >>>();

#ifdef DTIME
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        printElapsedTime("Cuda Init", ts_start, ts_end);
#endif
#ifdef MACTIME
        gettimeofday(&t2, NULL);
        // compute and print the elapsed time in millisec
        elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
        elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
        printf("Cuda Init %f ms\n",elapsedTime);// elapsedTime << " ms.\n";
#endif
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
#ifdef MEASURETIME
    clock_gettime(CLOCK_MONOTONIC, &measure_ts_start);
#endif

    // init pyramid properties
    _pyrprops->init(MAX_NLEVELS, 1);
    float currentfactor = 1;
    for(int i=0; i<_pyrprops->nlevels; ++i) {
        // calc height/width
        int h = currentfactor * height;
        int w = currentfactor * width;

        // limit min size
        if(i>0 && (h < MIN_SIDELENGTH || w < MIN_SIDELENGTH)) {
            _pyrprops->nlevels = i;
            break;
        }
        if(DEBUG) printf("i f w h pfx %d %f %d %d %d\n",i,currentfactor,w,h, _pyrprops->pfxsize[i] + h*w);

        // save
        _pyrprops->width[i] = w;
        _pyrprops->height[i] = h;
        _pyrprops->pfxsize[i+1] = _pyrprops->pfxsize[i] + h*w;

        currentfactor /= PYR_FACTOR;
    }
    if(DEBUG) printf("pyr size %d\n",_pyrprops->size());

    unsigned int timer;
    if(DEBUG) {
        (cutCreateTimer(&timer));
    }

    if(DEBUG) {
        cutResetTimer(timer);
        (cutStartTimer(timer));
    }

    // alloc mem for pyramid
    float** gpu_img = new float*[_pyrprops->nlevels];
    // each level
    for(int a=0; a < _pyrprops->nlevels; a++) {
        cutilSafeCall(cudaMallocPitch((void**)&(gpu_img[a]), &(_pyrprops->pitch[a]),
                                      _pyrprops->width[a] * sizeof(float), _pyrprops->height[a]));
    }

    // copy first layer image
    cutilSafeCall(cudaMemcpy2D(gpu_img[0], _pyrprops->pitch[0],
                               _img, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice));

    // copy constants, the pyramid properties
    cutilSafeCall(cudaMemcpyToSymbol(gpu_pyr_nlevels,   &_pyrprops->nlevels,   sizeof(int)));
    cutilSafeCall(cudaMemcpyToSymbol(gpu_pyr_npyramids, &_pyrprops->npyramids, sizeof(int)));
    cutilSafeCall(cudaMemcpyToSymbol(gpu_pyr_pfxsize, _pyrprops->pfxsize,  sizeof(int) * (_pyrprops->nlevels+1)));
    cutilSafeCall(cudaMemcpyToSymbol(gpu_pyr_width,   _pyrprops->width,    sizeof(int) * (_pyrprops->nlevels)));
    cutilSafeCall(cudaMemcpyToSymbol(gpu_pyr_height,  _pyrprops->height,   sizeof(int) * (_pyrprops->nlevels)));

    // texture descriptor
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CUT_CHECK_ERROR("Pyramid Resize Kernel init failed");

    if(DEBUG) {
        (cutStopTimer(timer));
        printf( "Pyramid Init time: %f (ms)\n", cutGetTimerValue(timer));
    }

    for(int ii=0; ii<DEBUG*9+1; ii++) //performance test
    {
        if(DEBUG) {
            cutResetTimer(timer);
            cutStartTimer(timer);
        }

        // process each level separately, starting from lvl=1
        for(int lvl=1; lvl < _pyrprops->nlevels; ++lvl) {
            dim3 blockSize(16, 16);
            dim3 gridSize((_pyrprops->width[lvl] / blockSize.x) + 1,
                          (_pyrprops->height[lvl] / blockSize.y) + 1);

            // bind texture of previous level
            cudaBindTexture2D(0, &gpu_pyr_tex_0, gpu_img[lvl-1], &channelDesc,
                              _pyrprops->width[lvl-1], _pyrprops->height[lvl-1], (_pyrprops->pitch)[lvl-1]);
            gpu_pyr_tex_0.filterMode = cudaFilterModePoint;

            // run kernel
            resizeKernel<<< gridSize,blockSize >>>(gpu_img[lvl], _pyrprops->pitch[lvl],
                                                   _pyrprops->width[lvl], _pyrprops->height[lvl], lvl);
            CUT_CHECK_ERROR("Pyramid Resize Kernel execution failed");

            cudaUnbindTexture(&gpu_pyr_tex_0);
        }

        cutilSafeCall(cudaThreadSynchronize());

        if(DEBUG) {
            (cutStopTimer(timer));
            printf( "Pyramid Processing time: %f (ms)\n", cutGetTimerValue(timer));
        }
    }
    CUT_CHECK_ERROR("Pyramid Resize Kernel finalization failed");

    if(DEBUG) {
        // Delete the timer
        cutDeleteTimer(timer);
    }

#ifdef DTIME
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printElapsedTime("Pyramid Resize", ts_start, ts_end);
#endif
#ifdef MACTIME
    gettimeofday(&t2, NULL);
    // compute and print the elapsed time in millisec
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    printf("Pyramid Resize %f ms\n",elapsedTime);// elapsedTime << " ms.\n";
#endif

    *gpu_pyr = gpu_img;
}

void pyramidResizeFreeGpu(float** gpu_pyr, t_pyrprops* pyrprops) {
    for (int i = pyrprops->nlevels; i--; ) {
        cutilSafeCall(cudaFree(gpu_pyr[i]));
    }
    CUT_CHECK_ERROR("Pyramid Resize cleanup/free failed");
}

// create image pyramid out of file
// stores result in device at gpu_pyr
// pyramid properies are in _pyrprops, and as constants in the device
void pyramidResizeImgFile2gpuPyr(float*** gpu_pyr, t_pyrprops* pyrprops, char* _filename) {
    // read input img
    float* img;
    uint width, height, channels;
    img = loadTGAImage(_filename, &width, &height, &channels);

    pyramidResizeImg2gpuPyr(gpu_pyr, pyrprops, img, width, height);

    delete[] img;
}

void pyramidResizeImgFile2PyrFile(char* _filename, char* _outfile) {
    float** gpu_pyr;
    t_pyrprops pyrprops;
    pyramidResizeImgFile2gpuPyr(&gpu_pyr, &pyrprops, _filename);
    savePyr_gpu2file(gpu_pyr, &pyrprops, _outfile);
    pyramidResizeFreeGpu(gpu_pyr, &pyrprops);
}


