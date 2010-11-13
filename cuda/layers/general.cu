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
// general CUDA constants, global variables and functions
// ==========================================================================

/*
== CONSTANTS AND VARIABLE DECLARATIONS
*/

typedef unsigned int uint;
typedef unsigned char uchar;

// constants that can be changed
const uint MAX_NLEVELS = 9;        // maximum number of levels, e.g. 9, 15, 18
const uint MIN_SIDELENGTH = 2*24;  // minimum sidelength of highest S1 pyramid layer. must not be too small

__device__ __constant__ int gpu_pyr_nlevels;
__device__ __constant__ int gpu_pyr_npyramids;
__device__ __constant__ int gpu_pyr_pfxsize[MAX_NLEVELS+1];
__device__ __constant__ int gpu_pyr_width[MAX_NLEVELS];
__device__ __constant__ int gpu_pyr_height[MAX_NLEVELS];

texture<float, 1, cudaReadModeElementType> gpu_pyr_tex;

// macro for texture variable creation:
// defines gpu_pyr_tex_0, gpu_pyr_tex_1, .. gpu_pyr_tex_14
// layeritem executes a quasi loop
#define LAYERITEM(a) \
texture<float, 2, cudaReadModeElementType> gpu_pyr_tex_ ## a;
#include "layeritem.itm"
#undef LAYERITEM

// debugging: print time
void printElapsedTime(char* description, struct timespec ts_start, struct timespec ts_end) {
    printf("%s Elapsed time: %luns\n", description,
           (ts_end.tv_sec - ts_start.tv_sec) * 1000000000 + (ts_end.tv_nsec - ts_start.tv_nsec));
    printf("%s Elapsed time: %lums\n", description,
           (ts_end.tv_sec - ts_start.tv_sec) * 1000 + (ts_end.tv_nsec - ts_start.tv_nsec) / 1000000);
}

/*
== PYRAMID ACCESS
*/

// macros for 1D structure
#define IS_INSIDE (x>=0 && x<gpu_pyr_width[level] && y>=0 && y<gpu_pyr_height[level])
#define PYRINDEX  (pyr * gpu_pyr_pfxsize[gpu_pyr_nlevels] + gpu_pyr_pfxsize[level] + y * gpu_pyr_width[level] + x)

// index for 1D array
// warning: no border check
__host__ __device__
int pIdx(int pyr, int level, int x, int y) {
    return PYRINDEX;
}

// index for 1D array
// normal border checks, returning -1 if out of border
__host__ __device__
int pIdx0(int pyr, int level, int x, int y) {
    if(IS_INSIDE)
        return PYRINDEX;
    else
        return -1;
}

// index for 1D array
// with padded border filling (keeps the same value over edge = Neumann)
__host__ __device__
int pIdxP(int pyr, int level, int x, int y) {
    if(x<0) x=0;
    else if(x>=gpu_pyr_width[level]) x=gpu_pyr_width[level]-1;
    if(y<0) y=0;
    else if(y>=gpu_pyr_height[level]) y=gpu_pyr_height[level]-1;
    return PYRINDEX;
}

// pyramid 1D texture read zeroed (Dirichlet)
__device__
float ptRead0(int pyr, int level, int x, int y) {
    if(IS_INSIDE)
        return tex1Dfetch(gpu_pyr_tex, PYRINDEX);
    else
        return 0;
}

// pyramid 1D texture read padded (Neumann)
__device__
float ptReadP(int pyr, int level, int x, int y) {
    int idx = pIdxP(pyr, level, x, y);
    return tex1Dfetch(gpu_pyr_tex, idx);
}

// pyramid global memory read zeroed (Dirichlet)
__host__ __device__
float pgRead0(float* ptr, int pyr, int level, int x, int y) {
    if(IS_INSIDE)
        return ptr[PYRINDEX];
    else
        return 0;
}

// pyramid global memory read padded (Neumann)
__host__ __device__
float pgReadP(float* ptr, int pyr, int level, int x, int y) {
    return ptr[pIdxP(pyr, level, x, y)];
}

// pyramid 2D texture read padded (Neumann)
__device__
float p2ReadP(int level, int x, int y) {
#define LAYERITEM(a) \
  if(level == a) return tex2D(gpu_pyr_tex_ ## a, x, y);
#include "layeritem.itm"
#undef LAYERITEM
    return 0;
}

// pyramid 2D texture read padded (Neumann)
// with multiple pictures (pyramid == which picture)
__device__
float p2ReadP(int pyramid, int level, int width, int x, int y) {
    // clamping / padding
    if(x<0) x=0;
    else if(x>=width) x=width-1;
    // offset / multiple pictures
    int offset = pyramid * width;
    return p2ReadP(level, x + offset, y);
}

// pyramid global memory write
// warning: no border checks here!
__host__ __device__
void pgWrite(float* ptr, int pyr, int level, int x, int y, float value) {
    ptr[PYRINDEX] = value;
}

// pyramid texture memory write
// on texture device pointer
// warning: no border checks here!
__host__ __device__
void p2Write(float* ptr, size_t pitch, int x, int y, float value) {
    ptr[(pitch*y)/sizeof(float) + x] = value;
}

// pyramid texture memory write
// on texture device pointer
// with multiple pictures (pyramid == which picture)
// warning: no border checks here!
__host__ __device__
void p2Write(float* ptr, size_t pitch, int pyramid, int width, int x, int y, float value) {
    int offset = pyramid * width;
    p2Write(ptr, pitch, x + offset, y, value);
}

// Caluclate pyramide coordinates out of virtual position
__host__ __device__
void pyr_pos2Coordinates (int& level, int& x, int& y, int position, int Nlevel) {
    int prefixOld = 0;
    int prefixNew = 0;
    for (level = 0; level < Nlevel; ++level) {
        prefixNew += (gpu_pyr_width[level] * gpu_pyr_height[level]);
        if(position < prefixNew) {
            int difference = position - prefixOld;
            x = difference % gpu_pyr_width[level];
            y = difference / gpu_pyr_width[level];
            return;
        }
        prefixOld = prefixNew;
    }
    level = Nlevel-1;
}

// Get the max position out of Coordinates
__host__ /*__device__ */
int pyr_maxPosition (int* w, int* h, int Nlevel) {
    int max = 0;
    for (int i = 0; i < Nlevel; ++i) {
        max += (w[i] * h[i]);
    }
    return max;
}
