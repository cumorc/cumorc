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
// implements functions declared in pyramid.h
// as well as general pyramid copy operations
// ==========================================================================

#include "pyramid.h"
#include <cstring>
#include <cutil_inline.h>

// methods for working with pyramid "objects" consisting of a memory
// region float* and the descriptor t_pyrprops

// clone, helper for creating new pyramid out of existing
void t_pyrprops::clone(t_pyrprops* t)
{
    t->nlevels = this->nlevels;
    t->npyramids = this->npyramids;
    t->pfxsize = new int[this->nlevels+1];
    t->width   = new int[this->nlevels];
    t->height  = new int[this->nlevels];
    t->pitch   = new size_t[this->nlevels];
    memcpy((void*)t->pfxsize,(void*)pfxsize,sizeof(int) * (nlevels+1));
    memcpy((void*)t->width,  (void*)width,  sizeof(int)   * nlevels);
    memcpy((void*)t->height, (void*)height, sizeof(int)  * nlevels);
    memcpy((void*)t->pitch,  (void*)pitch,  sizeof(size_t)   * nlevels);
}

// copies a pyramid into an array of images, one level per image
void memcpyGpuPyramidToHost(float** pyr, float** gpu_pyr, t_pyrprops* pyrprops)
{
    p("memcpyGpuPyramidToHost",0);
    // alloc mem for pyramid
    *pyr = new float[pyrprops->size() * pyrprops->npyramids];
    // copy each level
    for(int lvl = 0; lvl < pyrprops->nlevels; ++lvl) {
        cutilSafeCall(cudaMemcpy2D( *pyr + pyrprops->pfxsize[lvl] * pyrprops->npyramids,
                                    pyrprops->width[lvl] * pyrprops->npyramids * sizeof(float),
                                    gpu_pyr[lvl],
                                    pyrprops->pitch[lvl],
                                    pyrprops->width[lvl] * pyrprops->npyramids * sizeof(float),
                                    pyrprops->height[lvl],
                                    cudaMemcpyDeviceToHost));
        if(DEBUG) if((*pyr)[pyrprops->pfxsize[lvl]]==0) printf("layer %d is 0 at 0,0\n",lvl);
    }
}

void memcpyHostPyramidToGpu(float** gpu_pyr, float** pyr, t_pyrprops* pyrprops)
{
    p("memcpyGpuPyramidToHost",0);
    // alloc mem for pyramid
    gpu_pyr = new float*[pyrprops->nlevels];
    // copy each level
    for(int lvl = 0; lvl < pyrprops->nlevels; ++lvl) {
        cutilSafeCall(cudaMemcpy2D(  gpu_pyr[lvl],
                                     pyrprops->width[lvl] * pyrprops->npyramids * sizeof(float),
                                     *pyr + pyrprops->pfxsize[lvl] * pyrprops->npyramids,
                                     pyrprops->pitch[lvl],
                                     pyrprops->width[lvl] * pyrprops->npyramids * sizeof(float),
                                     pyrprops->height[lvl],
                                     cudaMemcpyHostToDevice));
        if(DEBUG) if((*pyr)[pyrprops->pfxsize[lvl]]==0) printf("layer %d is 0 at 0,0\n",lvl);
    }
}

// copies a pyramid into one out image
void memcpyGpuPyramidToHost(float** out_img, int* height, int* width, float** gpu_pyr, t_pyrprops* pyrprops)
{
    // build pfxheight
    int* pfxheight = new int[pyrprops->nlevels+1];
    pfxheight[0] = 0;
    for(int lvl=0; lvl<pyrprops->nlevels; ++lvl) {
        pfxheight[lvl+1] = pfxheight[lvl] + pyrprops->height[lvl];
    }

    // init picture
    int basewidth = pyrprops->basewidth(0);
    float* img = new float[pfxheight[pyrprops->nlevels] * basewidth];
    memset(img, 0, pfxheight[pyrprops->nlevels] * basewidth * sizeof(float)); //fill black

    // save levels
    for(int lvl=0; lvl<pyrprops->nlevels; ++lvl) {
        // download image
        cutilSafeCall(cudaMemcpy2D(
                          img + basewidth * pfxheight[lvl], basewidth * sizeof(float),
                          (gpu_pyr[lvl]), pyrprops->pitch[lvl],
                          pyrprops->basewidth(lvl) * sizeof(float), pyrprops->height[lvl], cudaMemcpyDeviceToHost)
                     );
    }

    *out_img = img;
    *width = basewidth;
    *height = pfxheight[pyrprops->nlevels];
    delete[] pfxheight;
}


// saves a pyramid to several files, one for each layer
// file name pattern must contain a %d
void savePyr_gpu2fileSeries(float** gpu_pyr, t_pyrprops* pyrprops, char* _filenamePattern, bool normalize = true)
{
    float* imgs;
    // download images
    memcpyGpuPyramidToHost(&imgs, gpu_pyr, pyrprops);

    // save images
    char* name = new char[strlen(_filenamePattern) + 5];
    for(int lvl=0; lvl<pyrprops->nlevels; lvl++) {
        sprintf(name, _filenamePattern, lvl);

        float* img = imgs + pyrprops->pos(lvl,0,0) * sizeof(float);
        int w = pyrprops->width[lvl];
        int h = pyrprops->height[lvl];
        if(normalize)
            normalizeForTGAOutput(img, w,h);
        saveTGAImage(img, name, w,h);
        delete[] img;
    }
    delete[] name;
    delete[] imgs;
}

// saves a pyramid to one TGA file
void savePyr_gpu2file(float** gpu_pyr, t_pyrprops* pyrprops, char* _filename, bool normalize = true)
{
    float* img;
    int h,w;

    // download image
    memcpyGpuPyramidToHost(&img, &h, &w, gpu_pyr, pyrprops);

    // save image
    if(normalize)
        normalizeForTGAOutput(img, w,h);
    saveTGAImage(img, _filename, w,h);

    delete[] img;
}

