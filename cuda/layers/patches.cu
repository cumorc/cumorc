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
// extract, read, write routines for patches
// supplements pyramid.cu
// ==========================================================================


#include "pyramid.h"
#include <cstdlib>

// number of sizes
#define NPATCHSIZES 4

// the sizes
const int PATCHSIZE[NPATCHSIZES] = {16, 12, 8, 4};


// inits patches for one single pyramid, it later combined to several patches
void initPatches(t_patches* pats, int norientations, int npatches_maxEveryLevel)
{
    pats->init(NPATCHSIZES, norientations);
    for(int level = 0; level < pats->nlevels; level++) {
        pats->width[level] = PATCHSIZE[level];
        pats->height[level] = PATCHSIZE[level];
        // reset npatches
        pats->npatches[level] = 0;
        pats->npatchesmax[level] = npatches_maxEveryLevel;
        pats->setPrefixSize(level);
    }
    // alloc mem
    pats->ptr = new float[pats->size()];
    memset(pats->ptr, 0, pats->size() * sizeof(float));
    pats->patchpositions = new t_patchposition[pats->npatchesMaxTotal()];
}

// random int between lower and higher
int rand(int lower, int higher)
{
    if(higher == lower) return 0;
    return (rand() % (higher - lower)) + lower;
}

int randLayer(int numberOfLayer)
{
    const float distribution[] = {1.000000 ,1.707107 ,2.207107 ,2.560660 ,2.810660 ,2.987437 ,3.112437 ,
                                  3.200825 ,3.263325 ,3.307519 ,3.338769 ,3.360866 ,3.376491 ,3.387540 ,3.395353
                                 };
    float r = (float)rand()/RAND_MAX;
//     printf("rand = % f, numberOfLayer = % i\n", r, numberOfLayer);
    int layer;
    for (layer = 0; layer < numberOfLayer; ++layer) {
        if( r < (distribution[layer] / distribution[numberOfLayer-1])) {
            break;
        }
    }
    return layer;
}

// extract patches from pyramid into patches memory
// from random locations
void extractPatchesRandomlyFromGpuPyramid(t_patches* patches, float** gpu_pyr, t_pyrprops* pyrprops, int npatches)
{

    if(DEBUG) printf("Extracting patches, MEMCPY\n");
    float* pyr;
    memcpyGpuPyramidToHost(&pyr, gpu_pyr, pyrprops);

    // save for debugging:
    // saveTGAImage(pyr,"pyr0.tga",pyrprops->width[0],pyrprops->height[0]*pyrprops->npyramids);

    if(DEBUG) printf("Extracting patches\n");
    //srand ( time(NULL) ); // init random seed
    //srand ( 12375295 ); // init random seed

    // read out patches
    for(int p=0; p<npatches; ++p) {
        // random vars
        int sizeindex=rand(0,4);
        int size = PATCHSIZE[sizeindex];
        // other version, each level same probability:
        int level = rand(0, pyrprops->nlevels);
        // lower levels higher probability:
        // int level = randLayer(pyrprops->nlevels);
        int picwidth = pyrprops->width[level];
        int x = rand(size/2, picwidth - size/2);
        int y = rand(size/2, pyrprops->height[level] - size/2);

        // image too small for patchsize?
        if( x < size/2 || y < size/2 || x > picwidth - size/2 || y > pyrprops->height[level] - size/2) {
            p--; // decrease loop variable
            continue; // retry
        }

        int basewidth = pyrprops->basewidth(0);
        // if(DEBUG) printf("Extract Patch Nr %i: x=%i, y=%i, z=%i, size=%d\n",p,x,y,level,size);
        // extract all orientations
        for(int ori = 0; ori<pyrprops->npyramids; ++ori) {
            // target base position
            int patchpos = patches->pos(sizeindex, patches->npatches[sizeindex], ori);
            float* patchptr = &(patches->ptr[patchpos]);
            // copy rows
            for(int ty = 0; ty < size; ty++) {
                // source position
                int startx = x - size/2 + picwidth * ori;
                int starty = ty + y - size/2;
                int rowpos = pyrprops->pfxsize[level]*pyrprops->npyramids + starty*picwidth*pyrprops->npyramids + startx;
                memcpy((patchptr + size*ty), pyr+rowpos, size * sizeof(float));
            }
        }

        // save patch position
        patches->patchpositions[
            patches->patchPositionIndex(sizeindex, patches->npatches[sizeindex])
        ].set(level, x, y);

        // inc number of patches
        patches->npatches[sizeindex]++;
    }

    delete[] pyr;
}


///////// READ/WRITE PATCHES

// simple error checking routine
#define checkneqErr(str, a,b) \
if(a!=b) {fprintf(stderr, "\nERROR: reading file %s, %s %d, expected %d\n", filename, str, a, b); return;}

void savePatches_descriptor(t_patches* patches, char* filename)
{
    FILE* f = fopen(filename, "w");
    if (f == 0) {
        fprintf(stderr, "\nERROR: could not save patch descriptor %s", filename);
        return;
    }
    int io;
    io = fprintf(f, "featureExtractor_Patches_Version %d\n", 2);
    if(!io) {
        fprintf(stderr, "\nERROR: saving file %s\n", filename);
        return;
    }

    fprintf(f, "%d %d\n",
            patches->nlevels,
            patches->norientations);

    int output_pfxsize = 0;
    int output_pfxpatch = 0;
    for(int level = 0; level < patches->nlevels; level++) {
        output_pfxsize += patches->basewidth(level) * patches->height[level];
        output_pfxpatch += patches->npatches[level];
        fprintf(f, "%d %d %d %d %d\n",
                patches->width[level],
                patches->height[level],
                patches->npatches[level],
                output_pfxsize,
                output_pfxpatch);
    }
    fclose(f);
    printf("Saved Patch Descriptor File %s\n", filename);
}

void readPatches_descriptor(t_patches* patches, char* filename)
{
    FILE* f = fopen(filename, "r");
    if (f == 0) {
        fprintf(stderr, "\nERROR: could not read patch descriptor %s", filename);
        return;
    }
    int io;
    int version;
    io = fscanf(f, "featureExtractor_Patches_Version %d\n", &version);
    if(!io) {
        fprintf(stderr, "\nERROR: reading file %s\n", filename);
        return;
    }
    checkneqErr("version number", version, 2);

    int nlevels;
    int norientations;
    fscanf(f, "%d %d\n",
           &nlevels,
           &norientations);
    patches->init(nlevels, norientations);
    for(int level = 0; level < patches->nlevels; level++) {
        fscanf(f, "%d %d %d %d %d\n",
               &(patches->width[level]),
               &(patches->height[level]),
               &(patches->npatches[level]),
               &(patches->pfxsize[level+1]),
               &(patches->pfxpatch[level+1]));
        patches->npatchesmax[level] = patches->npatches[level];
    }
    fclose(f);
    printf("Read Patch Descriptor File %s\n", filename);
}

void savePatches_fileSeries(t_patches* patches, char* _filenamePattern)
{
    char* name = new char[strlen(_filenamePattern) + 5];
    for(int lvl=0; lvl<patches->nlevels; lvl++) {
        sprintf(name, _filenamePattern, lvl);
        float* img = &(patches->ptr[patches->pos(lvl,0,0)]);
        int w = patches->width[lvl];
        int h = patches->height[lvl] * patches->norientations * patches->npatches[lvl];
        printf("SAVE patchwidthlvl=%d patchheightlvl=%d patchnori=%d patchnpatch=%d h=%d\n",patches->width[lvl], patches->height[lvl], patches->norientations, patches->npatches[lvl],h);
        //normalizeForTGAOutput(img, w,h);
        saveLargeTGAImage(img, name, w, h);
    }
    delete[] name;
}

void readPatches_fileSeries(t_patches* patches, char* _filenamePattern)
{
    // alloc mem for all patches
    patches->ptr = new float[patches->size()];
    // read out each level image
    char* filename = new char[strlen(_filenamePattern) + 5];
    printf("patches->nlevels=%d\n",patches->nlevels);
    for(int lvl=0; lvl<patches->nlevels; lvl++) {
        sprintf(filename, _filenamePattern, lvl);
        float* img;
        unsigned int h, w, nchans;
        img = loadLargeTGAImage(filename, &w, &h, &nchans);
        if(img == 0)
            continue;
        checkneqErr("channels", nchans, 1);
        checkneqErr("width", w, patches->width[lvl]);
        printf("LOAD patchwidthlvl=%d patchheightlvl=%d patchnori=%d patchnpatch=%d h=%d\n",patches->width[lvl], patches->height[lvl], patches->norientations, patches->npatches[lvl],h);
        checkneqErr("height", h, patches->height[lvl] * patches->norientations * patches->npatches[lvl]);
        memcpy(&(patches->ptr[patches->pos(lvl,0,0)]), img, h*w * sizeof(float));
        printf("file=%s w=%d h=%d pos=%d ptr=%p\n",filename, w, h, patches->pos(lvl,0,0), &(patches->ptr[patches->pos(lvl,0,0)]));
        delete[] img;
    }
    delete[] filename;
}

#undef checkneqErr

void savePatches(t_patches* patches, char* _filename)
{
    char* filenamedescriptor = new char[strlen(_filename) + 9];
    char* filenamepics = new char[strlen(_filename) + 9];
    sprintf(filenamedescriptor, "%s.pat", _filename);
    sprintf(filenamepics, "%s_%%d.tga", _filename);
    savePatches_descriptor(patches, filenamedescriptor);
    savePatches_fileSeries(patches, filenamepics);
    delete[] filenamedescriptor, filenamepics;
}

void readPatches(t_patches* patches, char* _filename)
{
    char* filenamedescriptor = new char[strlen(_filename) + 9];
    char* filenamepics = new char[strlen(_filename) + 9];
    sprintf(filenamedescriptor, "%s.pat", _filename);
    sprintf(filenamepics, "%s_%%d.tga", _filename);
    readPatches_descriptor(patches, filenamedescriptor);
    readPatches_fileSeries(patches, filenamepics);
    delete[] filenamedescriptor, filenamepics;
}

