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
// Contains the types that describe pyramid memory regions
// header for pyramid.cu
// ==========================================================================


#ifndef _io_pyramid_h_
#define _io_pyramid_h_

// type that describes a pyramid memory region
// general class to be derived of
class t_imgpyr
{
public:
    int nlevels;
    int* width;
    int* height;
    size_t* pitch;

    void init(int nlevels) {
        this->nlevels = nlevels;
        pitch   = new size_t[nlevels];
        width   = new int[nlevels];
        height  = new int[nlevels];
    }

    ~t_imgpyr() {
        delete[] pitch, width, height;
    }

    int size(int level) {
        return width[level] * height[level];
    }

    int basewidth(int level) {
        return width[level];
    }
};

// classical pyramid description.
// several pyramids possible (npyramids)
// in every level the same number (npyramids) of pyramids
// prefix size for easy calculation of memory address
// prefix size has to be set from outside
class t_pyrprops : public t_imgpyr
{
public:
    int npyramids;
    int* pfxsize; // pfxsize[0]=0, pfxsize[nlevels]=size()

    void init(int nlevels, int npyramids) {
        t_imgpyr::init(nlevels);
        this->npyramids = npyramids;
        pfxsize = new int[nlevels+1];
        pfxsize[0] = 0;
    }

    ~t_pyrprops() {
        delete[] pfxsize;
    }

    void clone(t_pyrprops* t); // see in pyrprops.cu

    t_pyrprops* clone() {
        t_pyrprops* t = new t_pyrprops();
        clone(t);
        return t;
    }

//    void memcpyGpuPyramidToHost(float** img, float* gpu_pyr);

    int size(int level) {
        return pfxsize[level+1] - pfxsize[level];
    }

    int size() {
        return pfxsize[nlevels];
    }

    int pos(int level, int x, int y) {
        return pfxsize[level] + y * width[level] + x;
    }

    int basewidth(int level) {
        return width[level] * npyramids;
    }
};

// simple container class
class t_patchposition
{
public:
    int x;
    int y;
    int level;

    void set(int _level, int _x, int _y) {
        level = _level;
        x = _x;
        y = _y;
    }
};

// patches are stored a bit differently.
// here the number of pictures are different for each level
class t_patches : public t_imgpyr
{
public:
    int norientations; // number of gabor orientations
    int* npatches;    // number of pictures is different for every level
    int* npatchesmax; // num of pictures max (amount of allocated memory)

    int* pfxpatch;    // prefix sum of npatchesmax (prefix sum of NUMBER OF PATCHES MAX)
    t_patchposition* patchpositions; // elements tell where patches has been extracted from (will NOT be STORED!).

    int* pfxsize;     // prefix sum of level size (actual size, number of floats!)
    float* ptr;       // pointer to memory, one for each level


    void init(int nlevels, int norientations) {
        t_imgpyr::init(nlevels);
        this->norientations = norientations;
        this->npatches = new int[nlevels];
        this->npatchesmax = new int[nlevels];
        this->pfxsize = new int[nlevels+1];
        this->pfxsize[0] = 0;
        this->pfxpatch = new int[nlevels+1];
        this->pfxpatch[0] = 0;
    }

    void setPrefixSize(int level) {
        pfxsize[level+1] = pfxsize[level] + basewidthmax(level) * height[level];
        pfxpatch[level+1] = pfxpatch[level] + npatchesmax[level];
    }

    ~t_patches() {
        delete[] npatches, npatchesmax, patchpositions, ptr, pfxsize, pfxpatch;
    }

    int npatchesTotal() {
        int sum=0;
        for(int level=nlevels; level--;)
            sum += npatches[level];
        return sum;
    }

    int npatchesMaxTotal() {
        return pfxpatch[nlevels];
    }

    int pos(int level, int patch, int orientation) {
        int size = t_imgpyr::size(level); // of single orientation
        return pfxsize[level] + patch * (norientations * size) + orientation * (size);
    }

    int sizeAllOrientations(int level) {
        return t_imgpyr::size(level) * norientations;
    }

    int basewidth(int level) {
        return width[level] * norientations * npatches[level];
    }

    int basewidthmax(int level) {
        return width[level] * norientations * npatchesmax[level];
    }

    int sizeLevel(int level) {
        return basewidth(level) * height[level];
    }

    int sizeLevelMax(int level) {
        return pfxsize[level+1] - pfxsize[level];
        // return basewidthmax(level) * height[level];
    }

    int size() {
        return pfxsize[nlevels];
    }

    int patchPositionIndex(int level, int patch) {
        return pfxpatch[level] + patch;
    }
};

void extractPatchesRandomlyFromGpuPyramid(t_patches* patches, float** gpu_pyr, t_pyrprops* pyrprops, int npatches);
void savePatches_fileSeries(t_patches* patches, char* _filenamePattern);
void pyramidResizeImg2gpuPyr(float*** gpu_pyr, t_pyrprops* _pyrprops, float* _img, unsigned int width, unsigned int height);
void pyramidResizeImgFile2gpuPyr(float*** gpu_pyr, t_pyrprops* pyrprops, char* _filename) ;
void pyramidResizeFreeGpu(float** gpu_pyr, t_pyrprops* pyrprops) ;
void pyramidResizeImgFile2PyrFile(char* _filename, char* _outfile);

#endif
