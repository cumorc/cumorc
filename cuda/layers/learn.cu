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
// methods that serve as the API
// ==========================================================================



// some constants:
// if S2 similarity is below the following value, it is rounded down
#define MINIMUM_SIMILARITY 0.05
// max value for array declaration
#define MAX_INPUT_PICS 10000

// write out weight matrix into libsvm formatted line
// pic_class needs to be >=1
void saveS2VectorSparsely(FILE* file, int pic_class, float* vector, int numberOfPatches) {
    fprintf(file, "%d", pic_class);
    for (int patch = 0; patch < numberOfPatches; ++patch) {
        float value = vector[patch];
        if(value < MINIMUM_SIMILARITY)
            continue;
        fprintf(file, " %d:%f", patch+1, value); // patch+1 because they begin with 1, not 0
    }
    fprintf(file, "\n");
}

// tests the S2 layer
void testS2(int npatches, char* _picname, char* _gaborfile, char* _patchpattern) {
    float** gpu_img;
    t_pyrprops pyrprops;
    calcC1_file2gpu(&gpu_img, &pyrprops, _picname, _gaborfile);
    savePyr_gpu2file(gpu_img, &pyrprops, "c1b.tga",false);
    t_patches patches;
    initPatches(&patches, pyrprops.npyramids, npatches);
    extractPatchesRandomlyFromGpuPyramid(&patches, gpu_img, &pyrprops, npatches);

    float* gpu_pointer;
    S2_loadPatches(&patches);
    S2_gpu2gpu(&gpu_pointer, &patches, gpu_img, &pyrprops);
    S2_clearPatches();
    float* host_result;
    S2_gpu2mem(&host_result, gpu_pointer, &pyrprops, patches.npatchesTotal());
    //outputS2(host_result, &pyrprops, &patches);

    //FILE* file = fopen("vector","w");
    //saveS2VectorSparsely(file, 1, host_result, patches.npatchesTotal());
    //fclose(file);

    cutilSafeCall(cudaFree(gpu_pointer));
    //s2call(&patches);
}


void extract_patches_from_pic(t_patches* patches, int npatches, char* _picname, char* _gaborfile, char* _patchpattern) {
    float** gpu_img;
    t_pyrprops pyrprops;
    calcC1_file2gpu(&gpu_img, &pyrprops, _picname, _gaborfile);
    if(DEBUG) savePyr_gpu2file(gpu_img, &pyrprops, "c1b.tga");
    extractPatchesRandomlyFromGpuPyramid(patches, gpu_img, &pyrprops, npatches);
    clearC1(gpu_img, &pyrprops);
}

void calc_npatches_per_pic(int** npatches, int npics, int npatches_total) {
    float* R = new float[npics];
    float acc = 0;
    for(int i=0; i<npics; ++i) {
        float r = rand()%(npatches_total*npics);
        R[i] = r;
        acc += r;
    }

    *npatches = new int[npics];
    float factor = npatches_total / acc;
    printf("acc=%f fac=%f\n",acc,factor);
    p("as",npatches_total);
    p("npi",npics);
    int sum=0;
    for(int i=0; i<npics; ++i) {
        int n = round(R[i] * factor);
        if(i==npics-1)
            n=npatches_total-sum;
        (*npatches)[i] = n;
        sum+=n;
        printf("sum=%d int=%d float=%f\n", sum,n,round(R[i] * factor ));
    }
}

void read_filenames(char*** files_, int** pic_class_, int* npics_) {
    char** files = new char*[MAX_INPUT_PICS];
    int* pic_class = new int[MAX_INPUT_PICS];
    int npics;
    for(npics=0; npics<MAX_INPUT_PICS; npics++) {
        files[npics] = new char[256];
        if(scanf("%s %d ", files[npics], &(pic_class[npics])) == EOF)
            break;
    }
    if(DEBUG) printf("%d input files found\n", npics);

    *files_ = files;
    *pic_class_ = pic_class;
    *npics_ = npics;
}

void read_filenames_cleanup(char** files, int npics) {
    for(int file=0; file<npics; file++) {
        delete[] (files[file]);
    }
    delete[] files;
}

// filenames and class over stdin
// npatches_total is npatches for each category
void extract_patches(int npatches_total, char* _gaborfile, char* _patchpattern, int randomseed) {
    srand(randomseed); // init random seed

    // find out norientations
    int norientations;
    {
        int* w_gabTemp,* h_gabTemp;
        float* gabor;
        readMatlabfile(_gaborfile, &w_gabTemp, &h_gabTemp, &gabor, norientations);
        delete[] w_gabTemp, h_gabTemp;
        delete[] gabor;
    }

    char** files;
    int* pic_class;
    int npics;
    read_filenames(&files, &pic_class, &npics);

    t_patches patches;
    if(DEBUG) printf("Extracting patches, INIT\n");
    //initPatches(&patches, norientations, npatches_total * npics / BOESERHACK);

    initPatches(&patches, norientations, npatches_total);
    double npatches_picture_approx = (double)npatches_total / (double)npics;

    // extract patches from each pic, append into patches-variable;
    for(int file=0; file<npics; file++) {
        int npatches_picture = (int)(npatches_picture_approx * (file+1)) - (int)(npatches_picture_approx * file);
        printf("EXTRACTING %d PATCHES FROM %s\n",  npatches_picture, files[file]);
        if( npatches_picture == 0 )
            continue;

#ifdef DTIME
        struct timespec ts_start, ts_end;
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
#endif
        extract_patches_from_pic(&patches, npatches_picture, files[file], _gaborfile, _patchpattern);
#ifdef DTIME
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        printElapsedTime("Patch Extraction", ts_start, ts_end);
#endif
    }

    savePatches(&patches, _patchpattern);
    read_filenames_cleanup(files, npics);
}

void columnMean(float** means_, float** A, int nrows, int ncols) {
    double* sums = new double[ncols];
    for(int c=ncols; c--;) { // in fear of using memset on floats
        sums[c] = 0;
    }
    for(int r=nrows; r--;) {
        for(int c=ncols; c--;) {
            sums[c] += A[r][c];
        }
    }
    *means_ = new float[ncols];
    for(int c=ncols; c--;) {
        (*means_)[c] = sums[c] / nrows;
    }
    delete[] sums;
}

// standard deviation
void columnStd(float** stds_, float* means, float** A, int nrows, int ncols) {
    double* sums = new double[ncols];
    for(int c=ncols; c--;) { // in fear of using memset on floats
        sums[c] = 0;
    }
    for(int r=nrows; r--;) {
        for(int c=ncols; c--;) {
            double v = A[r][c] - means[c];
            sums[c] += v*v;
        }
    }
    *stds_ = new float[ncols];
    for(int c=ncols; c--;) {
        (*stds_)[c] = sqrt(sums[c] / nrows);
    }
    delete[] sums;
}

void saveSpheringParams(char* _filename, float* means, float* stds, int ncols) {
    FILE* file = fopen(_filename, "w");
    for(int c=ncols; c--;) {
        fprintf(file, "%f %f\n", means[c], stds[c]);
    }
    fclose(file);
}

void loadSpheringParams(char* _filename, float** means, float** stds, int ncols) {
    *means = new float[ncols];
    *stds = new float[ncols];
    FILE* file = fopen(_filename, "r");
    for(int c=ncols; c--;) {
        fscanf(file, "%f %f\n", &((*means)[c]), &((*stds)[c]));
    }
    fclose(file);
}

// spheres the data with means and stds
void setSphering(float** A, float* means, float* stds, int nrows, int ncols) {
    for(int r=nrows; r--;) {
        for(int c=ncols; c--;) {
            A[r][c] = (A[r][c] - means[c]) / stds[c];
        }
    }
}

// filenames and class over stdin
// output weight vectors, appending to the file
void weight_patches(char* _gaborfile, char* _patchpattern, char* _weightfilename, char* _spheringfilename, bool _training) {
    // read files
    char** files;
    int* pic_class;
    int npics;
    read_filenames(&files, &pic_class, &npics);

    float** result_matrix = new float*[npics];

    // read patches + load them
    t_patches patches;
    readPatches(&patches, _patchpattern);
    S2_loadPatches(&patches);

    // weight each pic
    for(int file=0; file<npics; file++) {
        printf("WEIGHTING %s\n", files[file]);

#ifdef DTIME
        struct timespec ts_start, ts_end;
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
#endif
        float** gpu_img;
        t_pyrprops pyrprops;
        calcC1_file2gpu(&gpu_img, &pyrprops, files[file], _gaborfile);

        float* gpu_pointer;
        S2_gpu2gpu(&gpu_pointer, &patches, gpu_img, &pyrprops);
        S2_gpu2mem(&(result_matrix[file]), gpu_pointer, &pyrprops, patches.npatchesTotal());
        //outputS2(result_matrix[file], &pyrprops, &patches);

        clearC1(gpu_img, &pyrprops);
        cutilSafeCall(cudaFree(gpu_pointer));

#ifdef MEASURETIME
        clock_gettime(CLOCK_MONOTONIC, &measure_ts_end);
        printElapsedTime("MEASUREDTIME ", measure_ts_start, measure_ts_end);
#endif

#ifdef DTIME
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        printElapsedTime("Weighting", ts_start, ts_end);
#endif
    }
    S2_clearPatches();

    // calc sphering
#ifdef DTIME
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
#endif
    float* means;
    float* stds;
    if(_training) {
        columnMean(&means, result_matrix, npics, patches.npatchesTotal());
        columnStd(&stds, means, result_matrix, npics, patches.npatchesTotal());
        saveSpheringParams(_spheringfilename, means, stds, patches.npatchesTotal());
    } else {
        loadSpheringParams(_spheringfilename, &means, &stds, patches.npatchesTotal());
    }
    setSphering(result_matrix, means, stds, npics, patches.npatchesTotal());
#ifdef DTIME
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printElapsedTime("Sphering", ts_start, ts_end);
#endif
    {
#ifdef DTIME
        struct timespec ts_start, ts_end;
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
#endif
        // write out weight matrix into libsvm formatted file
        for(int file=0; file<npics; file++) {
            FILE* weightfile = fopen(_weightfilename,"a");
            saveS2VectorSparsely(weightfile, pic_class[file], result_matrix[file], patches.npatchesTotal());
            fclose(weightfile);
        }
#ifdef DTIME
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        printElapsedTime("Weight Writing", ts_start, ts_end);
#endif
    }
    read_filenames_cleanup(files, npics);
}

