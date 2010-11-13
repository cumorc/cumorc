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
// Contains the CUDA kernel for the S2 Layer and the C2 Layer (they are merged)
// The file contains as well the C++ Wrapper code to access the GPU
// ==========================================================================

#include "../util/io_tga.h"
//clear values from S1
#ifdef THREADS16
#undef THREADS16
#endif // THREADS16
#ifdef THREADS
#undef THREADS
#endif // THREADS
#ifdef THREADS2
#undef THREADS2
#endif // THREADS2
#ifdef DD 
#undef DD
#endif // DD

/**
  * Defining the number of threads per block
  */
#define THREADS 16  
#define THREADS2  (THREADS * THREADS)

/**
 *  when defined MULTIPLE_KERNEL S2 calls seperate kernels per layer
 *  otherwise just one kernel for all layers is called
 */
#define MULTIPLE_KERNEL                      

/** 
 *  when set CALCULATE_C2 is defined, the output is going to be an Arraz of length patch_size
 *  with the maximum resopnse of the patch over all positions, scales, and orientations
 *  otherwise the out putis the S2 layer. The array is _much_ larger. Only call with small
 *  patchsize
 */
#define CALCULATE_C2

/**
 *  Debug Makro. If set you get certain informations of the executing programm. Furthermore
 *  the kernel is called 10 times. Time mesurement takes place
 *  Otherwise the programm runs only once, printing just the necessary informations
 */
//#define DD 1


__global__ 
void correlatePatchesKernel(float*  dst,         // destination of ouput array of S2 layer
                            float*  patches,     // input array of patches
                            int*    prefixPatch, // Patch prefix carries 4 values
                            int*    sPatches,     // Patch size carries {16,12,8,4}
                            int*    nPatches,     // numer of patches, Size 4
                            int     nGabor,      // number of gabor orientations  
                            int     nlevel,      // Levels of the pyramid
#ifdef MULTIPLE_KERNEL
                            int     currentLevel,
#endif // MULTIPLE_KERNEL
                            int     maxPosition){

    extern __shared__ float patch[];
    int sPatch;

//Section 1) Copying Patch to shared memory
    {
        int cumulateNumberOld = 0; // cumulateNumberOld
        int cumulateNumberNew = 0; // cumulateNumberNew
        int category, offset;
        for(category = 0; category < NPATCHSIZES; ++category){
            cumulateNumberNew += nPatches[category];
            if(blockIdx.x < cumulateNumberNew){
                break;
            }
            cumulateNumberOld = cumulateNumberNew;
        }
        sPatch = sPatches[category];
        offset = prefixPatch[category] + (blockIdx.x-cumulateNumberOld) * (sPatch*sPatch*nGabor);
        // load patches -> shared memory
        for (int posBlock = 0; posBlock < (sPatch*sPatch*nGabor); posBlock += THREADS2){
            int pos = posBlock + threadIdx.y * blockDim.x + threadIdx.x;
            if(pos < sPatch*sPatch*nGabor){
                patch[pos]  =  patches[offset + pos];
            }
        }
    }
    __syncthreads(); 

//Section 2) Correlate Patch and image
#ifdef CALCULATE_C2
    float maxVal = 0; //  maximum of each thread 
    float pX, pY, pL;
#endif // CALCULATE_C2
#ifdef MULTIPLE_KERNEL
    int level = currentLevel;
    int width = gpu_pyr_width[level];
    for (int posBlockY = sPatch/2; posBlockY <= gpu_pyr_height[level]-sPatch/2; posBlockY += THREADS){ 
        for (int posBlockX = sPatch/2; posBlockX <= width-sPatch/2; posBlockX += THREADS){
            if(posBlockY+threadIdx.y <= gpu_pyr_height[level]-sPatch/2 && posBlockX+threadIdx.x <= width-sPatch/2){
                float value = 0;                     
                for(int orientation=0; orientation < nGabor; ++orientation){ // iteration over gabor orientations
                    for(int i = 0; i < sPatch; ++i){                         // iteration in y direction
                        for(int e = 0; e < sPatch; ++e){                     // iteration in x direction
                            float pic = p2ReadP(orientation, 0,  width, posBlockX+threadIdx.x-sPatch/2+e, posBlockY+threadIdx.y-sPatch/2+i);
                            float filt = patch[orientation*sPatch*sPatch + i*sPatch + e];
                            value = value + (pic-filt) * (pic-filt);
                        }
                    }
                }

                value = expf(-(value/(2 * (sPatch/4) * (sPatch/4))));
#ifdef CALCULATE_C2 
                if(value > maxVal){ //calculates the maximum value of every thread
                    maxVal = value;
                    pX = posBlockX+threadIdx.x;
                    pY = posBlockY+threadIdx.y;
                    pL = level;
                }
#else       
                dst[blockIdx.x*nlevel*gpu_pyr_width[0]*gpu_pyr_height[0] + level*gpu_pyr_width[0]*gpu_pyr_height[0] + gpu_pyr_width[0]*y + x] = value;
#endif // CALCULATE_C2
            }  
        }
    }
#else
    for (int posBlock = 0; posBlock < (maxPosition); posBlock += THREADS2){ 
        int pos = posBlock + threadIdx.y * blockDim.x + threadIdx.x;
        if( pos < maxPosition){
            int x, y;
            int level;
            pyr_pos2Coordinates (level, x, y, pos, nlevel); 
            int width = gpu_pyr_width[level];
            float value = 0;                     
            float pic, filt;   
            for(int orientation=0; orientation < nGabor; ++orientation){ // iteration over gabor orientations
                for(int i = 0; i < sPatch; ++i){                         // iteration in y direction
                    for(int e = 0; e < sPatch; ++e){                     // iteration in x direction
                        pic = p2ReadP(orientation, level, width, x-sPatch/2+e, y-sPatch/2+i);
                        filt = patch[orientation*sPatch*sPatch + i*sPatch + e];
                        value = value + (pic-filt) * (pic-filt);
                    }
                }
            }
            value = expf(-(value/(2 * (sPatch/4) * (sPatch/4))));
#ifdef CALCULATE_C2 
            if(value > maxVal){ //calculates the maximum value of every thread
                maxVal = value;
                pX = x;
                pY = y;
                pL = level;
            }
#else       
            dst[blockIdx.x*nlevel*gpu_pyr_width[0]*gpu_pyr_height[0] + level*gpu_pyr_width[0]*gpu_pyr_height[0] + gpu_pyr_width[0]*y + x] = value;
#endif // CALCULATE_C2
        }  
    }
#endif // MULTIPLE_KERNEL

//Section 3) Reduce Maxima to one single value per block 
#ifdef CALCULATE_C2
     {
        __syncthreads();
        int pos = threadIdx.y * blockDim.x + threadIdx.x;
        patch[pos + 0*THREADS2] = maxVal; // copy local max to shared memory
        patch[pos + 1*THREADS2] = pX;
        patch[pos + 2*THREADS2] = pY;
        patch[pos + 3*THREADS2] = pL;

        //loop log(n) times = find max over all threads
//         for(int stride = 1; stride < THREADS2; stride *= 2){
        for (unsigned int stride = THREADS2>>1; stride > 32; stride >>= 1) {
            __syncthreads(); 
//             if(pos % (2*stride) == 0){
            if (pos < stride){
                if (patch[pos+stride] > patch[pos]){
                    patch[pos + 0*THREADS2] = patch[pos+stride + 0*THREADS2];
                    patch[pos + 1*THREADS2] = patch[pos+stride + 1*THREADS2];
                    patch[pos + 2*THREADS2] = patch[pos+stride + 2*THREADS2];
                    patch[pos + 3*THREADS2] = patch[pos+stride + 3*THREADS2];
                }
            }
        }
        __syncthreads(); 
        if (pos <= 32) { // unroll last 6 predicated steps
              if (patch[pos+32] > patch[pos]){
                    patch[pos + 0*THREADS2] = patch[pos+32 + 0*THREADS2];
                    patch[pos + 1*THREADS2] = patch[pos+32 + 1*THREADS2];
                    patch[pos + 2*THREADS2] = patch[pos+32 + 2*THREADS2];
                    patch[pos + 3*THREADS2] = patch[pos+32 + 3*THREADS2];
               }
              if (patch[pos+16] > patch[pos]){
                    patch[pos + 0*THREADS2] = patch[pos+16 + 0*THREADS2];
                    patch[pos + 1*THREADS2] = patch[pos+16 + 1*THREADS2];
                    patch[pos + 2*THREADS2] = patch[pos+16 + 2*THREADS2];
                    patch[pos + 3*THREADS2] = patch[pos+16 + 3*THREADS2];
               }
              if (patch[pos+8] > patch[pos]){
                    patch[pos + 0*THREADS2] = patch[pos+8 + 0*THREADS2];
                    patch[pos + 1*THREADS2] = patch[pos+8 + 1*THREADS2];
                    patch[pos + 2*THREADS2] = patch[pos+8 + 2*THREADS2];
                    patch[pos + 3*THREADS2] = patch[pos+8 + 3*THREADS2];
               }
              if (patch[pos+4] > patch[pos]){
                    patch[pos + 0*THREADS2] = patch[pos+4 + 0*THREADS2];
                    patch[pos + 1*THREADS2] = patch[pos+4 + 1*THREADS2];
                    patch[pos + 2*THREADS2] = patch[pos+4 + 2*THREADS2];
                    patch[pos + 3*THREADS2] = patch[pos+4 + 3*THREADS2];
               }
              if (patch[pos+2] > patch[pos]){
                    patch[pos + 0*THREADS2] = patch[pos+2 + 0*THREADS2];
                    patch[pos + 1*THREADS2] = patch[pos+2 + 1*THREADS2];
                    patch[pos + 2*THREADS2] = patch[pos+2 + 2*THREADS2];
                    patch[pos + 3*THREADS2] = patch[pos+2 + 3*THREADS2];
               }
              if (patch[pos+1] > patch[pos]){
                    patch[pos + 0*THREADS2] = patch[pos+1 + 0*THREADS2];
                    patch[pos + 1*THREADS2] = patch[pos+1 + 1*THREADS2];
                    patch[pos + 2*THREADS2] = patch[pos+1 + 2*THREADS2];
                    patch[pos + 3*THREADS2] = patch[pos+1 + 3*THREADS2];
               }
        }

//         __syncthreads();

#ifdef DD
        if(pos < 4){
#else
        if(pos == 0){
#endif // DD
#ifdef  MULTIPLE_KERNEL
          //if (patch[0] > dst[4*blockIdx.x]){
            if (patch[0] > dst[blockIdx.x + pos * gridDim.x]){
#endif // MULTIPLE_KERNEL
//                dst[4*blockIdx.x + pos] = patch[0 + pos*THREADS2];
                dst[blockIdx.x + pos * gridDim.x] = patch[0 + pos*THREADS2];
#ifdef  MULTIPLE_KERNEL
            }
#endif // MULTIPLE_KERNEL
        }
    }
#endif // CALCULATE_C2


}


//Global variables
int*   s2_gpuPatchsize,* s2_gpuNpatches,* s2_gpuPrefix;
float* s2_gpuPatches;


int S2_loadPatches(t_patches* patches){

    CUDA_SAFE_CALL( cudaMalloc( (void**)&s2_gpuPatchsize, NPATCHSIZES * sizeof(int) ) ); 
    CUDA_SAFE_CALL( cudaMalloc( (void**)&s2_gpuNpatches,  NPATCHSIZES * sizeof(int) ) ); 
    CUDA_SAFE_CALL( cudaMalloc( (void**)&s2_gpuPrefix,    NPATCHSIZES * sizeof(int) ) ); 
    CUDA_SAFE_CALL( cudaMalloc( (void**)&s2_gpuPatches,   patches-> size() * sizeof(float)));  
    // memcpy
    CUDA_SAFE_CALL( cudaMemcpy(s2_gpuPatchsize, patches->width,    NPATCHSIZES * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(s2_gpuNpatches,  patches->npatches, NPATCHSIZES * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(s2_gpuPrefix,    patches->pfxsize,  NPATCHSIZES * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(s2_gpuPatches,   patches->ptr,      patches-> size() * sizeof(float), cudaMemcpyHostToDevice) );
    return 0;
}

int S2_clearPatches(){
    // cleanup 
    CUDA_SAFE_CALL(cudaFree(s2_gpuPatchsize));  
    CUDA_SAFE_CALL(cudaFree(s2_gpuNpatches));  
    CUDA_SAFE_CALL(cudaFree(s2_gpuPrefix));  
    CUDA_SAFE_CALL(cudaFree(s2_gpuPatches));  
    return 0;
}


int S2_gpu2gpu(float** gpu_pointer, t_patches* patches, float** c1_pyramid, t_pyrprops* pyramidProperties){

#ifdef DTIME
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
#endif
#ifdef MACTIME
    timeval t1, t2;
    double elapsedTime;
    gettimeofday(&t1, NULL);
#endif

    if(patches->norientations != pyramidProperties->npyramids){
        printf("Pyramid and Patch are not compatible! Terminating S2\n");
        exit(1);
    }

    cutilSafeCall(cudaMemcpyToSymbol(gpu_pyr_width, pyramidProperties->width,  sizeof(int) * (pyramidProperties->nlevels)));
    cutilSafeCall(cudaMemcpyToSymbol(gpu_pyr_height,pyramidProperties->height, sizeof(int) * (pyramidProperties->nlevels)));

    float* gpu_result;
#ifdef CALCULATE_C2
    int totalOutputSize = 4 * 
                          patches->npatchesTotal()*
                          sizeof(float);
#else 
    int totalOutputSize = patches->npatchesTotal()*
                          pyramidProperties->width[0] * 
                          pyramidProperties->height[0] * 
                          pyramidProperties->nlevels * 
                          sizeof(float);
#endif // CALCULATE_C2
    CUDA_SAFE_CALL( cudaMalloc( (void**)&gpu_result, totalOutputSize) );     
    CUDA_SAFE_CALL( cudaMemset( gpu_result, 0, totalOutputSize));

    dim3 dimBlock( THREADS, THREADS );   
    dim3 dimGrid(patches->npatchesTotal(),1);
    size_t shmSize = 16*16*patches->norientations * sizeof(float); 
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    size_t offset;
#ifdef DD
    // Print out Kernel arguments 
    printf("calling S2 with %d patches\n", patches->npatchesTotal());
    printf("gpuPrefix    %i,%i,%i,%i\n",patches->pfxsize[0],patches->pfxsize[1],patches->pfxsize[2],patches->pfxsize[3]);
    printf("gpuPatchsize %i,%i,%i,%i\n", patches->width[0],patches->width[1],patches->width[2],patches->width[3]);
    printf("gpuNpatches, %i,%i,%i,%i\n", patches->npatches[0],patches->npatches[1],patches->npatches[2],patches->npatches[3]);
    printf("patches->norientations: %i\n",patches->norientations);
    printf("pyramidProperties->nlevels: %i\n",pyramidProperties->nlevels);
    printf("pyramidProperties->size(): %i\n",pyramidProperties->size());

    unsigned int timer;
    for(int ii=10;ii--;){
    (cutCreateTimer(&timer));
    cutResetTimer(timer); 
    cutStartTimer(timer);
#endif // DD
#ifdef MULTIPLE_KERNEL
    for(int level=0; level < pyramidProperties->nlevels; ++level){
        cudaBindTexture2D(&offset,
                          &gpu_pyr_tex_0,
                          c1_pyramid[level], 
                          &channelDesc, 
                          pyramidProperties->width[level] * pyramidProperties->npyramids, 
                          pyramidProperties->height[level], 
                          pyramidProperties->pitch[level]); 
        gpu_pyr_tex_0.filterMode = cudaFilterModePoint;

        int maxPos = pyramidProperties->width[level] * pyramidProperties->height[level];
        correlatePatchesKernel<<< dimGrid, dimBlock, shmSize >>>(gpu_result, 
                          s2_gpuPatches, s2_gpuPrefix, s2_gpuPatchsize, s2_gpuNpatches, patches->norientations, pyramidProperties->nlevels, level, maxPos);
        CUT_CHECK_ERROR("correlatePatchesKernel Kernel failed");

        cudaUnbindTexture(&gpu_pyr_tex_0);          
    }

#else
#define LAYERITEM(a) \
    if(a < pyramidProperties->nlevels) { \
            cudaBindTexture2D(&offset,\
                              &gpu_pyr_tex_ ## a,\
                              c1_pyramid[a],\
                              &channelDesc,\
                              pyramidProperties->width[a] * pyramidProperties->npyramids,\
                              pyramidProperties->height[a],\
                              pyramidProperties->pitch[a]);\
                  }
#include "layeritem.itm"
#undef LAYERITEM
    correlatePatchesKernel<<< dimGrid, dimBlock, shmSize >>>(gpu_result, 
                      gpuPatches, gpuPrefix, gpuPatchsize, gpuNpatches, patches->norientations, pyramidProperties->nlevels, pyramidProperties->size());
    CUT_CHECK_ERROR("correlatePatchesKernel Kernel failed");
#define LAYERITEM(a) \
    if(a < pyramidProperties->nlevels) { \
        cudaUnbindTexture(&gpu_pyr_tex_ ## a);\
    }
#include "layeritem.itm"
#undef LAYERITEM
#endif // MULTIPLE_KERNEL
    CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifdef DD
    (cutStopTimer(timer));
    printf( "S2 Processing time: %f (ms)\n", cutGetTimerValue(timer));
    (cutDeleteTimer(timer));      
    }
#endif // DD


    CUT_CHECK_ERROR("S2 cleanup failed");
 
#ifdef DTIME
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printElapsedTime("S2", ts_start, ts_end);
#endif
#ifdef MACTIME
    gettimeofday(&t2, NULL);
    // compute and print the elapsed time in millisec
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    printf("S2 %f ms\n",elapsedTime);// elapsedTime << " ms.\n";
#endif

    * gpu_pointer = gpu_result;
    return 0;
}


int S2_gpu2mem(float** mem_pointer, float* gpu_result, t_pyrprops* c1_pyrProps, int numberOfPatches){
#ifdef CALCULATE_C2
#ifdef DD
    int totalOutputSize = 4 *
                          numberOfPatches *
                          sizeof(float);
#else
    int totalOutputSize = 1 *
                          numberOfPatches *
                          sizeof(float);
#endif // DD
#else
    int totalOutputSize = numberOfPatches *
                          c1_pyrProps->width[0] * 
                          c1_pyrProps->height[0] * 
                          c1_pyrProps->nlevels * 
                          sizeof(float);
#endif // CALCULATE_C2
    * (mem_pointer) = (float*)malloc(totalOutputSize);
    CUDA_SAFE_CALL( cudaMemcpy(* (mem_pointer), gpu_result, totalOutputSize, cudaMemcpyDeviceToHost) );
    return 0;

}

int outputS2(float* result, t_pyrprops*  pyramidProperties,  t_patches* patches){
     int numberOfPatches = patches->npatchesTotal();
     int numPatch[NPATCHSIZES];
     memcpy((void*)numPatch, (void*)(patches->npatches), NPATCHSIZES*sizeof(float));
     int cat=0;
#ifdef CALCULATE_C2
    printf("C2 Vector of input image:\n\n");
    for (int patch = 0; patch < numberOfPatches; ++patch){
        while (numPatch[cat]-- <= 0){
            cat++;
        }
//         if(result[4*patch + 0]<0.98){
        printf("PatchNr = %i, ",patch);
        printf("Val = %f, ",  result[patch + 0*numberOfPatches]);
#ifdef DD
        printf("X = %1.0f, ", result[patch + 1*numberOfPatches]);
        printf("Y = %1.0f, ", result[patch + 2*numberOfPatches]);
        printf("Z = %1.0f, ", result[patch + 3*numberOfPatches]);
#endif // DD
        printf("Size = %i",   patches->width[cat]);
        printf("\n");
//         }
    }
    printf("\n\n-----\n Total length=%i\n\n",numberOfPatches);
#else
    float* ptr_out = result;
    char* name = new char[255];
    normalizeForTGAOutput(result, pyramidProperties->width[0],pyramidProperties->height[0] * pyramidProperties->nlevels); 
    
    for (int patch = 0; patch < numberOfPatches; ++patch){
          sprintf(name, "test_out%d.tga", patch);
          saveTGAImage(ptr_out,
                name, 
                pyramidProperties->width[0], 
                pyramidProperties->height[0] * pyramidProperties->nlevels);
          ptr_out += pyramidProperties->width[0] * pyramidProperties->height[0] * pyramidProperties->nlevels;

    }
#endif // CALCULATE_C2
    return 0;
}



int S2_file2gpu(float** gpu_pointer, t_patches* patches, char* inputfile, char* gaborfilters){
    t_pyrprops pyramidProperties;
    float** c1_pyramid;

    calcC1_file2gpu(&c1_pyramid ,&pyramidProperties ,inputfile ,gaborfilters);
//   calculateS1(&c1_pyramid ,&pyramidProperties ,inputfile ,gaborfilters );
#ifdef DD
    printf("S1 loaded succesfully\n nlevels:%i, size0:%ix%i\n",pyramidProperties.nlevels, 
                                                              pyramidProperties.width[0], 
                                                              pyramidProperties.height[0]);
#endif // DD
    S2_gpu2gpu(gpu_pointer, patches, c1_pyramid, &pyramidProperties);


  return 0;
}

