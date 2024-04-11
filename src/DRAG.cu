#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <assert.h>
#include <cublas_v2.h>

#include "DRAG.cuh"
#include "DRAG_types.hpp"
#include "kernels.cuh"
#include "cpu_kernels.hpp"
#include "verify_results.hpp"
#include "scan/scan2/scan.cuh"
//#include "cuda_error_hadling.h"


#define DEBUG 1

int THREADS_PER_BLOCK2 = 512;
int ELEMENTS_PER_BLOCK2 = THREADS_PER_BLOCK2 * 2;


Discord do_DRAG(float *h_T, unsigned int n, unsigned int m, unsigned int w, float r)
{

    printf("r = %f\n", r);
    
    Discord h_top1_discord_gpu = {-1, -FLT_MAX};
    
    unsigned int N = n - m + 1;
    unsigned int C_size = m, C_size_new = 0;
    int smemSize = (BLOCK_SIZE <= 32) ? 2*BLOCK_SIZE*sizeof(int) : BLOCK_SIZE*sizeof(int);
    bool nIsPow2 = false;

    int *h_C = (int *)malloc(N*sizeof(int));
    int *h_C_ind = (int *)malloc(N*sizeof(int));
    
    for (int i = 0; i < C_size; i++)
    {
        h_C[i] = 1;
        h_C_ind[i] = i+1;
    }
    
    // Allocate memory space on the device
    float *d_T, *d_mean, *d_std, *d_top1_discord_dist;
    int *d_C, *d_C_ind, *d_C_ind_temp, *d_scan_out, *d_partial_sum, *d_sum_result, *d_top1_discord_ind;
    
    cudaMalloc((void **) &d_T, n*sizeof(float));
    cudaMalloc((void **) &d_mean, N*sizeof(float));
    cudaMalloc((void **) &d_std, N*sizeof(float));
    cudaMalloc((void **) &d_C, N*sizeof(int));
    cudaMalloc((void **) &d_C_ind, N*sizeof(int));
    cudaMalloc((void **) &d_C_ind_temp, N*sizeof(int));
    cudaMalloc((void **) &d_scan_out,  N*sizeof(int));
    cudaMalloc((void **) &d_partial_sum, ((int)ceil(N/(float)BLOCK_SIZE))*sizeof(int));
    cudaMalloc((void **) &d_sum_result, sizeof(int));
    cudaMalloc((void **) &d_top1_discord_ind, sizeof(int));
    cudaMalloc((void **) &d_top1_discord_dist, sizeof(float));
    
    // copy time series from host to device memory
    cudaMemcpy(d_T, h_T, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ind, h_C_ind, N*sizeof(int), cudaMemcpyHostToDevice);
    
#if DEBUG
    unsigned int C_size_new_cpu = 0;

    float *h_mean = (float *)malloc(N*sizeof(float));
    float *h_std = (float *)malloc(N*sizeof(float));
    int *h_scan_out = (int *)malloc(N*sizeof(int));
    
    float *h_mean_gpu = (float *)malloc(N*sizeof(float));
    float *h_std_gpu = (float *)malloc(N*sizeof(float));
    int *h_C_gpu = (int *)malloc(N*sizeof(int));
    int *h_C_ind_gpu = (int *)malloc(N*sizeof(int));
    int *h_scan_out_gpu = (int *)malloc(N*sizeof(int));

#endif
    
    
    // 1. Calculate mean and std
    unsigned int num_blocks = (int)ceil(N/(float)BLOCK_SIZE);
    dim3 blockDim = dim3(BLOCK_SIZE, 1, 1);
    dim3 gridDim = dim3(num_blocks, 1, 1);
    gpu_compute_statistics<<<gridDim, blockDim>>>(d_T, d_mean, d_std, N, m);
    
    
#if DEBUG
    cudaMemcpy(h_mean_gpu, d_mean, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_std_gpu, d_std, N*sizeof(float), cudaMemcpyDeviceToHost);

    cpu_compute_statistics(h_T, h_mean, h_std, N, m);
    
    verify_arrays(h_mean, h_mean_gpu, N);
    verify_arrays(h_std, h_std_gpu, N);
#endif
    
    // 2. Phase 1. Candidate Selection Algorithm
    for (int i = m; i < N; i++)
    {
#if DEBUG
        printf("\n\nIteration %d, number of the candidate is %d\n\n", i-m, i);
#endif
        
        num_blocks = ceil(C_size/(float)BLOCK_SIZE);
        blockDim = dim3(BLOCK_SIZE, 1, 1);
        gridDim = dim3(num_blocks, 1, 1);

        gpu_find_candidates<<<gridDim, blockDim>>>(d_T, d_mean, d_std, d_C, d_C_ind, C_size, i, r, m);

#if DEBUG
        printf("1. Find Candidates\n");
        cudaMemcpy(h_C_gpu, d_C, N*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_ind_gpu, d_C_ind, N*sizeof(int), cudaMemcpyDeviceToHost);

        
        cpu_find_candidates(h_T, h_mean, h_std, h_C, h_C_ind, C_size, i, r, m);

        verify_arrays(h_C, h_C_gpu, C_size);
        verify_arrays(h_C_ind, h_C_ind_gpu, C_size);
#endif
        
        gpu_reduce<<<gridDim, blockDim, smemSize>>>(d_C, d_partial_sum, d_C, d_C_ind, C_size, C_size, i+1, nIsPow2, false);
        gpu_reduce<<<1, blockDim, smemSize>>>(d_partial_sum, d_sum_result, d_C, d_C_ind, num_blocks, C_size, i+1, nIsPow2, true);
        
        cudaMemcpy(&C_size_new, d_sum_result, sizeof(int), cudaMemcpyDeviceToHost);
        
#if DEBUG
        printf("2. Sum Reduction\n");
        C_size_new_cpu = cpu_reduce(h_C, h_C_ind, C_size, i+1);
        
        printf("C_size_new_cpu = %d, C_size_new_gpu = %d\n", C_size_new_cpu, C_size_new);
#endif
        
        // Scan (Prefix Sum)
        if ((C_size+1) > ELEMENTS_PER_BLOCK2)
            scanLargeDeviceArray(d_scan_out, d_C, C_size+1, true);
        else
            scanSmallDeviceArray(d_scan_out, d_C, C_size+1, true);
        
#if DEBUG
        printf("3. Scan\n");
        cudaMemcpy(h_scan_out_gpu, d_scan_out, (C_size+1)*sizeof(int), cudaMemcpyDeviceToHost);
        
        cpu_scan(h_scan_out, h_C, C_size+1);

        verify_arrays(h_scan_out, h_scan_out_gpu, C_size+1);
#endif

        num_blocks = ceil((C_size+1)/(float)BLOCK_SIZE);
        blockDim = dim3(BLOCK_SIZE, 1, 1);
        gridDim = dim3(num_blocks, 1, 1);
        gpu_update_index_candidates<<<gridDim, blockDim>>>(d_C, d_C_ind, d_C_ind_temp, d_scan_out, C_size);
        
        num_blocks = ceil((C_size_new)/(float)BLOCK_SIZE);
        blockDim = dim3(BLOCK_SIZE, 1, 1);
        gridDim = dim3(num_blocks, 1, 1);
        gpu_update_candidates<<<gridDim, blockDim>>>(d_C_ind, d_C_ind_temp, C_size_new);
        
#if DEBUG
        printf("4. Update candidates\n");
        cudaMemcpy(h_C_ind_gpu, d_C_ind, C_size_new*sizeof(int), cudaMemcpyDeviceToHost);
        
        cpu_update_candidates(h_C, h_C_ind, h_scan_out, C_size);
        
        verify_arrays(h_C_ind, h_C_ind_gpu, C_size_new_cpu);
#endif
        
        if (C_size == C_size_new)
            C_size++;
        else
            C_size = C_size_new;
        
    }

    printf("Array of discords index\n");
    for (int i = 0; i < C_size; i++)
	printf("h_C_ind_gpu[%d] = %d\n", i, h_C_ind_gpu[i]);

    // 3. Phase 2. Discords Refinement Algorithm
    
    float *d_min_dist;
    cudaMalloc((void **) &d_min_dist, C_size*sizeof(float));
    
    float *h_min_dist = (float *)malloc(C_size*sizeof(float));
    float *h_min_dist_gpu = (float *)malloc(C_size*sizeof(float));
    
    for (int i = 0; i < C_size; i++)
        h_min_dist[i] = FLT_MAX;
    
    cudaMemcpy(d_min_dist, h_min_dist, C_size*sizeof(float), cudaMemcpyHostToDevice);
    
    blockDim = dim3(BLOCK_DIM, BLOCK_DIM, 1);
    gridDim = dim3((int)ceil(N/(float)BLOCK_DIM), (int)ceil(C_size/(float)BLOCK_DIM), 1);
    
    gpu_compute_min_distances<<<gridDim, blockDim>>>(d_T, d_C_ind, d_mean, d_std, d_min_dist, N, C_size, m, 0);

    
 #if DEBUG
    printf("5. Find minimum distances between candidates and subsequences\n");

    cudaMemcpy(h_min_dist_gpu, d_min_dist, C_size*sizeof(float), cudaMemcpyDeviceToHost);

    cpu_compute_min_distances(h_T, h_C_ind, h_mean, h_std, h_min_dist, N, C_size, m, r);
    
    verify_arrays(h_min_dist, h_min_dist_gpu, C_size);
    
    printf("Array of matrix profile of discords\n");
    for (int i = 0; i < C_size; i++)
	printf("GPU: h_min_gpu[%d] = %f, CPU: h_min_cpu[%d] = %f\n", i, h_min_dist_gpu[i],  i, h_min_dist[i]);

#endif


    blockDim = dim3(BLOCK_SIZE, 1, 1);
    gridDim = dim3((int)ceil(C_size/(float) BLOCK_SIZE), 1, 1);
    
    find_top1_discord<<<gridDim, blockDim>>>(d_min_dist, d_C_ind, C_size, d_top1_discord_dist, d_top1_discord_ind);
    cudaMemcpy(&h_top1_discord_gpu.ind, d_top1_discord_ind, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_top1_discord_gpu.dist, d_top1_discord_dist, sizeof(float), cudaMemcpyDeviceToHost);


#if DEBUG
    printf("6. Find top-1 discord\n");
    Discord h_top1_discord = {-1, -FLT_MAX};

    h_top1_discord = cpu_find_top1_discord(h_min_dist, h_C_ind, C_size, r);

    if ((h_top1_discord_gpu.dist == h_top1_discord.dist) && (h_top1_discord_gpu.ind == h_top1_discord.ind))
        printf("The found top-1 discord by CPU and GPU are same");
    else
    {
	printf("The found top-1 discord by CPU and GPU are different\n");
	printf("CPU: Top-1 discord index = %d, distance = %f\n", h_top1_discord.ind, h_top1_discord.dist);
	printf("GPU: Top-1 discord index = %d, distance = %f\n", h_top1_discord_gpu.ind, h_top1_discord_gpu.dist);
    }

#endif

    return h_top1_discord_gpu;

}

