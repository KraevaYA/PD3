#pragma once

#define BLOCK_SIZE 1024
#define BLOCK_DIM 32

//int THREADS_PER_BLOCK2 = 512;
//int ELEMENTS_PER_BLOCK2 = THREADS_PER_BLOCK2 * 2;


__global__ void gpu_compute_statistics(float *d_T, float *d_mean, float *d_std, unsigned int N, unsigned int m);

__global__ void gpu_update_statistics(float *d_T, float *d_mean, float *d_std, unsigned int N, unsigned int m);

__global__ void gpu_find_candidates(float *d_T, float *d_mean, float *d_std, int *d_C, int *d_C_ind, unsigned int N, int subs_ind, float r, unsigned int m);

__global__ void gpu_update_index_candidates(int *d_C, int *d_C_ind, int *d_C_ind_temp, int *d_scan_out, unsigned int N);

__global__ void gpu_update_candidates(int *d_C_ind, int *d_C_ind_temp, unsigned int N);

__global__ void gpu_reduce(int *d_array_in, int *d_array_out, int* d_C, int *d_C_ind, unsigned int N, unsigned int C_size, unsigned int subs_ind, bool nIsPow2, bool update);

__global__ void max_reduce(float* d_array, float* d_max, unsigned int N);

__global__ void gpu_compute_min_distances(float *d_T, int *d_C_ind, float *d_mean, float *d_std, float *global_min, unsigned int N, unsigned int C_size, unsigned int m, float r);

__global__ void find_top1_discord(const float* __restrict__ min_dist, int* d_C_ind, const unsigned int C_size, float* top1_discord_dist, int* top1_discord_ind);

