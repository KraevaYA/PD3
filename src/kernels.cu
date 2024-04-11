#include <math.h>
#include <stdio.h>
#include <float.h>

#include "kernels.cuh"


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4


#ifdef ZERO_BANK_CONFLICTS
    #define CONFLICT_FREE_OFFSET(n) \ ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
    #define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

/////////////////////////////////////////////////////////////////////////


__global__ void gpu_compute_statistics(float *d_T, float *d_mean, float *d_std, unsigned int N, unsigned int m)
{
    unsigned int tid = threadIdx.x;
    unsigned int thread_idx = (blockIdx.x*blockDim.x)+tid;

    float mean, std;

    while (thread_idx < N)
    {
        mean = 0;
        std = 0;

        for (int i = 0; i < m; i++)
        {
            mean += d_T[thread_idx+i];
            std += d_T[thread_idx+i]*d_T[thread_idx+i];
        }

        mean = mean/m;
	std = std/m;

        d_mean[thread_idx] = mean;
        d_std[thread_idx] = (float)sqrt(std-mean*mean);

        thread_idx += blockDim.x*gridDim.x;
    }
}


/////////////////////////////////////////////////////////////////////////


__global__ void gpu_update_statistics(float *d_T, float *d_mean, float *d_std, unsigned int N, unsigned int m)
{
    unsigned int tid = threadIdx.x;
    unsigned int thread_idx = (blockIdx.x*blockDim.x)+tid;

    float mean;
    float std;

    while (thread_idx < N)
    {
        mean = 0;
        std = 0;

        mean = ((float)(m-1)/m)*d_mean[thread_idx] + d_T[thread_idx+m-1]/m;
        std = (float)sqrt(((float)(m-1)/m)*(d_std[thread_idx]*d_std[thread_idx]+(d_mean[thread_idx]-d_T[thread_idx+m-1])*(d_mean[thread_idx]-d_T[thread_idx+m-1])/m));

        d_mean[thread_idx] = mean;
        d_std[thread_idx] = std;

        thread_idx += blockDim.x*gridDim.x;
    }
}


/////////////////////////////////////////////////////////////////////////

__global__ void gpu_find_candidates(float *d_T, float *d_mean, float *d_std, int *d_C, int *d_C_ind, unsigned int N, int subs_ind, float r, unsigned int m)
{
    unsigned int tid = threadIdx.x;
    unsigned int thread_idx = (blockIdx.x*blockDim.x)+tid;
    float dot = 0;

    if (thread_idx < N)
    {
        for (int i = 0; i < m; i++)
            dot += d_T[d_C_ind[thread_idx]+i-1]*d_T[subs_ind+i];
        //float ED_dist = 2*m*(1-(dot-m*d_mean[thread_idx]*d_mean[subs_ind])/(m*d_std[thread_idx]*d_std[subs_ind]));

        float ED_dist = 2*m*(1-(dot-m*d_mean[d_C_ind[thread_idx]-1]*d_mean[subs_ind])/(m*d_std[d_C_ind[thread_idx]-1]*d_std[subs_ind]));

        //bool non_overlap = (abs((int)(thread_idx - subs_ind)) < m) ? 0 : 1;

        bool non_overlap = (abs((int)(d_C_ind[thread_idx]-1 - subs_ind)) < m) ? 0 : 1;

        if (non_overlap)
            d_C[thread_idx] = (ED_dist < r) ? 0 : 1;
    }

}


/////////////////////////////////////////////////////////////////////////


__global__ void gpu_update_index_candidates(int *d_C, int *d_C_ind, int *d_C_ind_temp, int *d_scan_out, unsigned int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int thread_idx = (blockIdx.x*blockDim.x)+tid;
    int ind;

    if (thread_idx < N)
    {
	if (d_C[thread_idx] == 1) 
	{
	    //ind = d_scan_out[thread_idx]+d_C[thread_idx]-1;
	    ind = d_scan_out[thread_idx];
	    d_C_ind_temp[ind] = d_C_ind[thread_idx];  
	}
    }
}


////////////////////////////////////////////////////////////////////////////


__global__ void gpu_update_candidates(int *d_C_ind, int * d_C_ind_temp, unsigned int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int thread_idx = (blockIdx.x*blockDim.x)+tid;

    if (thread_idx < N)
    {
	d_C_ind[thread_idx] = d_C_ind_temp[thread_idx];  
    }
}


/////////////////////////////////////////////////////////////////////////


__device__ void warpReduce(volatile int *dynamicMem1, unsigned int tid, int blockSize)
{
    if (blockSize >= 64)
        dynamicMem1[tid] += dynamicMem1[tid + 32];
    if (blockSize >= 32)
        dynamicMem1[tid] += dynamicMem1[tid + 16];
    if (blockSize >= 16)
        dynamicMem1[tid] += dynamicMem1[tid + 8];
    if (blockSize >= 8)
        dynamicMem1[tid] += dynamicMem1[tid + 4];
    if (blockSize >= 4)
        dynamicMem1[tid] += dynamicMem1[tid + 2];
    if (blockSize >= 2)
        dynamicMem1[tid] += dynamicMem1[tid + 1];
}


/////////////////////////////////////////////////////////////////////////


__global__ void gpu_reduce(int *d_array_in, int *d_array_out, int* d_C, int *d_C_ind, unsigned int N, unsigned int C_size, unsigned int subs_ind, bool nIsPow2, bool update)
{
    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;

    extern __shared__ int dynamicMem1[];

    int sum = 0;

    while (i < N)
    {
        sum += d_array_in[i];

        if (nIsPow2 || i + blockSize < N)
            sum += d_array_in[i+blockSize];

        i += gridSize;
    }

    //if (update)
	//printf("GPU: dynamicMem1[%d] = %d\n", tid, sum);

    dynamicMem1[tid] = sum;
    
    __syncthreads();

    if (blockSize >= 1024)
    {
        if (tid < 512)
            dynamicMem1[tid] += dynamicMem1[tid+ 512];
        __syncthreads();
    }

    if (blockSize >= 512)
    {
        if (tid < 256)
            dynamicMem1[tid] += dynamicMem1[tid+256];
        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
            dynamicMem1[tid] += dynamicMem1[tid+128];
        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid < 64)
            dynamicMem1[tid] += dynamicMem1[tid+64];
        __syncthreads();
    }

    if (tid < 32)
	warpReduce(dynamicMem1, tid, blockSize);


    if (tid == 0)
    {
        d_array_out[blockIdx.x] = dynamicMem1[0];
        if (update)
        {
            d_C_ind[C_size] = subs_ind;
            if (C_size == dynamicMem1[0])
                d_C[C_size] = 1;
            else
                d_C[C_size] = 0;
        }
     }
}


/////////////////////////////////////////////////////////////////////////


__device__ void atomicMin(volatile float* const address, const float value)
{
    if (*address <= value)
    {
	return;
    }

    int* const addressAsI = (int*)address;
    int old = *addressAsI, assumed;

    do 
    {
	assumed = old;
	if (__int_as_float(assumed) <= value)
	{
		break;
	}

	old = atomicCAS(addressAsI, assumed, __float_as_int(value));

    } while (assumed != old);
}


/////////////////////////////////////////////////////////////////////////


__global__ void gpu_compute_min_distances(float *d_T, int *d_C_ind, float *d_mean, float *d_std, float *global_min, unsigned int N, unsigned int C_size, unsigned int m, float r)
{

// 2*(BLOCK_DIM*BLOCK_DIM) + BLOCK_DIM + 4*BLOCK_DIM = shared memory (48 KB)
// 2*BLOCK_DIM^2 + 5*BLOCK_DIM <= 12288 (# floats)
// 2*BLOCK_DIM^2 + 5*BLOCK_DIM - 12288 <= 0
// 2*BLOCK_DIM^2 + 5*BLOCK_DIM - 12288 = 0 
// D = b^2 - 4ac = 5^2 + 4*2*12288 = 98329
// x_1 = (-b + sqrt(D)) / 2a = (-5 + 313.57) / 4 = (-5 + 313) / 4 = 77
// x_2 = (-b - sqrt(D)) / 2a = (-5 - 313.57) / 4 = (-5 -313) / 4 = -79

// BLOCK_DIM \in [0; 77]

    __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM]; // used to store candidates from global memory 
    __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM]; // used to store subsequences from global memory
    __shared__ float shared_min[BLOCK_DIM]; // used to store local minimum of distance between candidates and subsequences

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = BLOCK_DIM*blockIdx.x + tx;
    int col = BLOCK_DIM*blockIdx.y + ty;

    float tmp;
    float dot = 0;
    float dist = 0;
    int ind_cand;

    if (col < C_size)
       ind_cand = d_C_ind[col]-1;

    if (tx+BLOCK_DIM*ty < BLOCK_DIM)
	shared_min[tx] = FLT_MAX;

    for (int i = 0; i < m; i += BLOCK_DIM)
    {
	if (i+tx < m) 
	    shared_A[ty][tx] = (col < C_size) ? d_T[i+ind_cand+tx] : 0;
	else
	    shared_A[ty][tx] = 0;
   

	if (i+ty < m)
	    shared_B[ty][tx] = (row < N) ? d_T[i+ty+row] : 0;
	    //shared_B[ty][tx] = d_T[row+ i + ty];
	else
	    shared_B[ty][tx] = 0;
	
	__syncthreads();


#pragma unroll
	for (int k = 0; k < BLOCK_DIM; k++)
	{
	    //tmp = shared_A[ty][k] - shared_B[k][tx];
	    //dot += tmp*tmp;
	    tmp = shared_A[ty][k]*shared_B[k][tx];
	    dot += tmp;
	}

	__syncthreads();

    }

    //dist = (float)2*m*(1-(dot-m*d_mean[d_C_ind[row]]*d_mean[ind_cand])/(m*d_std[d_C_ind[row]]*d_std[ind_cand]));

    dist = (float)2*m*(1-(dot-m*d_mean[row]*d_mean[ind_cand])/(m*d_std[row]*d_std[ind_cand]));

/*if (blockIdx.x + blockIdx.y + tx + ty == 0)
{
    printf("ind_cand = %d\n", ind_cand);
    printf("row = %d\n", row);
    printf("col = %d\n", col);
    printf("d_mean of subsequence = %f\n", d_mean[row]);
    printf("d_mean of candidate = %f\n", d_mean[ind_cand]);
    printf("d_std of subsequence = %f\n", d_std[row]);
    printf("d_std of candidate = %f\n", d_std[ind_cand]);
    printf("dot = %f\n", dot);*/
    printf("dist = %f\n", dist);
//}

    dist = (abs((int)(row - ind_cand)) < m) || (dist < r) ? FLT_MAX : dist;


    if (row < N)
        atomicMin(shared_min+ty, dist);

    __syncthreads();


    if (tx+BLOCK_DIM*ty < BLOCK_DIM)
    {
        atomicMin(global_min+blockIdx.y*BLOCK_DIM+tx, shared_min[tx]);
    }

    //__threadfence_block();

}


/////////////////////////////////////////////////////////////////////////


__device__ float atomicMax(float* address, float val)
{
    int *address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    while (val > __int_as_float(old)) 
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    }

    return __int_as_float(old);
}


/////////////////////////////////////////////////////////////////////////


__global__ void find_top1_discord(const float* __restrict__ min_dist, int* d_C_ind, const unsigned int C_size, float* top1_discord_dist, int* top1_discord_ind)
{
    __shared__ float sharedMax;
    __shared__ int sharedMaxIdx;
  
    if (0 == threadIdx.x)
    {
        sharedMax = 0.f;
        sharedMaxIdx = 0;
    }
  
    __syncthreads();
  
    float localMax = 0.f;
    int localMaxIdx = 0;
  
    for (int i = threadIdx.x; i < C_size; i += blockDim.x)
    {
        float val = min_dist[i];
  
        if (localMax < val)
        {
            localMax = val;
            localMaxIdx = i;
        }
    }
  
    atomicMax(&sharedMax, localMax);
  
    __syncthreads();
  
    if (sharedMax == localMax)
        sharedMaxIdx = localMaxIdx;
  
    __syncthreads();
  
    if (0 == threadIdx.x)
    {
        *top1_discord_dist = sharedMax;
        *top1_discord_ind = d_C_ind[sharedMaxIdx];
    }
}
