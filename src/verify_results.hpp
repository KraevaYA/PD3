#pragma once

#include  <stdlib.h>

template <typename T>
void verify_arrays(T *cpu_array, T *gpu_array, unsigned int size)
{
    int error_count = 0;
    
    for (int i = 0; i < size; i++)
    {
        if (cpu_array[i] != gpu_array[i])
        {
             error_count++;
            //break;
        }
    }
    
    if (error_count == 0)
        printf("Arrays from CPU and GPU are same\n");
    else
    {
        printf("Arrays from CPU and GPU are different\n");
        printf("Error count = %d\n", error_count);
        //exit(0);
    }
    
    /*for (int i = 0; i < size; i++)
    {
        printf("cpu_array[%d] = %f, gpu_array[%d] = %f\n", i, cpu_array[i], i, gpu_array[i]);
    }*/
}







































































// #ifndef __KERNELS_CUH__
// #define __KERNELS_CUH__

// __global__ void dot_product_kernel(float *x, float *y, float *dot, unsigned int n);

// #endif
