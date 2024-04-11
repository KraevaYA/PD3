#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include "cpu_kernels.hpp"

void cpu_compute_statistics(float *T, float *means, float *stds, unsigned int N, unsigned int m)
{
    float mean = 0, std = 0;
    
    for (int i = 0; i < N; i++)
    {
        mean = 0;
        std = 0;
        
        for (int j = 0; j < m; j++)
        {
            mean += T[i+j];
            std += T[i+j]*T[i+j];
        }
        
        mean = mean/m;
        std = std/m;
        std = (float)sqrt(std - mean*mean);
        
        means[i] = mean;
        stds[i] = std;
    }
}


void cpu_update_statistics(float *T, float *means, float *stds, unsigned int N, unsigned int m)
{
    float mean = 0, std = 0;
    
    for (int i = 0; i < N; i++)
    {
        mean = 0;
        std = 0;
        
        mean = ((float)(m-1)/m)*means[i] + T[i+m-1]/m;
        std = (float)sqrt(((float)(m-1)/m)*(stds[i]*stds[i]+(means[i]-T[i+m-1])*(means[i]-T[i+m-1])/m));
        
        means[i] = mean;
        stds[i] = std;
    }
}


void cpu_find_candidates(float *T, float *means, float *stds, int *C, int *C_ind, unsigned int C_size, unsigned int subs_ind, float r, unsigned int m)
{
    
    float dot = 0;
    float ED_dist = 0;
    bool non_overlap = false;
    
    for (int i = 0; i < C_size; i++)
    {
        dot = 0;
        for (int j = 0; j < m; j++)
            dot += T[C_ind[i]+j-1]*T[subs_ind+j];
        
        //ED_dist = 2*m*(1-(dot-m*means[i]*means[subs_ind])/(m*stds[i]*stds[subs_ind]));
  
        //non_overlap = (abs(i-subs_ind) < m) ? 0 : 1;
        
        ED_dist = 2*m*(1-(dot-m*means[C_ind[i]-1]*means[subs_ind])/(m*stds[C_ind[i]-1]*stds[subs_ind]));
        
        non_overlap = (abs(C_ind[i] - 1 -subs_ind) < m) ? 0 : 1;
        
        if (non_overlap)
            C[i] = (ED_dist < r) ? 0 : 1;
    }
    
}


int cpu_reduce(int *C, int *C_ind, unsigned int C_size, unsigned int subs_ind)
{
    
    int cand_count = 0;
    
    for (int i = 0; i < C_size; i++)
        cand_count += C[i];
   
    C_ind[C_size] = subs_ind;
    
    if (C_size == cand_count)
        C[C_size] = 1;
    else
        C[C_size] = 0;

    return cand_count;
}


void cpu_scan(int *scan_out, int *C, unsigned int C_size)
{
    
    scan_out[0] = 0; // since this is a prescan, not a scan
    
    for (int i = 1; i < C_size; ++i)
        scan_out[i] = C[i-1] + scan_out[i-1];
}


void cpu_update_candidates(int *C, int *C_ind, int *scan_out, unsigned int N)
{
    
    int ind;
    
    for (int i = 0; i < N; i++)
    {
        if (C[i] == 1)
        {
            ind = scan_out[i]+C[i]-1;
            C_ind[ind] = C_ind[i];
        }
    }
}


void cpu_compute_min_distances(float *T, int *C_ind, float *means, float *stds, float *min_dist, unsigned int N, unsigned int C_size, unsigned int m, float r)
{
    
    float cand_std, cand_mean, ED_min, ED_dist, dot;
    int cand_ind;
    bool non_overlap;
 
    for (int j = 0; j < C_size; j++)
    {
        cand_ind = C_ind[j]-1;
        cand_std = stds[cand_ind];
        cand_mean = means[cand_ind];
        ED_min = FLT_MAX;
        
        for (int i = 0; i < N; i++)
        {
            non_overlap = (abs((int)(cand_ind - i)) < m) ? 0 : 1;
            
            if (non_overlap)
            {
                dot = 0;
                
                for (int j = 0; j < m; j++)
                    dot += T[cand_ind+j]*T[i+j];
                
                ED_dist = 2*m*(1-(dot-m*means[i]*cand_mean)/(m*stds[i]*cand_std));
                
                if (ED_min > ED_dist)
                    ED_min = ED_dist;
            }
        }
    
        min_dist[j] = ED_min;
    }
}


Discord cpu_find_top1_discord(float *min_dist, int *C_ind, unsigned int C_size, float r)
{
    Discord top1_discord = {-1, FLT_MAX};
    
    for (int i = 0; i < C_size; i++)
        if (min_dist[i] > r && min_dist[i] > top1_discord.dist)
        {
            top1_discord.dist = min_dist[i];
            top1_discord.ind = C_ind[i];
        }
    
    printf("Discord in function: index = %d, distance = %f\n", top1_discord.ind, top1_discord.dist);
    
    return top1_discord;
}
