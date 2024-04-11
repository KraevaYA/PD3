#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <assert.h>
#include "DRAG.cuh"
#include "DRAG_types.hpp"
#include "IOdata.hpp"


int main(int argc, char *argv[])
{
    char *file_name = argv[1]; // name of input file with time series
    unsigned int n = atoi(argv[2]); // length of time series
    unsigned int m = atoi(argv[3]); // length of subsequence
    unsigned int w = atoi(argv[4]); // window
    float r = atof(argv[5]); // range of discords

    float *T = (float *)malloc(n*sizeof(float));

    read_ts(file_name, T);

    Discord top1_discord = {-1, -FLT_MAX};

    top1_discord = do_DRAG(T, n, m, w, r*r);

    printf("The top-1 discord of length %d is at %d, with a discord distance of %.2f\n", m, top1_discord.ind, sqrt(top1_discord.dist));


    return 0;
}

