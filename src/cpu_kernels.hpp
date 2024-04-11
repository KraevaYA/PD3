#pragma once

#include "DRAG_types.hpp"

void cpu_compute_statistics(float *T, float *means, float *stds, unsigned int N, unsigned int m);
void cpu_update_statistics(float *T, float *means, float *stds, unsigned int N, unsigned int m);
void cpu_find_candidates(float *T, float *means, float *stds, int *C, int *C_ind, unsigned int C_size, unsigned int subs_ind, float r, unsigned int m);
int cpu_reduce(int *C, int *C_ind, unsigned int C_size, unsigned int subs_ind);
void cpu_scan(int *scan_out, int *C, unsigned int C_size);
void cpu_update_candidates(int *C, int *C_ind, int *scan_out, unsigned int N);
void cpu_compute_min_distances(float *T, int *C_ind, float *means, float *stds, float *min_dist, unsigned int N, unsigned int C_size, unsigned int m, float r);
Discord cpu_find_top1_discord(float *min_dist, int *C_ind, unsigned int C_size, float r);
