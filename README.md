# PD3: Parallel DRAG-based Discord Discovery
This repository is related to the PD3 (Parallel DRAG-based Discord Discovery) algorithm that accelerates anomaly discovery in time series with a graphics processor. PD3 is authored by Yana Kraeva (kraevaya@susu.ru) and Mikhail Zymbler (mzym@susu.ru), South Ural State University, Chelyabinsk, Russia. The repository contains the PD3's source code (in C, CUDA), accompanying datasets, and experimental results. Please cite an article that describes PD3 as shown below.

PD3 is based on the DRAG serial algorithm by Yankov et al. [https://doi.org/10.1109/ICDM.2007.61]. The PD3 uses the concept of discord, which is a subsequence of the time series farthest from its nearest neighbor, while the nearest neighbor of this subsequence is a non-self-match at a minimum distance away. Unlike the original algorithm, the PD3 uses the square of normalized Euclidean distance as the metric to speed up computation. In addition, a preprocessing phase was added to the PD3 algorithm, which consists in parallel computation of mean values and standard deviations of all subsequences of the time series. The results are then used at the candidate selection and refinement phases to calculate the distances between subsequences of the time series. Each of these phases of the algorithm is parallelized separately on the basis of the concept of data parallelism and using vector data structures.

# Citation
```
@article{KraevaZymbler2023,
 author    = {Yana Kraeva and Mikhail Zymbler},
 title     = {A Parallel Discord Discovery Algorithm for a Graphics Processor},
 journal   = {Pattern Recognition and Image Analysis},
 volume    = {33},
 number    = {2},
 pages     = {101-112},
 year      = {2023},
 doi       = {10.1134/S1054661823020062},
 url       = {https://doi.org/10.1134/S1054661823020062}
}
```
# Acknowledgement
This work was financially supported by the Russian Science Foundation (grant no. 23-21-00465).
