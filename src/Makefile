#Makefile 
#define variables
objects= main.o DRAG.o IOdata.o kernels.o cpu_kernels.o scan/scan2/scan.o scan/scan2/kernels.o scan/scan2/utils.o
#compare_cpu_gpu.o 
NVCC= /usr/local/cuda-10.2/bin/nvcc               #cuda c compiler
opt= -O2 -g -G           #optimization flag
ARCH= -arch=sm_60        #cuda compute capability
LIBS=  
execname= main


#compile
$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS) 

cpu_kernels.o: cpu_kernels.cpp
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c cpu_kernels.cpp
DRAG.o: DRAG.cu
	$(NVCC) $(opt) $(ARCH) -c DRAG.cu
IOdata.o: IOdata.cpp
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c IOdata.cpp
#compare_cpu_gpu.o: compare_cpu_gpu.cpp
#	$(NVCC) $(opt) $(ARCH) -std=c++11 -c compare_cpu_gpu.cpp
kernels.o: kernels.cu
	$(NVCC) $(opt) $(ARCH) -c kernels.cu
main.o: main.cpp
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c main.cpp


#clean Makefile
clean:
	rm $(objects)
#end of Makefile

