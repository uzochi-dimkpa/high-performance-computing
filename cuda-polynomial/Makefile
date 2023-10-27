# LD=g++
NV=nvcc --ptxas-options=-v --machine=64 --gpu-architecture=compute_75 --gpu-code=sm_75,compute_75 # --- -gencode arch=compute_52,code=sm_52
NVCLUSTER=nvcc --ptxas-options=-v --machine=64 --gpu-architecture=compute_72 --gpu-code=sm_72,compute_72
LDOMP=g++ -fopenmp
LDOPS=-fopenmp -O3 -mfma -mavx2 -march=znver2 -mtune=znver2 -ffast-math -save-temps -fverbose-asm
LDNOOPS=-fopenmp -mfma -mavx2 -save-temps -fverbose-asm
VALGRINDFLAGS=--track-origins=yes --leak-check=full --show-leak-kinds=all --show-reachable=yes --log-file=VALGRIND.log -s
CACHEGRINDFLAGS=--tool=cachegrind --log-file=CACHEGRIND.log
HELGRINDFLAGS=--tool=helgrind --log-file=HELGRIND.log
CALLGRINDFLAGSS=--tool=callgrind --log-file=CALLGRIND.log



all: clean compile run

# ops: clean compile_ops run

# noops: clean compile_noops run


run:
	- ./cuda 1000000 256 512


debug: clean_debug compile
	- valgrind $(VALGRINDFLAGS) ./cuda



queue:
	- sh cuda.sh



compile:
	- $(NV) -o cuda cuda.cu

compile_cluster:
	- $(NVCLUSTER) -o cuda cuda.cu

# compile_ops:
# 	- $(NV) $(LDOPS) -o cuda cuda.cu

# compile_noops:
# 	- $(NV) $(LDNOOPS) -o cuda cuda.cu



clean:
	- rm -rf *.o
	- rm -rf *.s
	- rm -rf *.ii
	- rm -rf cuda_.out cuda*_.txt
	- rm -rf cuda

clean_debug:
	- rm -rf *.o
	- rm -rf *.s
	- rm -rf *.ii
	- rm -rf cuda_debug.out cuda*_debug.txt
	- rm -rf *GRIND.log
	- rm -rf cuda



clear:
	- clear
