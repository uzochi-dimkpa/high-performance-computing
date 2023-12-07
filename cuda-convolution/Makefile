LD=g++
NV=nvcc --ptxas-options=-v --machine=64 --gpu-architecture=compute_75 --gpu-code=sm_75,compute_75
NVCLUSTER=nvcc --ptxas-options=-v --machine=64 --gpu-architecture=compute_72 --gpu-code=sm_72,compute_72
LDOMP=g++ -fopenmp
LDOPS=-fopenmp -O3 -mfma -mavx2 -march=znver2 -mtune=znver2 -ffast-math -save-temps -fverbose-asm
LDNOOPS=-fopenmp -mfma -mavx2 -save-temps -fverbose-asm
VALGRINDFLAGS=--track-origins=yes --leak-check=full --show-leak-kinds=all --show-reachable=yes --log-file=VALGRIND.log -s
CACHEGRINDFLAGS=--tool=cachegrind --log-file=CACHEGRIND.log
HELGRINDFLAGS=--tool=helgrind --log-file=HELGRIND.log
CALLGRINDFLAGSS=--tool=callgrind --log-file=CALLGRIND.log



all: clean compile run_all

# ops: clean compile_ops run

# noops: clean compile_noops run


run_all:
	- ./cnvl 20 40 3 >> cuda_cnvl_.out
	- ./cnvl 1024 768 7 >> cuda_cnvl_.out
#	- ./cnvl 2048 2048 5 >> cuda_cnvl_.out
#	- ./cnvl 4096 4096 15 >> cuda_cnvl_.out


debug: clean_debug compile
#	- valgrind $(VALGRINDFLAGS) ./cnvl 20 40 3 >> cuda_cnvl_debug.out
#	- valgrind $(VALGRINDFLAGS) ./cnvl 1024 768 7 >> cuda_cnvl_debug.out
#	- valgrind $(VALGRINDFLAGS) ./cnvl 2048 2048 5 >> cuda_cnvl_debug.out
#	- valgrind $(VALGRINDFLAGS) ./cnvl 4096 4096 15 >> cuda_cnvl_debug.out



queue:
	- sh cnvl.sh



compile:
	- $(NV) -o cnvl cnvl.cu

compile_cluster:
	- $(NVCLUSTER) -o cnvl cnvl.cu

# compile_ops:
# 	- $(NV) $(LDOPS) -o cnvl cnvl.cu

# compile_noops:
# 	- $(NV) $(LDNOOPS) -o cnvl cnvl.cu



clean:
	- rm -rf *.o
	- rm -rf *.s
	- rm -rf *.ii
	- rm -rf cuda_cnvl_.out cuda_cnvl*_.txt
	- rm -rf cnvl

clean_debug:
	- rm -rf *.o
	- rm -rf *.s
	- rm -rf *.ii
	- rm -rf cuda_cnvl_debug.out cuda_cnvl*_debug.txt
	- rm -rf *GRIND.log
	- rm -rf cnvl



clear:
	- clear
