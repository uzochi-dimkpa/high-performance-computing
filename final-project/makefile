LD=g++
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



debug: clean_debug compile run_debug



run:
	- ./main message >> main_.out



run_debug:
	- valgrind $(VALGRINDFLAGS) ./main message >> main_debug.out



compile:
	- $(LD) -o main main.cpp



clean:
	- rm -rf *.o
	- rm -rf *.s
	- rm -rf *.ii
	- rm -rf main_.out main*_.txt
	- rm -rf main

clean_debug:
	- rm -rf *.o
	- rm -rf *.s
	- rm -rf *.ii
	- rm -rf main_debug.out main*_debug.txt
	- rm -rf *GRIND.log
	- rm -rf main



clear:
	- clear