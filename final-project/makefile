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



opt: clean_opt compile_opt run_opt



nv: clean_nv compile_nv run_nv



debug: clean_debug compile compile_opt run_debug



run:
	- ./main message >> main_.out

run_opt:
	- ./main_opt message >> main_opt_.out

run_nv:
	- ./main_nv message >> main_nv_.out
#	- compute-sanitizer --tool memcheck ./main_nv message >> main_nv_.out
#	- cuda-memcheck --tool memcheck ./main_nv message >> main_nv_.out

run_debug:
	- valgrind $(VALGRINDFLAGS) ./main message >> main_debug.out
	- valgrind $(VALGRINDFLAGS) ./main_opt message >> main_opt_debug.out
	- valgrind $(VALGRINDFLAGS) ./main_nv message >> main_nv_debug.out




compile:
	- $(LD) -o main main.cpp

compile_opt:
	- $(LD) -o main_opt main_opt.cpp

compile_nv:
	- $(NV) -o main_nv main_nv.cu



clean:
	- rm -rf *.o
	- rm -rf *.s
	- rm -rf *.ii
	- rm -rf main_.out main*_.txt
	- rm -rf main

clean_opt:
	- rm -rf *.o
	- rm -rf *.s
	- rm -rf *.ii
	- rm -rf main_opt_.out main_opt*_.txt
	- rm -rf *GRIND.log
	- rm -rf main_opt

clean_debug:
	- rm -rf *.o
	- rm -rf *.s
	- rm -rf *.ii
	- rm -rf main_debug.out main_opt_debug.out main_nv_debug.out main*_debug.txt main_opt*_debug.txt main_nv*_debug.txt
	- rm -rf *GRIND.log
	- rm -rf main main_opt main_nv

clean_nv:
	- rm -rf *.op
	- rm -rf *.s
	- rm -rf *.ii
	- rm -rf main_nv_.out main_nv*_.txt
	- rm -rf *GRIND.log
	- rm -rf main_nv



clear:
	- clear