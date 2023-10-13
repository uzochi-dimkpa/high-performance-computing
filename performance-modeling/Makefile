LD=g++
LDFLAGS=-fopenmp -O3 -mfma -mavx2 -march=znver2 -mtune=znver2 -ffast-math -save-temps -fverbose-asm
LDNOOPS=-fopenmp -mfma -mavx2 -save-temps -fverbose-asm
VALGRINDFLAGS=--track-origins=yes --leak-check=full --show-leak-kinds=all --show-reachable=yes -s



all: clean compile run_all run_all_parallel

run: clean compile run_all

parallel: clean compile run_all_parallel

ops: clean compile_ops run_all

noops: clean compile_noops run_all



simple: clean compile
	- ./convolution 100 200 3 >> convolution_simple.out
	- ./convolution 400 900 15 >> convolution_simple.out
	- ./convolution 8192 1024 7 >> convolution_simple.out
#	- ./convolution 8192 8192 3 >> convolution_simple.out
#	- ./convolution 4194304 768 3 >> convolution_simple.out
#	- ./convolution 16777216 768 3 >> convolution_simple.out
# --- Killed (ignored)



run_all:
	- ./convolution 1024 768 3 >> convolution_k3.out
	- ./convolution 2048 2048 3 >> convolution_k3.out
	- ./convolution 8192 8192 3 >> convolution_k3.out
#	- ./convolution 4194304 768 3 >> convolution_k3.out
#	- ./convolution 16777216 768 3 >> convolution_k3.out

	- ./convolution 1024 768 5 >> convolution_k5.out
	- ./convolution 2048 2048 5 >> convolution_k5.out
	- ./convolution 8192 8192 5 >> convolution_k5.out
#	- ./convolution 4194304 768 5 >> convolution_k5.out
#	- ./convolution 16777216 768 5 >> convolution_k5.out

	- ./convolution 1024 768 7 >> convolution_k7.out
	- ./convolution 2048 2048 7 >> convolution_k7.out
	- ./convolution 8192 8192 7 >> convolution_k7.out
#	- ./convolution 4194304 768 7 >> convolution_k7.out
#	- ./convolution 16777216 768 7 >> convolution_k7.out

	- ./convolution 1024 768 9 >> convolution_k9.out
	- ./convolution 2048 2048 9 >> convolution_k9.out
	- ./convolution 8192 8192 9 >> convolution_k9.out
#	- ./convolution 4194304 768 9 >> convolution_k9.out
#	- ./convolution 16777216 768 9 >> convolution_k9.out

	- ./convolution 1024 768 11 >> convolution_k11.out
	- ./convolution 2048 2048 11 >> convolution_k11.out
	- ./convolution 8192 8192 11 >> convolution_k11.out
#	- ./convolution 4194304 768 11 >> convolution_k11.out
#	- ./convolution 16777216 768 11 >> convolution_k11.out

	- ./convolution 1024 768 13 >> convolution_k13.out
	- ./convolution 2048 2048 13 >> convolution_k13.out
	- ./convolution 8192 8192 13 >> convolution_k13.out
#	- ./convolution 4194304 768 13 >> convolution_k13.out
#	- ./convolution 16777216 768 13 >> convolution_k13.out

	- ./convolution 1024 768 15 >> convolution_k15.out
	- ./convolution 2048 2048 15 >> convolution_k15.out
	- ./convolution 8192 8192 15 >> convolution_k15.out
#	- ./convolution 4194304 768 15 >> convolution_k15.out
#	- ./convolution 16777216 768 15 >> convolution_k15.out



run_all_parallel:
	- ./convolution 1024 768 3 >> convolution_k3_p.out
	- ./convolution 2048 2048 3 >> convolution_k3_p.out
	- ./convolution 8192 8192 3 >> convolution_k3_p.out
#	- ./convolution 4194304 768 3 >> convolution_k3_p.out
#	- ./convolution 16777216 768 3 >> convolution_k3_p.out

	- ./convolution 1024 768 5 >> convolution_k5_p.out
	- ./convolution 2048 2048 5 >> convolution_k5_p.out
	- ./convolution 8192 8192 5 >> convolution_k5_p.out
#	- ./convolution 4194304 768 5 >> convolution_k5_p.out
#	- ./convolution 16777216 768 5 >> convolution_k5_p.out

	- ./convolution 1024 768 7 >> convolution_k7_p.out
	- ./convolution 2048 2048 7 >> convolution_k7_p.out
	- ./convolution 8192 8192 7 >> convolution_k7_p.out
#	- ./convolution 4194304 768 7 >> convolution_k7_p.out
#	- ./convolution 16777216 768 7 >> convolution_k7_p.out

	- ./convolution 1024 768 9 >> convolution_k9_p.out
	- ./convolution 2048 2048 9 >> convolution_k9_p.out
	- ./convolution 8192 8192 9 >> convolution_k9_p.out
#	- ./convolution 4194304 768 9 >> convolution_k9_p.out
#	- ./convolution 16777216 768 9 >> convolution_k9_p.out

	- ./convolution 1024 768 11 >> convolution_k11_p.out
	- ./convolution 2048 2048 11 >> convolution_k11_p.out
	- ./convolution 8192 8192 11 >> convolution_k11_p.out
#	- ./convolution 4194304 768 11 >> convolution_k11_p.out
#	- ./convolution 16777216 768 11 >> convolution_k11_p.out

	- ./convolution 1024 768 13 >> convolution_k13_p.out
	- ./convolution 2048 2048 13 >> convolution_k13_p.out
	- ./convolution 8192 8192 13 >> convolution_k13_p.out
#	- ./convolution 4194304 768 13 >> convolution_k13_p.out
#	- ./convolution 16777216 768 13 >> convolution_k13_p.out

	- ./convolution 1024 768 15 >> convolution_k15_p.out
	- ./convolution 2048 2048 15 >> convolution_k15_p.out
	- ./convolution 8192 8192 15 >> convolution_k15_p.out
#	- ./convolution 4194304 768 15 >> convolution_k15_p.out
#	- ./convolution 16777216 768 15 >> convolution_k15_p.out



debug: clean compile
	- valgrind $(VALGRINDFLAGS) ./convolution 50 50 3 >> debug.out
	- valgrind $(VALGRINDFLAGS) ./convolution 50 100 15 >> debug.out
	- valgrind $(VALGRINDFLAGS) ./convolution 8192 8192 3 >> debug.out
#	- valgrind $(VALGRINDFLAGS) ./convolution 4194304 768 3 >> debug.out
# --- terminate called after throwing an instance of 'std::bad_array_new_length' what():  std::
# --- Killed Error 137 (ignored)
#	- valgrind $(VALGRINDFLAGS) ./convolution 16777216 768 3 >> debug.out
# --- Segmentation fault (ignored)

compile:
	- $(LD) -o convolution convolution.cpp

compile_ops:
	- $(LD) $(LDFLAGS) -o convolution convolution.cpp

compile_noops:
	- $(LD) $(LDNOOPS) -o convolution convolution.cpp


clean:
	- rm -rf *.o
	- rm -rf *.s
	- rm -rf *.ii
	- rm -rf *.out
	- rm -rf convolution_*.txt
	- rm -rf convolution

clear:
	- clear
