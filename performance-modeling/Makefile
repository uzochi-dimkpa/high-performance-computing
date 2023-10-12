LD=g++
LDFLAGS=-fopenmp -O3 -mfma -mavx2 -march=znver2 -mtune=znver2 -ffast-math -save-temps -fverbose-asm
LDNOOPS=-fopenmp -mfma -mavx2 -save-temps -fverbose-asm
VALGRINDFLAGS=--track-origins=yes --leak-check=full --show-leak-kinds=all --show-reachable=yes -s


simple: clean compile
	- ./convolution 100 200 3
#	- ./convolution 8192 8192 3
#	- ./convolution 4194304 768 3
#	- ./convolution 16777216 768 3
# --- Killed (ignored)


all: clean compile run_all

ops: clean compile_ops run_all

noops: clean compile_noops run_all



run_all:
	- ./convolution 1024 768 3
	- ./convolution 2048 2048 3
	- ./convolution 8192 8192 3
#	- ./convolution 4194304 768 3
#	- ./convolution 16777216 768 3

	- ./convolution 1024 768 5
	- ./convolution 2048 2048 5
	- ./convolution 8192 8192 5
#	- ./convolution 4194304 768 5
#	- ./convolution 16777216 768 5

	- ./convolution 1024 768 7
	- ./convolution 2048 2048 7
	- ./convolution 8192 8192 7
#	- ./convolution 4194304 768 7
#	- ./convolution 16777216 768 7

	- ./convolution 1024 768 9
	- ./convolution 2048 2048 9
	- ./convolution 8192 8192 9
#	- ./convolution 4194304 768 9
#	- ./convolution 16777216 768 9

	- ./convolution 1024 768 11
	- ./convolution 2048 2048 11
	- ./convolution 8192 8192 11
#	- ./convolution 4194304 768 11
#	- ./convolution 16777216 768 11

	- ./convolution 1024 768 13
	- ./convolution 2048 2048 13
	- ./convolution 8192 8192 13
#	- ./convolution 4194304 768 13
#	- ./convolution 16777216 768 13

	- ./convolution 1024 768 15
	- ./convolution 2048 2048 15
	- ./convolution 8192 8192 15
#	- ./convolution 4194304 768 15
#	- ./convolution 16777216 768 15



debug: clean compile
	- valgrind $(VALGRINDFLAGS) ./convolution 50 50 3 >> debug.out
	- valgrind $(VALGRINDFLAGS) ./convolution 50 100 15 >> debug.out
#	- valgrind $(VALGRINDFLAGS) ./convolution 8192 8192 3 >> debug.out
#	- valgrind $(VALGRINDFLAGS) ./convolution 4194304 768 3 >> debug.out
# --- terminate called after throwing an instance of 'std::bad_array_new_length' what():  std::bad_array_new_length
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
