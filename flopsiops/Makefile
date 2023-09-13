LD=g++
LDFLAGS=-fopenmp -Ofast -mfma -mavx2 -march=znver2 -mtune=znver2 -ffast-math -save-temps -fverbose-asm
LDNOOPS=-fopenmp -mfma -mavx2 -save-temps -fverbose-asm
VALGRINDFLAGS=--track-origins=yes --leak-check=full --show-leak-kinds=all --show-reachable=yes -s


all: clean compile run_flops run_iops

_flops: clean compile run_flops

_iops: clean compile run_iops

run_flops:
	- ./flops

run_iops:
	- ./iops

debug: clear clean compile
	- valgrind $(VALGRINDFLAGS) ./flops
	- valgrind $(VALGRINDFLAGS) ./iops

compile:
	- $(LD) $(LDFLAGS) -o flops flops.cpp
	- $(LD) $(LDFLAGS) -o iops iops.cpp
#	- $(LD) $(LDNOOPS) -o iops iops.cpp

clean:
	- rm -rf *.o
	- rm -rf *.s
	- rm -rf *.ii
	- rm -rf *.out
	- rm -rf flops_*.txt iops_*.txt
	- rm -rf flops iops

clear:
	- clear
