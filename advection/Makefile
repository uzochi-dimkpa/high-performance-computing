LD=g++

all: clean compile run

run: clean compile
	- ./advection 103 0.0009
	- ./advection 1003 0.00009

test: clean compile
	- ./advection 40 0.01
	- ./advection 20 0.005

debug: clean compile
	- valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all -s ./advection 200 0.000005

queue:
	- sh queue_advection.sh

advec:
	- ./advection 103 0.0009
	- ./advection 1003 0.00009

advec_test:
	- ./advection 40 0.01
	- ./advection 20 0.005

advec_debug:
	- valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all -s ./advection 200 0.000005

compile:
	- $(LD) -o advection advection.cpp

clean:
	- rm -f *.o
	- rm -f *.out
	- rm -f advec_*.txt
	- rm -f advection

clear:
	- clear
