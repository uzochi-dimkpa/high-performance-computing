LD=g++
LDOMP=g++ -fopenmp
LDFLAGS=-fopenmp -O3 -mfma -mavx2 -march=znver2 -mtune=znver2 -ffast-math -save-temps -fverbose-asm
LDNOOPS=-fopenmp -mfma -mavx2 -save-temps -fverbose-asm
VALGRINDFLAGS=--track-origins=yes --leak-check=full --show-leak-kinds=all --show-reachable=yes --log-file=VALGRIND.log -s
CACHEGRINDFLAGS=--tool=cachegrind --log-file=CACHEGRIND.log
HELGRINDFLAGS=--tool=helgrind --log-file=HELGRIND.log
CALLGRINDFLAGSS=--tool=callgrind --log-file=CALLGRIND.log



all: clean compile run

squential: clean_sequential compile_sequential run_all_sequential

parallel: clean_parallel compile_parallel run_all_parallel

# ops: clean compile_ops run_all

# noops: clean compile_noops run_all

plot:
	- rm -rf perf-model-plt*.png
	- python plot.py



queue:
	- sh convolution.sh

queue_static:
	- sbatch --partition=Centaurus --chdir=`pwd` --time=24:00:00 --nodes=1 --ntasks-per-node=16 --mem=120G --mail-type=END,FAIL,REQUEUE,TIME_LIMIT --mail-user=udimkpa@uncc.edu --job-name=convl convol.sh

queue_dynamic:
	- sbatch --partition=Centaurus --chdir=`pwd` --time=24:00:00 --nodes=1 --ntasks-per-node=16 --mem=120G --mail-type=END,FAIL,REQUEUE,TIME_LIMIT --mail-user=udimkpa@uncc.edu --job-name=convlpar convol_par.sh



debug: clean clean_parallel clean_debug compile compile_parallel
	- valgrind $(VALGRINDFLAGS) ./convolution 50 50 3 >> debug_seq.out
	- valgrind $(VALGRINDFLAGS) ./convolution 50 100 15 >> debug_seq.out
	- valgrind $(VALGRINDFLAGS) ./convolution 2048 1024 9 >> debug_seq.out
	- valgrind $(VALGRINDFLAGS) ./convolution_par 50 50 3 >> debug_par.out
	- valgrind $(VALGRINDFLAGS) ./convolution_par 50 100 15 >> debug_par.out
	- valgrind $(VALGRINDFLAGS) ./convolution_par 2048 1024 9 >> debug_par.out
#	- valgrind $(VALGRINDFLAGS) ./convolution 4194304 768 3 >> debug.out
# --- terminate called after throwing an instance of 'std::bad_array_new_length' what():  std::
# --- Killed Error 137 (ignored)
#	- valgrind $(VALGRINDFLAGS) ./convolution 16777216 768 3 >> debug.out
# --- Segmentation fault (ignored)



simple: clean compile
	- ./convolution 100 200 3 >> convolution_simple.out
	- ./convolution 400 900 15 >> convolution_simple.out
	- ./convolution 8192 1024 7 >> convolution_simple.out
#	- ./convolution 8192 8192 3 >> convolution_simple.out
#	- ./convolution 4194304 768 3 >> convolution_simple.out
#	- ./convolution 16777216 768 3 >> convolution_simple.out
# --- Killed (ignored)



run: run_all_sequential run_all_parallel

run_all_sequential:
	- ./convolution 1024 768 3 >> convolution_k3_.out
	- ./convolution 2048 2048 3 >> convolution_k3_.out
	- ./convolution 8192 8192 3 >> convolution_k3_.out
#	- ./convolution 4194304 768 3 >> convolution_k3_.out
#	- ./convolution 16777216 768 3 >> convolution_k3_.out

	- ./convolution 1024 768 5 >> convolution_k5_.out
	- ./convolution 2048 2048 5 >> convolution_k5_.out
	- ./convolution 8192 8192 5 >> convolution_k5_.out
#	- ./convolution 4194304 768 5 >> convolution_k5_.out
#	- ./convolution 16777216 768 5 >> convolution_k5_.out

	- ./convolution 1024 768 7 >> convolution_k7_.out
	- ./convolution 2048 2048 7 >> convolution_k7_.out
	- ./convolution 8192 8192 7 >> convolution_k7_.out
#	- ./convolution 4194304 768 7 >> convolution_k7_.out
#	- ./convolution 16777216 768 7 >> convolution_k7_.out

	- ./convolution 1024 768 9 >> convolution_k9_.out
	- ./convolution 2048 2048 9 >> convolution_k9_.out
	- ./convolution 8192 8192 9 >> convolution_k9_.out
#	- ./convolution 4194304 768 9 >> convolution_k9_.out
#	- ./convolution 16777216 768 9 >> convolution_k9_.out

	- ./convolution 1024 768 11 >> convolution_k11_.out
	- ./convolution 2048 2048 11 >> convolution_k11_.out
	- ./convolution 8192 8192 11 >> convolution_k11_.out
#	- ./convolution 4194304 768 11 >> convolution_k11_.out
#	- ./convolution 16777216 768 11 >> convolution_k11_.out

	- ./convolution 1024 768 13 >> convolution_k13_.out
	- ./convolution 2048 2048 13 >> convolution_k13_.out
	- ./convolution 8192 8192 13 >> convolution_k13_.out
#	- ./convolution 4194304 768 13 >> convolution_k13_.out
#	- ./convolution 16777216 768 13 >> convolution_k13_.out

	- ./convolution 1024 768 15 >> convolution_k15_.out
	- ./convolution 2048 2048 15 >> convolution_k15_.out
	- ./convolution 8192 8192 15 >> convolution_k15_.out
#	- ./convolution 4194304 768 15 >> convolution_k15_.out
#	- ./convolution 16777216 768 15 >> convolution_k15_.out



run_all_parallel:
	- ./convolution_par 1024 768 3 >> convolution_k3_p.out
	- ./convolution_par 2048 2048 3 >> convolution_k3_p.out
	- ./convolution_par 8192 8192 3 >> convolution_k3_p.out
#	- ./convolution_par 4194304 768 3 >> convolution_k3_p.out
#	- ./convolution_par 16777216 768 3 >> convolution_k3_p.out

	- ./convolution_par 1024 768 5 >> convolution_k5_p.out
	- ./convolution_par 2048 2048 5 >> convolution_k5_p.out
	- ./convolution_par 8192 8192 5 >> convolution_k5_p.out
#	- ./convolution_par 4194304 768 5 >> convolution_k5_p.out
#	- ./convolution_par 16777216 768 5 >> convolution_k5_p.out

	- ./convolution_par 1024 768 7 >> convolution_k7_p.out
	- ./convolution_par 2048 2048 7 >> convolution_k7_p.out
	- ./convolution_par 8192 8192 7 >> convolution_k7_p.out
#	- ./convolution_par 4194304 768 7 >> convolution_k7_p.out
#	- ./convolution_par 16777216 768 7 >> convolution_k7_p.out

	- ./convolution_par 1024 768 9 >> convolution_k9_p.out
	- ./convolution_par 2048 2048 9 >> convolution_k9_p.out
	- ./convolution_par 8192 8192 9 >> convolution_k9_p.out
#	- ./convolution_par 4194304 768 9 >> convolution_k9_p.out
#	- ./convolution_par 16777216 768 9 >> convolution_k9_p.out

	- ./convolution_par 1024 768 11 >> convolution_k11_p.out
	- ./convolution_par 2048 2048 11 >> convolution_k11_p.out
	- ./convolution_par 8192 8192 11 >> convolution_k11_p.out
#	- ./convolution_par 4194304 768 11 >> convolution_k11_p.out
#	- ./convolution_par 16777216 768 11 >> convolution_k11_p.out

	- ./convolution_par 1024 768 13 >> convolution_k13_p.out
	- ./convolution_par 2048 2048 13 >> convolution_k13_p.out
	- ./convolution_par 8192 8192 13 >> convolution_k13_p.out
#	- ./convolution_par 4194304 768 13 >> convolution_k13_p.out
#	- ./convolution_par 16777216 768 13 >> convolution_k13_p.out

	- ./convolution_par 1024 768 15 >> convolution_k15_p.out
	- ./convolution_par 2048 2048 15 >> convolution_k15_p.out
	- ./convolution_par 8192 8192 15 >> convolution_k15_p.out
#	- ./convolution_par 4194304 768 15 >> convolution_k15_p.out
#	- ./convolution_par 16777216 768 15 >> convolution_k15_p.out



compile: compile_sequential compile_parallel

compile_sequential:
	- $(LD) -o convolution convolution.cpp

compile_parallel:
	- $(LDOMP) -o convolution_par convolution_par.cpp

# compile_ops:
# 	- $(LD) $(LDFLAGS) -o convolution convolution.cpp

# compile_noops:
# 	- $(LD) $(LDNOOPS) -o convolution convolution.cpp



clean: clean_sequential clean_parallel clean_debug

clean_sequential:
	- rm -rf *.o
	- rm -rf *.s
	- rm -rf *.ii
	- rm -rf *_.out !debug_seq.out
	- rm -rf convolution*_.txt
	- rm -rf convolution

clean_parallel:
	- rm -rf *.o
	- rm -rf *.s
	- rm -rf *.ii
	- rm -rf *_p.out !debug_par.out
	- rm -rf convolution*_par.txt
	- rm -rf convolution_par

clean_debug:
	- rm -rf *.o
	- rm -rf *.s
	- rm -rf *.ii
	- rm -rf debug_seq.out debug_par.out
	- rm -rf convolution convolution_par
#	- rm -rf convolution*_.txt convolution*_par.txt



clear:
	- clear
