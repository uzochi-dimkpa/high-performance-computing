#!/bin/sh
make clean compile
make advec > advec_out.txt
make advec_test > advec_test_out.txt
make advec_debug > advec_debug_out.txt
