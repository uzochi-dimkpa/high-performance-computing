#include <iostream>
#include <chrono>
#include <omp.h>
#include <immintrin.h>



int main() {
  //// WARNING:
  // Make sure to use the `-Ofast` and `-ffast-math`
  // flags when compiling, otherwise your CPU will
  // begin overclocking and won't stop until the
  // program runs to its conclusion
  
  // Initialize necessary variables, registers, and arrays
  long double niops = 0.0; int size = 65536; unsigned long long int loop_end = 720000000; // -- 14400000000; // -- 2100000000; // -- 536870912; // -- 429496729;
  int a[size], b[size];
  for (int i = 0; i < size; ++i) {a[i] = 4; b[i] = 7;}
  __m256i am = _mm256_load_si256((__m256i*) a) , bm = _mm256_load_si256((__m256i*) b);

  // Time start
  std::chrono::time_point<std::chrono::system_clock> time_start = std::chrono::system_clock::now();


  /// BEGIN: resource/memory saturation

  // OpenMP parallel
  #pragma omp parallel
  {
    //// OLD:
    // -- //
    // __m256i out = _mm256_load_si256((__m256i*) a);
    // __m256i out2= _mm256_load_si256((__m256i*) a+8);
    // __m256i out3= _mm256_load_si256((__m256i*) a+16);
    // __m256i out7= _mm256_load_si256((__m256i*) a+24);
    // __m256i out8= _mm256_load_si256((__m256i*) a+32);
    // __m256i out9= _mm256_load_si256((__m256i*) a+40);
    // -- //

    //// OLD:
    // -- //
    // __m256i out4= _mm256_load_si256((__m256i*) a+16);
    // __m256i out5= _mm256_load_si256((__m256i*) a+24);
    // __m256i out6= _mm256_load_si256((__m256i*) a+32);
    // __m256i out10= _mm256_load_si256((__m256i*) a+64);
    // __m256i out11= _mm256_load_si256((__m256i*) a+72);
    // __m256i out12= _mm256_load_si256((__m256i*) a+80);
    // -- //

    //// NEW:
    // -- //
    __m256i out;
    __m256i out2;
    __m256i out3;
    __m256i out7;
    __m256i out8;
    __m256i out9;
    // -- //

    // #pragma omp parallel for
    for (unsigned long long int i = 0; i < loop_end; ++i) {
      //// OLD:
      // -- //
      // out7 = _mm256_add_epi8(am, out);
      // out8 = _mm256_add_epi8(am, out2);
      // out9 = _mm256_add_epi8(am, out3);
      // out = _mm256_add_epi8(bm, out7);
      // out2 = _mm256_add_epi8(bm, out8);
      // out3 = _mm256_add_epi8(bm, out9);
      // -- //
      
      //// NEW:
      // -- //
      out7 = _mm256_add_epi8(am, out);
      out8 = _mm256_add_epi8(out, out2);
      out9 = _mm256_add_epi8(out2, out3);
      out = _mm256_add_epi8(bm, out7);
      out2 = _mm256_add_epi8(out7, out8);
      out3 = _mm256_add_epi8(out8, out9);
      // -- //
    }
    //// OLD:
    // -- //
    // out = _mm256_add_epi8(out, out2);
    // out = _mm256_add_epi8(out, out3);
    // out = _mm256_add_epi8(out, out7);
    // out = _mm256_add_epi8(out, out8);
    // out = _mm256_add_epi8(out, out9);
    // _mm256_store_si256((__m256i*) a, out);
    // -- //

    //// NEW:
    // -- //
    out = _mm256_add_epi8(out, out2);
    out = _mm256_add_epi8(out3, out7);
    out = _mm256_add_epi8(out8, out9);
    _mm256_store_si256((__m256i*) a, out);
    // -- //
  }

  /// END: resource/memory saturation


  // Time end
  std::chrono::time_point<std::chrono::system_clock> time_end = std::chrono::system_clock::now();

  // Time elapsed
  std::chrono::duration<double> elapsed_seconds = time_end-time_start;

  // Time duration, # of Flops, Flops / sec
  std::cout << "Time elapsed (s): " << elapsed_seconds.count() << std::endl;
  int cores=16;
  int nbinstrperloop=6;
  int nbopperinst=256/8;
  niops = (cores*nbinstrperloop*nbopperinst) * loop_end;
  if (niops >= 0) {
    std::cout << "Loop end: " << niops / (cores*nbinstrperloop*nbopperinst) << std::endl;
    std::cout << "# of Iops: " << niops << std::endl;
    std::cout << "Iops / sec == " << (double) niops / elapsed_seconds.count()<< std::endl;
  }

  // Memory deallocation
  // --

  return 0;
}