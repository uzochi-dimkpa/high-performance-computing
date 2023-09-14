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
  long double niops = 0.0; int size = 65536; unsigned long long int loop_end = 7200000000; // -- 14400000000; // -- 2100000000; // -- 536870912; // -- 429496729;
  int a[size], b[size];
  for (int i = 0; i < size * size; ++i) {a[i] = 4; b[i] = 7;}
  __m256i out; __m256i am = _mm256_load_si256((__m256i*) a); // -- , bm = _mm256_load_si256((__m256i*) b);

  // Time start
  std::chrono::time_point<std::chrono::system_clock> time_start = std::chrono::system_clock::now();


  /// BEGIN: resource/memory saturation

  // OpenMP parallel
  #pragma omp parallel
  {
    // #pragma omp parallel for
    for (unsigned long long int i = 0; i < loop_end; ++i) {
      out = _mm256_add_epi8(am, out);
      __m256i out2 = _mm256_add_epi8(am, out2);
      __m256i out3 = _mm256_add_epi8(am, out3);
      __m256i out4 = _mm256_add_epi8(am, out4);
      __m256i out5 = _mm256_add_epi8(am, out5);
      __m256i out6 = _mm256_add_epi8(am, out6);
      __m256i out7 = _mm256_add_epi8(am, out7);
      __m256i out8 = _mm256_add_epi8(am, out8);
      __m256i out9 = _mm256_add_epi8(am, out9);
      __m256i out10 = _mm256_add_epi8(am, out10);
      __m256i out11 = _mm256_add_epi8(am, out11);
      __m256i out12 = _mm256_add_epi8(am, out12);
      __m256i out13 = _mm256_add_epi8(am, out13);
      __m256i out14 = _mm256_add_epi8(am, out14);
      __m256i out15 = _mm256_add_epi8(am, out15);
      __m256i out16 = _mm256_add_epi8(am, out16);
      __m256i out17 = _mm256_add_epi8(am, out17);
      __m256i out18 = _mm256_add_epi8(am, out18);
      __m256i out19 = _mm256_add_epi8(am, out18);
      __m256i out20 = _mm256_add_epi8(am, out19);
      __m256i out21 = _mm256_add_epi8(am, out20);
      __m256i out22 = _mm256_add_epi8(am, out22);
      __m256i out23 = _mm256_add_epi8(am, out23);
      __m256i out24 = _mm256_add_epi8(am, out24);
      __m256i out25 = _mm256_add_epi8(am, out25);
      __m256i out26 = _mm256_add_epi8(am, out26);
      __m256i out27 = _mm256_add_epi8(am, out27);
      __m256i out28 = _mm256_add_epi8(am, out28);
      __m256i out29 = _mm256_add_epi8(am, out29);
      __m256i out30 = _mm256_add_epi8(am, out30);
      __m256i out31 = _mm256_add_epi8(am, out31);
      __m256i out32 = _mm256_add_epi8(am, out32);
    }
  }

  /// END: resource/memory saturation


  // Time end
  std::chrono::time_point<std::chrono::system_clock> time_end = std::chrono::system_clock::now();

  // Time elapsed
  std::chrono::duration<double> elapsed_seconds = time_end-time_start;

  // Time duration, # of Flops, Flops / sec
  std::cout << "Time elapsed (s): " << elapsed_seconds.count() << std::endl;
  niops = (16 * 32) * loop_end;
  if (niops >= 0) {
    std::cout << "Loop end: " << niops / (16 * 32) << std::endl;
    std::cout << "# of Iops: " << niops << std::endl;
    std::cout << "Iops / sec == " << (double) niops / elapsed_seconds.count()<< std::endl;
  }

  // Memory deallocation
  // --

  return 0;
}