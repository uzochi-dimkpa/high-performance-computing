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
  long double niops = 0.0; int size = 65536; unsigned long long int loop_end = 140737488355328; // -- 2100000000; // -- 536870912; // -- 429496729;
  int a[size], b[size];
  for (int i = 0; i < size * size; ++i) {a[i] = 4; b[i] = 7;}
  __m256i out; __m256i am = _mm256_load_si256((__m256i*) a), bm = _mm256_load_si256((__m256i*) b);

  // Time start
  std::chrono::time_point<std::chrono::system_clock> time_start = std::chrono::system_clock::now();


  /// BEGIN: resource/memory saturation

  // OpenMP parallel
  #pragma omp parallel
  {
    #pragma omp parallel for
    for (int i = 0; i < loop_end * loop_end * loop_end * loop_end; ++i) {
      out = _mm256_add_epi8(am, bm);
      __m256i out2 = _mm256_add_epi8(am, bm);
      __m256i out3 = _mm256_add_epi8(am, bm);
      __m256i out4 = _mm256_add_epi8(am, bm);
      __m256i out5 = _mm256_add_epi8(am, bm);
      __m256i out6 = _mm256_add_epi8(am, bm);
      __m256i out7 = _mm256_add_epi8(am, bm);
      __m256i out8 = _mm256_add_epi8(am, bm);
      __m256i out9 = _mm256_add_epi8(am, bm);
      __m256i out10 = _mm256_add_epi8(am, bm);
      __m256i out11 = _mm256_add_epi8(am, bm);
      __m256i out12 = _mm256_add_epi8(am, bm);
      __m256i out13 = _mm256_add_epi8(am, bm);
      __m256i out14 = _mm256_add_epi8(am, bm);
      __m256i out15 = _mm256_add_epi8(am, bm);
      __m256i out16 = _mm256_add_epi8(am, bm);
      __m256i out17 = _mm256_add_epi8(am, bm);
      __m256i out18 = _mm256_add_epi8(am, bm);
      __m256i out19 = _mm256_add_epi8(am, bm);
      __m256i out20 = _mm256_add_epi8(am, bm);
      __m256i out21 = _mm256_add_epi8(am, bm);
      __m256i out22 = _mm256_add_epi8(am, bm);
      __m256i out23 = _mm256_add_epi8(am, bm);
      __m256i out24 = _mm256_add_epi8(am, bm);
      __m256i out25 = _mm256_add_epi8(am, bm);
      __m256i out26 = _mm256_add_epi8(am, bm);
      __m256i out27 = _mm256_add_epi8(am, bm);
      __m256i out28 = _mm256_add_epi8(am, bm);
      __m256i out29 = _mm256_add_epi8(am, bm);
      __m256i out30 = _mm256_add_epi8(am, bm);
      __m256i out31 = _mm256_add_epi8(am, bm);
      __m256i out32 = _mm256_add_epi8(am, bm);
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
  niops *= loop_end;
  niops *= loop_end;
  niops *= loop_end;
  if (niops >= 0) {
    std::cout << "Loop end: " << loop_end << std::endl;
    std::cout << "# of Iops: " << niops << std::endl;
    std::cout << "Iops / sec == " << (double) niops / elapsed_seconds.count()<< std::endl;
  }

  // Memory deallocation
  // --

  return 0;
}