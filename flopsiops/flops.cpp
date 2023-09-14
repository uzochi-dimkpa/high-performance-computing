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
  long double nflops = 0.0; int size = 65536; unsigned long long int loop_end = 2100000000; // -- 536870912; // -- 429496729;
  float a[size], b[size];
  for (int i = 0; i < size * size; ++i) {a[i] = float(1.7); b[i] = float(3.4);}
  __m256 out; __m256 am = _mm256_loadu_ps(a), bm = _mm256_loadu_ps(b);

  // Time start
  std::chrono::time_point<std::chrono::system_clock> time_start = std::chrono::system_clock::now();


  //// OLD:
  /* ... */
  // Initialize array and floats for flops

  /// NOTE: The size of this array is too large
  // -- float arr[(int) pow(2.0, 32.0)];

  /// NOTE: This array initialization saturates RAM
  // -- float* arr = new float[(int) pow(2.0, 32.0)];

  /// NOTE:
  // The pow() function cannot be called at compile time;
  // it is a 'non-constexpr' function
  // -- const std::size_t size = (std::size_t) pow(2.0, 16.0);
  // -- std::array<float, (std::size_t) pow(2.0, 16.0)> arr2;
  // std::array<float, (std::size_t) (1 << 19)> arr;
  // float* arr2 = new float[size];
  // float a = 1.7, b = 3.4;
  // int th_nflops = 0;
  /* ... */


  /// BEGIN: resource/memory saturation

  // OpenMP parallel
  #pragma omp parallel
  {
    // #pragma omp parallel for
    for (int i = 0; i < loop_end; ++i) {
      out = _mm256_fmadd_ps(am, bm, out);
      __m256 out2 = _mm256_fmadd_ps(am, bm, out2);
      __m256 out3 = _mm256_fmadd_ps(am, bm, out3);
      __m256 out4 = _mm256_fmadd_ps(am, bm, out4);
      __m256 out5 = _mm256_fmadd_ps(am, bm, out5);
      __m256 out6 = _mm256_fmadd_ps(am, bm, out6);
      __m256 out7 = _mm256_fmadd_ps(am, bm, out7);
      __m256 out8 = _mm256_fmadd_ps(am, bm, out8);
      __m256 out9 = _mm256_fmadd_ps(am, bm, out9);
      __m256 out10 = _mm256_fmadd_ps(am, bm, out10);
      __m256 out11 = _mm256_fmadd_ps(am, bm, out11);
      __m256 out12 = _mm256_fmadd_ps(am, bm, out12);
      __m256 out13 = _mm256_fmadd_ps(am, bm, out13);
      __m256 out14 = _mm256_fmadd_ps(am, bm, out14);
      __m256 out15 = _mm256_fmadd_ps(am, bm, out15);
      __m256 out16 = _mm256_fmadd_ps(am, bm, out16);
      __m256 out17 = _mm256_fmadd_ps(am, bm, out17);
      __m256 out18 = _mm256_fmadd_ps(am, bm, out18);
      __m256 out19 = _mm256_fmadd_ps(am, bm, out19);
      __m256 out20 = _mm256_fmadd_ps(am, bm, out20);
      __m256 out21 = _mm256_fmadd_ps(am, bm, out21);
      __m256 out22 = _mm256_fmadd_ps(am, bm, out22);
      __m256 out23 = _mm256_fmadd_ps(am, bm, out23);
      __m256 out24 = _mm256_fmadd_ps(am, bm, out24);
      __m256 out25 = _mm256_fmadd_ps(am, bm, out25);
      __m256 out26 = _mm256_fmadd_ps(am, bm, out26);
      __m256 out27 = _mm256_fmadd_ps(am, bm, out27);
      __m256 out28 = _mm256_fmadd_ps(am, bm, out28);
      __m256 out29 = _mm256_fmadd_ps(am, bm, out29);
      __m256 out30 = _mm256_fmadd_ps(am, bm, out30);
    }
  }

  /// END: resource/memory saturation


  // Time end
  std::chrono::time_point<std::chrono::system_clock> time_end = std::chrono::system_clock::now();

  // Time elapsed
  std::chrono::duration<double> elapsed_seconds = time_end-time_start;

  // Time duration, # of Flops, Flops / sec
  std::cout << "Time elapsed (s): " << elapsed_seconds.count() << std::endl;
  nflops = (16 * 30 * 2) * loop_end;
  if (nflops > 0) {
    std::cout << "Loop end: " << nflops / (16 * 30 * 2) << std::endl;
    std::cout << "# of Flops: " << nflops << std::endl;
    std::cout << "Flops / sec == " << (double) nflops / elapsed_seconds.count() << std::endl;
  }

  // Memory deallocation
  // --

  return 0;
}