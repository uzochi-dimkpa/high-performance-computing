// #include <fstream>
// #include <array>
#include <vector>

#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>

#include <omp.h>
#include <immintrin.h>



// Print filter
void printFilter(float** arr, int k) {
  std::cout << "Print filter (k = " << k << ")" << std::endl;
  
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      std::cout << arr[i][j] << "  ";
    }

    std::cout << std::endl;
  }
}


// Print image (using vectors)
void printImageVector(std::vector<std::vector<float>>& arr, unsigned long int n, unsigned long int m, int k, std::string name) {
  std::cout << "Print " << name << " (vector) (n = " << n + (k - 1) << ", m = " << m + (k - 1) << ")" << std::endl;

  // 2-dimensional
  /**/
  for (unsigned long int i = 0; i < 50; ++i) {
    for (unsigned long int j = 0; j < 50; ++j) {
      std::cout << arr[i][j] << "\t";
    }

    std::cout << std::endl;
  }
  /**/
  
  // 1-dimensional
  /*
  for (unsigned long int i = 0; i < (n + (k - 1)) * (m + (k - 1)); ++i) {
    std::cout << arr[i] << "\t";
  }
  std::cout << std::endl;
  /**/
}


// Print image (using arrays)
void printImage(float* arr, unsigned long int n, unsigned long int m, int k, std::string name) {
  std::cout << "Print " << name << " (n = " << n + (k - 1) << ", m = " << m + (k - 1) << ")" << std::endl;

  // 2-dimensional
  /**/
  for (unsigned long int i = 0; i < 50; ++i) {
    for (unsigned long int j = 0; j < 50; ++j) {
      std::cout << arr[i * (m + (k - 1)) + j] << "\t";
    }

    std::cout << std::endl;
  }
  /**/
  /*
  for (unsigned long int i = 0; i < n + (k - 1); ++i) {
    for (unsigned long int j = 0; j < m + (k - 1); ++j) {
      std::cout << arr[i * (m + (k - 1)) + j] << "\t";
    }

    std::cout << std::endl;
  }
  /**/
  
  // 1-dimensional
  /*
  for (unsigned long int i = 0; i < (n + (k - 1)) * (m + (k - 1)); ++i) {
    std::cout << arr[i] << "\t";
  }
  std::cout << std::endl;
  /**/
}


// Print output (using vectors)
void printOutputVector(std::vector<std::vector<float>>& arr, unsigned long int n, unsigned long int m, int k, std::string name) {
  std::cout << "Print " << name << " (vector) (n = " << n << ", m = " << m << ")" << std::endl;

  // 2-dimensional
  /**/
  for (unsigned long int i = 0; i < 50; ++i) {
    for (unsigned long int j = 0; j < 50; ++j) {
      std::cout << arr[i][j] << "\t";
    }

    std::cout << std::endl;
  }
  /**/

  // 1-dimensional
  /*
  for (unsigned long int i = 0; i < (n + (k - 1)) * (m + (k - 1)); ++i) {
    std::cout << arr[i] << "\t";
  }
  std::cout << std::endl;
  /**/
}


// Print output (using arrays)
void printOutput(float* arr, unsigned long int n, unsigned long int m, int k, std::string name) {
  std::cout << "Print " << name << " (n = " << n << ", m = " << m << ")" << std::endl;

  // 2-dimensional
  /**/
  for (unsigned long int i = 0; i < 50; ++i) {
    for (unsigned long int j = 0; j < 50; ++j) {
      std::cout << arr[i * (m) + j] << "\t";
    }

    std::cout << std::endl;
  }
  /**/
  /*
  for (unsigned long int i = 0; i < n; ++i) {
    for (unsigned long int j = 0; j < m; ++j) {
      std::cout << arr[i * (m) + j] << "\t";
    }

    std::cout << std::endl;
  }
  /**/

  // 1-dimensional
  /*
  for (unsigned long int i = 0; i < (n + (k - 1)) * (m + (k - 1)); ++i) {
    std::cout << arr[i] << "\t";
  }
  std::cout << std::endl;
  /**/
}


// Convolution calculation (using vectors)
/**/
void convolVector(std::vector<std::vector<float>>& img, std::vector<std::vector<float>>& out, float** filter, unsigned long int _x, unsigned long int _y, unsigned long int n, unsigned long int m, int k) {
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      out[(_x - (k / 2))][(_y - (k / 2))] += (filter[i][j] * img[(_x - (k / 2) + i)][(_y - (k / 2) + j)]);
    }
  }
}
/**/


// Convolution calculation (using arrays)
/**/
void convol(float* img, float* out, float** filter, unsigned long int _x, unsigned long int _y, unsigned long int n, unsigned long int m, int k) {
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      out[(_x - (k / 2)) * m + (_y - (k / 2))] += (filter[i][j] * img[(_x - (k / 2) + i) * (m + (k - 1)) + (_y - (k / 2) + j)]);
    }
  }
}
/**/


// main
int main(int argc, char* argv[]) {
  // Checking argv
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <n>" << " <m>" << " <k>" << std::endl;
    return -1;
  }

  // Parse arguments
  const unsigned long int n = std::atol(argv[1]), m = std::atol(argv[2]), k = std::atol(argv[3]);

  // Generate convolution filter
  float** filter = new float*[k];
  for (int i = 0; i < k; ++i) {
    filter[i] = new float[k];

    for (int j = 0; j < k; ++j) {
      filter[i][j] = (i % 2) + (j % 2);
    }
  }

  // Declare & initialize constants
  const unsigned long int SIZE = n * m;
  const unsigned long int SIZE_PADDED = (n + (k - 1)) * (m + (k - 1));
  
  // Generate & populate input image
  // std::vector<std::vector<float>> image2(n + (k - 1), std::vector<float>(m + (k - 1), 0));
  float* image = new float[SIZE_PADDED]; std::fill_n(image, SIZE_PADDED, 0);
  for (unsigned long int i = (k / 2); i < n + (k / 2); ++i) {
    for (unsigned long int j = (k / 2); j < m + (k / 2); ++j) {
      image[i * (m + (k - 1)) + j] = 1;
      // image2[i][j] = 1;
    }
  }
  /// DEBUG:
  /*
  std::cout << (k / 2) << " --x--> " << (n + (k / 2)) << std::endl;
  std::cout << (k / 2) << " --y--> " << (m + (k / 2)) << std::endl;
  std::cout << "n: " << (n + (k - 1)) << ", m: " << (m + (k - 1)) << std::endl;
  for (unsigned long int i = (k / 2); i < n + (k / 2); ++i) {
    for (unsigned long int j = (k / 2); j < m + (k / 2); ++j) {
      std::cout << "(" << i << "," << j << ")";
      std::cout << " pos: " << i * (m + (k - 1)) + j << "   ";
    }

    std::cout << std::endl;
  }
  /**/

  // Generate output image
  // std::vector<std::vector<float>> out2(n, std::vector<float>(m, 0));
  float* out = new float[SIZE]; std::fill_n(out, SIZE, 0);
  
  
  /// TODO: Make model plots
  /* ... */
  
  
  // Time start
  std::chrono::time_point<std::chrono::high_resolution_clock> time_start = std::chrono::high_resolution_clock::now();


  /// TODO: Convolution
  /// COMPLETE:
  /*
  for (unsigned long int i = (k / 2); i < n + (k / 2); ++i) {
    for (unsigned long int j = (k / 2); j < m + (k / 2); ++j) {
      convol(image, out, filter, i, j, n, m, k);
      // convolVector(image2, out2, filter, i, j, n, m, k);
    }
  }
  /**/


  /// TODO: Convolution optimized
  /// INCOMPLETE:
  /**/
  #pragma omp parallel for num_threads(16) schedule(dynamic, 1024) collapse(2)
    for (unsigned long int i = (k / 2); i < n + (k / 2); ++i) {
      for (unsigned long int j = (k / 2); j < m + (k / 2); ++j) {
        convol(image, out, filter, i, j, n, m, k);
        // convolVector(image2, out2, filter, i, j, n, m, k);
      }
    }
  /**/


  // Time end
  std::chrono::time_point<std::chrono::high_resolution_clock> time_end = std::chrono::high_resolution_clock::now();
  

  // Time elapsed
  std::chrono::duration<double> elapsed_seconds = time_end - time_start;
  

  // Time duration & result/data output
  std::cout << argv[0] << " " << n << " " << m << " " << k << std::endl; // --- " parallel: " << true <<
  std::cout << "Time elapsed (s): " << elapsed_seconds.count() << std::endl;
  std::cout << "SIZE: " << SIZE << std::endl;
  std::cout << "SIZE_PADDED: " << SIZE_PADDED << std::endl;
  printFilter(filter, k); printImage(image, n, m, k, "image"); printOutput(out, n, m, k, "output");
  // printFilter(filter, k); printImageVector(image2, n, m, k, "image"); printOutputVector(out2, n, m, k, "output");
  std::cout << "\n\n\n\n";
  

  // Memory deallocation
  for (int i = 0; i < k; ++i) delete[] filter[i]; delete[] filter;
  delete[] image; delete[] out;
  // delete &n; delete &m; delete &k; delete &image; delete &out;
  // ---

  
  return 0;
}