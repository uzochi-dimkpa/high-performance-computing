#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>



// Print CUDA error
void CUDA_err() {
  std::cout << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
};


// CUDA globally shared memory
extern __shared__ float filter[];


// CUDA convolution calculation
__device__ void cnvl_calc(float* d_image, float* d_out, float* filter, std::uint64_t n, std::uint64_t m, std::uint64_t k, std::uint64_t _x, std::uint64_t _y) {
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      d_out[(_x - (k / 2)) * m + (_y - (k / 2))] += (filter[i * k + j] * d_image[(_x - (k / 2) + i) * (m + (k - 1)) + (_y - (k / 2) + j)]);
    }
  }
}


// CUDA convolution
__global__ void cnvl(float* d_image, float* d_out, std::uint64_t n, std::uint64_t m, std::uint64_t k) {
  // Calculate index
  // std::uint32_t index = threadIdx.x + (blockIdx.x * blockDim.x);
  
  // Initialize & populate filter kernel on shared memory.
  // extern __shared__ float filter[];

  // 1 thread populate filter
  /**/
  for (std::uint32_t i = 0; i < k*k; ++i) filter[i] = i % 2;
  /**/
  
  // 1 thread print filter
  /**/
  printf("Print filter (k = %d)\n", k);
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      printf("%.0f  ", filter[i * k + j]);
    } printf("\n");
  }
  /**/

  // Calculate boundary conditions
  std::uint64_t dimx = n + (k / 2);
  std::uint64_t dimy = m + (k / 2);
  
  /// CONVOLUTION:
  // 1 thread
  /**/
  for (std::uint64_t _x = (k / 2); _x < dimx; ++_x) {
    for (std::uint64_t _y = (k / 2); _y < dimy; ++_y) {
      cnvl_calc(d_image, d_out, filter, n, m, k, _x, _y);
    }
  }
  /**/
}


// CUDA convolution
__global__ void cnvl_opt(float* d_image, float* d_out, std::uint64_t n, std::uint64_t m, std::uint64_t k) {
  // Calculate index values
  std::uint64_t index_x = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint64_t index_y = blockIdx.y * blockDim.y + threadIdx.y;
  // printf("(index_x, index_y): (%d, %d)\n", index_x, index_y);

  // Synchronize threads
  __syncthreads();
  // extern __shared__ float filter2[];
  // Multithreaded populate filter
  /**/

  // if (index_x < k && index_y < k) filter[index_x * k + index_y] = (index_x + index_y) % 2;
  if (index_x < k && index_y < k) filter[index_x * k + index_y] = 0;
  /**/

  // Synchronize threads
  __syncthreads();

  if (index_x == 1 && index_y == 1) filter[index_x * k + index_y] = 1;

  // Synchronize threads
  __syncthreads();
  
  // Multithreaded print filter
  /**/
  if (index_x == 0 && index_y == 0) printf("Print filter (k = %d)\n", k);

  // Synchronize threads
  __syncthreads();
  
  if (index_x < k && index_y < k) printf("%.0f  ", filter[index_x * k + index_y]);

  // Synchronize threads
  __syncthreads();
  
  if (index_x == 0 && index_y == 0) printf("\n");
  /**/
  
  // Synchronize threads
  __syncthreads();

  // Calculate boundary conditions
  std::uint64_t dimx = n + (k / 2);
  std::uint64_t dimy = m + (k / 2);

  /// CONVOLUTION:
  // Multithreaded
  /**/
  if (index_x < n && index_y < m) {
    for (std::uint32_t i = 0; i < k; ++i) {
      for (std::uint32_t j = 0; j < k; ++j) {
        // printf("(index_x, index_y): (%d, %d)\n", index_x, index_y);
        // printf("image[%d][%d]: %.2f\n", index_x, index_y, d_image[index_x * m + index_y]);

        // d_out[(index_x - (k / 2)) * m + (index_y - (k / 2))] += (filter[i * k + j] * d_image[(index_x - (k / 2) + i) * (m + (k - 1)) + (index_y - (k / 2) + j)]);
        
        
        // d_out[(index_x - (k / 2)) * (m + k - 1) + (index_y - (k / 2))] += (filter[i * k + j] * d_image[(index_x - (k / 2) + i) * (m + (k - 1)) + (index_y - (k / 2) + j)]);
        // d_out[(index_x - (k / 2)) * (m + (k / 2)) + (index_y - (k / 2))] += (filter[i * k + j] * d_image[(index_x - (k / 2) + i) * (m + (k - 1)) + (index_y - (k / 2) + j)]);
        // d_out[(index_x) * (m + (k / 2)) + (index_y)] += (filter[i * k + j] * d_image[(index_x - (k / 2) + i) * (m + (k - 1)) + (index_y - (k / 2) + j)]);
        // d_out[(index_x * m + index_y)] += (filter[i * k + j] * d_image[(index_x - (k / 2) + i) * (m + (k - 1)) + (index_y - (k / 2) + j)]);
        // d_out[(index_x * m + index_y)] += (filter[i * k + j] * d_image[(index_x + i) * (m + (k - 1)) + (index_y + j)]);

        // d_out[(index_x * m + index_y)] = 7; // Is seeing the entire image and writing to the entire output
        // d_out[(index_x * m + index_y)] = blockIdx.x * 100 + blockIdx.y; // Prints the block ID on every pixel
        // d_out[(index_x * m + index_y)] = threadIdx.x * 100 + threadIdx.y; // Prints the thread ID on every pixel
        // So the left-hand side of the `=` is correct, but the right-hand side
        // d_out[(index_x * m + index_y)] = d_image[(index_x + i) * (m + (k - 1)) + (index_y + j)]; // Does not work
        // d_out[(index_x * m + index_y)] = d_image[(index_x + j) * (m + (k - 1)) + (index_y + i)]; // No difference
        // d_out[(index_x * m + index_y)] = filter[i * k + j] + 5; // Reads entire image, writes to entire output
        // d_out[(index_x * m + index_y)] += filter[i * k + j]; // `+=` breaks it for some reason
        // d_out[0] = 9.75; d_out[31] = 9.75; d_out[32] = 9.75; // Writes values to where they should be; working properly
        // d_out[index_x * m + index_y] += 9.75; // Writes values to where they should be; working properly
        // d_out[index_x * m + index_y] += filter[1] * 2; // Writes the correct values, but only to the first 32x32 block in the grid
        // For some reason, indexing the filter is what breaks the writing output
        // Attemptig to try to index only image to see if that works
        // d_out[index_x * m + index_y] += d_image[m * k + (k / 2) + 1] * 3; // Writes correct values to entire output; sees the entire image and output

        // Indexing the image only works; as in, the entire output file is update
        
        /// PROBLEM: For some reason, indexing the filter is what breaks the writing output
        // d_out[index_x * m + index_y] = -1; // Writes to the entire image
        // d_out[index_x * m + index_y] += d_image[(index_x + i) * (m + (k - 1)) + (index_y + j)] * 4; // Writes correct values to entire output; sees the entire image and output
        
        /// PROBLEM: For some reason, indexing the filter is what breaks the writing output
        // d_out[index_x * m + index_y] += 12; // Writes to every pixel of the output image
        // d_out[index_x * m + index_y] += filter[1] * 6; // Only writes to the first block
        
        /// DEDUCTION: OF: PROBLEM: SOURCE:
        // Filter doesn't seem to exist on any other block. Either this, or maybe because `index_x` and
        // `index_y` aren't allowed to go beyond n & m respectively, the filter isn't queried because
        // the indices `index_x` abd `index_y` are beyond the scope of the if-condition boundary.
        
        d_out[index_x * m + index_y] += 12 * d_image[(index_x + i) * (m + (k - 1)) + (index_y + j)]; // Writes to every pixel of the output image
        d_out[index_x * m + index_y] += filter[i * k + j] * 6; // Only writes to the first block
      }
    }
  }

  // Synchronize threads
  __syncthreads();

  // if (filter[1] != NULL) printf("Filter exists on block {%d, %d}! ~ ", blockIdx.x, blockIdx.y);

  // Synchronize threads
  __syncthreads();

  // if (index_x == 0 && index_y == 0) printf("\n");
  /**/

  // Synchronize threads
  __syncthreads();

  /// DEBUG:
  /*
  printf ("threadIdx={%d, %d, %d} blockIdx={%d, %d, %d} blockDim={%d, %d, %d} gridDim={%d, %d, %d}\n",
	  threadIdx.x, threadIdx.y, threadIdx.z,
	  blockIdx.x, blockIdx.y, blockIdx.z,
	  blockDim.x, blockDim.y, blockDim.z,
	  gridDim.x, gridDim.y, gridDim.z);
  
  // Synchronize threads
  // __syncthreads();

  // printf("X CALC: %d\n", (threadIdx.x + blockIdx.x * blockDim.x));
  // printf("Y CALC: %d\n", (threadIdx.y + blockIdx.y * blockDim.y));
  /**/
}


// Print filter
void printFilter(float* arr, int k) {
  std::cout << "Print filter (k = " << k << ")" << std::endl;
  
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      std::cout << arr[i * k + j] << "  ";
    } std::cout << std::endl;
  }
}


// Print image (using arrays)
void printImage(float* arr, unsigned long int n, unsigned long int m, std::uint32_t k, std::string name) {
  std::cout << "Print " << name << " (n = " << n + (k - 1) << ", m = " << m + (k - 1) << ")" << std::endl;

  // Image dimension check
  int dimx = n + (k - 1), dimy = m + (k - 1);
  if (dimx > 50) dimx = 50; if (dimy > 50) dimy = 50;

  // Print image
  for (std::uint64_t i = 0; i < dimx; ++i) {
    for (std::uint64_t j = 0; j < dimy; ++j) {
      std::cout << arr[i * (m + (k - 1)) + j] << "\t";
    } std::cout << std::endl;
  }
}


// Print output (using arrays)
void printOutput(float* arr, std::uint64_t n, std::uint64_t m, std::uint32_t k, std::string name) {
  std::cout << "Print " << name << " (n = " << n << ", m = " << m << ")" << std::endl;

  // Output dimension check
  int dimx = n, dimy = m;
  if (dimx > 50) dimx = 50; if (dimy > 50) dimy = 50;

  // Print output
  for (std::uint64_t i = 0; i < dimx; ++i) {
    for (std::uint64_t j = 0; j < dimy; ++j) {
      std::cout << arr[i * (m) + j] << "\t";
    } std::cout << std::endl;
  }
}


// Convolution calculation (using arrays)
void convol(float* img, float* out, float* filter, std::uint64_t _x, std::uint64_t _y, std::uint64_t n, std::uint64_t m, std::uint64_t k) {
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      out[(_x - (k / 2)) * m + (_y - (k / 2))] += (filter[i * k + j] * img[(_x - (k / 2) + i) * (m + (k - 1)) + (_y - (k / 2) + j)]);
    }
  }
}


// main
int main(int argc, char* argv[]) {
  // Declaring arguments
  std::uint64_t n, m, k;

  // Checking argv
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " <n> <m> <k>" << std::endl;
    return -1;
  }

  // Parse & initialize arguments
  n = std::atol(argv[1]), m = std::atol(argv[2]), k = std::atol(argv[3]);
  // numBlocks = std::atoll(argv[4]), blockSize = std::atoll(argv[5]);

  // Declare convolution filter host copy
  /*
  float* filter = new float[k * k];
  for (int i = 0; i < k*k; ++i) filter[i] = i % 2;
  /**/

  // Declare & initialize constants
  std::uint64_t SIZE = n * m;
  std::uint64_t SIZE_PADDED = (n + (k - 1)) * (m + (k - 1));

  // Generate & populate input image
  float *image = new float[SIZE_PADDED]; std::fill_n(image, SIZE_PADDED, 0);
  for (std::uint64_t i = (k / 2); i < n + (k / 2); ++i) {
    for (std::uint64_t j = (k / 2); j < m + (k / 2); ++j) {
      image[i * (m + (k - 1)) + j] = 1;

      if (i == 10 && j == 16) image[i * (m + (k - 1)) + j] = 2;
    }
  }

  // Generate output image
  float *out = new float[SIZE]; std::fill_n(out, SIZE, 0);

  // Declare CUDA variables;
  float *d_image, *d_out;
  // float *d_filter;

  // Declare & initialize CUDA kernel block count dimensions
  std::uint32_t numBlockx, numBlocky, nBlockx, nBlocky;
  if (n < 32) {
    numBlockx = 1;
  } else if (n % 32 == 0) {
    numBlockx = n / 32;
  } else {
    numBlockx = (n / 32) + 1;
  }
  // nBlockx = (numBlockx + 32 - 1) / 32;

  if (m < 32) {
    numBlocky = 1;
  } else if (m % 32 == 0) {
    numBlocky = m / 32;
  } else {
    numBlocky = (m / 32) + 1;
  }
  // nBlocky = (numBlocky + 32 - 1) / 32;

  // Declare & initialize CUDA kernel dimensions
  dim3 gridSize (numBlockx, numBlocky);
  dim3 blockSize (32, 32);

  // Time start
  std::chrono::time_point<std::chrono::high_resolution_clock> time_start = std::chrono::high_resolution_clock::now();

  /// TODO: CUDA Convolution
  cudaMalloc((void **) &d_image, SIZE_PADDED * sizeof(float));
  cudaMalloc((void **) &d_out, SIZE * sizeof(float));
  // cudaMalloc((void **) &d_filter, k * k * sizeof(float));

  cudaMemcpy(d_image, image, SIZE_PADDED * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_filter, filter, k * k * sizeof(float), cudaMemcpyHostToDevice);

  // cudaFuncSetCacheConfig(cnvl_opt, 2);

  // 1 thread
  /*
  cnvl<<<1, 1, k*k*sizeof(float)>>>(d_image, d_out, n, m, k);
  /**/

  // Multithreaded
  /**/
  cnvl_opt<<<gridSize, blockSize, k*k*sizeof(float)>>>(d_image, d_out, n, m, k);
  /**/

  CUDA_err();

  cudaMemcpy(out, d_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(filter, d_filter, k * k * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_image); cudaFree(d_out);
  // cudaFree(d_filter);
  /// TODO: CUDA Convolution

  // Sequential convolution
  /*
  for (std::uint64_t i = (k / 2); i < n + (k / 2); ++i) {
    for (std::uint64_t j = (k / 2); j < m + (k / 2); ++j) {
      convol(image, out, filter, i, j, n, m, k);
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
  std::cout << "Grid dimensions: (" << numBlockx << ", " << numBlocky << ")" << std::endl;
  // std::cout << "nBlocks: (" << nBlockx << ", " << nBlocky << ")" << std::endl;
  // printFilter(filter, k);
  printImage(image, n, m, k, "image"); printOutput(out, n, m, k, "output");

  // Memory Dellocation
  delete[] image; delete[] out;
  // delete[] filter;

  return 0;
}