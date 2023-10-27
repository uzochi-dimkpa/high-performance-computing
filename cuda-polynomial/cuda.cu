#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

const int N (1024);



// Print CUDA error
void CUDA_err() {
  std::cout << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
};


// CUDA polynomial expansion kernel
__global__ void expand_poly(float *d_x, float *d_coeff, int size, int sizeCoeff) {
  // Declare & initialize index
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  
  // Synchronize threads
  // __syncthreads();
  
  // Polynomial expansion
  if (index < size) {
    float temp  = 0.;

    // float out = 0.
    for (int i = 0; i < sizeCoeff; ++i) {
      temp += d_coeff[i] * std::pow(d_x[index], i);
    }
    
    d_x[index] = temp;
  }

  /// NOTE:
  /// Possibly does not read values of array copied over to device.
  /// Perhaps printf happens before data is copied over?
  // printf("index: %d\n", index);
  // printf("d_x[%d]: %d\n", index, d_x[index]);
  // printf("d_coeff[%d]: %d\n", index, d_coeff[index]);
};


// main
int main(int argc, char* argv[]) {
  // Declaring arguments
  int a, b, d;

  // Checking argv
  if (argc <= 3) {
    // Gather arguments
    std::cerr << "usage: " << argv[0] << " <array size> <block size> <degree>" << std::endl;
    return -1;
    // std::cout << "Enter array size:\n"; std::cin >> a;
    // std::cout << "Enter block size:\n"; std::cin >> b;
    // std::cout << "Enter polynomial degree:\n"; std::cin >> d;
  } else {
    // std::cerr << "usage: " << argv[0] << " <array size> <block size> <degree>" << std::endl;
    // return -1;
    a = std::atoi(argv[1]); b = std::atoi(argv[2]); d = std::atoi(argv[3]);
  }

  // Calculating sizes
  int numBlocks = std::ceil(double(a) / double(b));
  int size = a, sizeCoeff = d + 1;
  // int numBlocks = (a > b) ? std::ceil(double(a) / double(b)) : 1;

  // Open output file
  std::ofstream file("cuda_.out");


  /// TODO: Polynomial expansion
  /// COMPLETE: Polynomial expansion
  float *x = new float[size];
  float *coeff = new float[sizeCoeff];
  float *d_x, *d_coeff;

  // Fill input array
  for (int i = 0; i < size; ++i) {x[i] = float(i + 1);}
  for (int i = 0; i < sizeCoeff; ++i) {coeff[i] = float(i + 1);}
  
  /// DEBUG: print before
  // std::cout << "x: ";
  // for (int i = 0; i < size; ++i) {std::cout << x[i] << " ";}
  // std::cout << std::endl;
  // std::cout << "coeff: ";
  // for (int i = 0; i < sizeCoeff; ++i) {std::cout << coeff[i] << " ";}
  // std::cout << std::endl;
  
  cudaMalloc((void **) &d_x, size * sizeof(float));
  cudaMalloc((void **) &d_coeff, sizeCoeff * sizeof(float));

  cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_coeff, coeff, sizeCoeff * sizeof(float), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  // Time start
  std::chrono::time_point<std::chrono::high_resolution_clock> time_start = std::chrono::high_resolution_clock::now();

  /// MEASUERMENTS:
  // PCI-express latency & bandwidth
  // cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
  // for (int i = 0; i < 100; ++i) {
  //   cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
  // }

  // GPU bandwidth & flops
  // expand_poly<<<numBlocks,b>>>(d_x, d_coeff, size, sizeCoeff);
  /// MEASUREMENTS:

  CUDA_err();

  // cudaMemcpy(x, d_x, size * sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(coeff, d_coeff, sizeCoeff * sizeof(float), cudaMemcpyDeviceToHost);

  // cudaFree(d_x);
  // cudaFree(d_coeff);

  cudaDeviceSynchronize();
  
  // Time end
  std::chrono::time_point<std::chrono::high_resolution_clock> time_end = std::chrono::high_resolution_clock::now();
  
  // Time elapsed
  std::chrono::duration<double> elapsed_seconds = time_end - time_start;

  /// DEBUG: print  after
  // std::cout << "x: ";
  // for (int i = 0; i < size; ++i) {std::cout << x[i] << " ";}
  // std::cout << std::endl;
  // std::cout << "coeff: ";
  // for (int i = 0; i < sizeCoeff; ++i) {std::cout << coeff[i] << " ";}
  // std::cout << std::endl;


  // Time duration & result/data output
  std::cout << argv[0] << " " << a << " " << b << " " << d << " " << std::endl;
  std::cout << "numBlocks: " << numBlocks << std::endl;
  std::cout << "Size: " << size << std::endl;
  std::cout << "Size coeff: " << sizeCoeff << std::endl;
  // std::cout << "x: ";
  // for (int i = 0; i < size; ++i) {std::cout << x[i] << " ";}
  // std::cout << std::endl;
  // std::cout << "coeff: ";
  // for (int i = 0; i < sizeCoeff; ++i) {std::cout << coeff[i] << " ";}
  // std::cout << std::endl;
  std::cout << "Time elapsed (s): " << elapsed_seconds.count() << std::endl;
  
  file << argv[0] << " " << a << " " << b << " " << d << " " << std::endl;
  file << "numBlocks: " << numBlocks << std::endl;
  file << "Size: " << size << std::endl;
  file << "Size coeff: " << sizeCoeff << std::endl;
  // file << "x: ";
  // for (int i = 0; i < size; ++i) {file << x[i] << " ";}
  // file << std::endl;
  // file << "coeff: ";
  // for (int i = 0; i < sizeCoeff; ++i) {file << coeff[i] << " ";}
  // file << std::endl;
  file << "Time elapsed (s): " << elapsed_seconds.count() << std::endl;
  
  file.close();

  /// MEASUREMENTS:
  long double flops = (size * sizeCoeff * sizeCoeff * 2);
  flops /= elapsed_seconds.count();
  flops /= 1000000000000;
  // std::cout << "latency (s): " << elapsed_seconds.count() / 100 << std::endl;
  // std::cout << "bandwidth (GB/s): " << ((size * sizeof(float)) / elapsed_seconds.count()) / 1000000000 << std::endl;
  std::cout << "flops (TFlops/s): " << flops << std::endl;
  
  // Memory deallocation
  delete[] x; delete[] coeff;
  
  return 0;
}
