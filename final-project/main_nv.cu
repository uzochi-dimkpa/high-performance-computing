#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <chrono>
#include <string>


#include <cuda/std/cstdint>
#include <curand.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <stdint.h>



// Print CUDA error
void CUDA_err() {
  std::cout << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
};



// CUDA variables
unsigned char* d_inputMsgChar;
unsigned char* d_msgDigest;
std::uint8_t* d_shiftPerRound;
std::uint32_t* d_sineConstArr;
curandState* d_randState;
std::uint64_t* d_numGuessesArr;
__device__ bool guessFound = false;




// Stringify input
#define STRINGIFY(s) #s

// Bitwise operations
// Rotate hex values left by c bits
#define LEFTROTATE(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

// Functions F, G, H, I
#define ONE(B, C, D) ((B & C) | ((~B) & D))
#define TWO(B, C, D) ((D & B) | ((~D) & C))
#define THREE(B, C, D) (B ^ C ^ D)
#define FOUR(B, C, D) (C ^ (B | (~D)))

// CUDA kernel dimensions
#define NBLOCKS 16
// #define NBLOCKS 128
#define NTHREADS 256

// 1 byte; unsigned char
// __device__ std::uint8_t d_shiftPerRound[64];

// 4 bytes; unsigned int
// __device__ std::uint8_t d_sineConstArr[64];

// 4 bytes; unsigned int
// extern __global__ unsigned char d_msgDigest[16];




// Fake-MD5 algorithm
// 
/// TODO: fix calculations; currently produces incorrect result; unsure why
// 
__device__ unsigned char* fakeMD5(unsigned char* msg, std::uint64_t msgLength, std::uint64_t paddedLength, std::uint8_t* d_shiftPerRound, std::uint32_t* d_sineConstArr) {
  // Unsigned 32-bit integer values & arrays
  std::uint32_t a0 = 0x01234567, b0 = 0x89abcdef, c0 = 0xfedcba98, d0 = 0x76543210;
  std::uint32_t A, B, C, D;
  unsigned char* local_msgUint8Arr = (unsigned char *) malloc((paddedLength / 8) * sizeof(unsigned char));
  unsigned char* outputDigest = (unsigned char *) malloc(16 * sizeof(unsigned char));
  
  paddedLength -= 64;

  // Populating unsigned 8-bit integer vector with message characters
  for (int i = 0; i < msgLength; ++i) {
    local_msgUint8Arr[i] = msg[i];
  }
  local_msgUint8Arr[msgLength] = (unsigned char) 0x80;

  for (int i = msgLength + 1; i < (paddedLength / 8); ++i) {
    local_msgUint8Arr[i] = (unsigned char) 0;
  }
  std::uint32_t* msgUint32Arr = (std::uint32_t *) local_msgUint8Arr;

  std::uint64_t msgUint32ArrLength = ((paddedLength + 64) / 8) / 4;

  std::uint64_t numIterations = msgUint32ArrLength / 16;

  /// DIGEST:
  for (std::uint64_t j = 0; j < numIterations; ++j) {
    
    A = a0, B = b0, C = c0, D = d0;

    for (std::uint32_t i = 0; i < 64; ++i) {
      std::uint32_t F, g;

      if (i >= 0 && i <= 15) {
        F = ONE(B, C, D);
        g = i;
      } else if (i >= 16 && i <= 31) {
        F = TWO(B, C, D);
        g = ((5 * i) + 1) % 16;
      } else if (i >= 32 && i <= 47) {
        F = THREE(B, C, D);
        g = ((3 * i) + 5) % 16;
      } else {
        F = FOUR(B, C, D);
        g = (7 * i) % 16;
      }

      F = (F + A + d_sineConstArr[i] + msgUint32Arr[((j * 16) + g)]);
      A = D;
      D = C;
      C = B;
      B = B + LEFTROTATE(F, d_shiftPerRound[i]);
    }

    a0 = a0 + A;
    b0 = b0 + B;
    c0 = c0 + C;
    d0 = d0 + D;
  }
  /// DIGEST:

  // Append hex values into final digest
  for (int i = 0; i <= 3; ++i) {
    outputDigest[i] = ((a0 >> (24 - (8 * i))) & 0x000000FF);
    outputDigest[4 + i] = ((b0 >> (24 - (8 * i))) & 0x000000FF);
    outputDigest[8 + i] = ((c0 >> (24 - (8 * i))) & 0x000000FF);
    outputDigest[12 + i] = ((d0 >> (24 - (8 * i))) & 0x000000FF);
  }

  delete[] local_msgUint8Arr;

  return outputDigest;

  delete[] outputDigest;
  delete[] msgUint32Arr;
}


__device__ float genRand(uint64_t seed, int tid, curandState* d_randState) {
    return curand_uniform(&d_randState[tid]);
}


// Create random guess string
__device__ unsigned char* createGuess(unsigned long long int msgLength, curandState* d_randState) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Declare guess message char array (same length as input message)
  unsigned char* guessMsg = (unsigned char*) malloc(msgLength * sizeof(unsigned char));

  // Declaring random guess float
  float randFloat;

  // Initialize guess string
  for (int i = 0; i < msgLength; ++i) {

    // Generating random guess float
    randFloat = genRand(msgLength, tid, d_randState);
    // printf("~\nrandFloat: %f~", randFloat);

    // Converting random guess float to integer between 32 & 126
    int rnd_char = (int) (randFloat * (127 - 32)) + 32;
    rnd_char = (rnd_char != 127) ? rnd_char : 126;

    // printf("~\nrnd_char int: {%d}~", rnd_char );
    // printf("~\nrnd_char char: [%c]~", (unsigned char) rnd_char );
    guessMsg[i] =(unsigned char) rnd_char;
  }

  return guessMsg;

  delete[] guessMsg;
}



// Compare digests
__device__ int compare(unsigned char* messageDigest, unsigned char* guessDigest) {
  for (int i = 0; i < 16; ++i) {
    // printf("%u vs. %u\n", (std::uint8_t) messageDigest[i], (std::uint8_t) guessDigest[i]);
    if (messageDigest[i] != guessDigest[i]) {
      // printf("\ncompare false\n");
      return 0;
    }
  }
  // printf("\ncompare false\n");
  return 1;
}



__global__ void init_rand(curandState *d_randState) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  curand_init(tid * 1337, tid, 0, &d_randState[tid]);
}



// Make guesses until input message is detected
__device__ std::uint64_t guess(curandState* d_randState, unsigned char* d_inputMsgChar, std::uint64_t msgLength, std::uint64_t paddedLength, std::uint8_t* d_shiftPerRound, std::uint32_t* d_sineConstArr, unsigned char* d_msgDigest, unsigned char* inputMsg) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  std::uint64_t numGuesses = 0;

  int compareDigests = 0;

  unsigned char* guessMsg = (unsigned char *) malloc(msgLength * sizeof(unsigned char));
  // guessMsg = createGuess(msgLength, d_randState);

  unsigned char* guessDigest = (unsigned char *) malloc(16 * sizeof(unsigned char));
  // guessDigest = fakeMD5(guessMsg, msgLength, paddedLength, d_shiftPerRound, d_sineConstArr);

  /// DEBUG:
  // printf("~=~\ntid: %d\n", tid);
  // printf("~+~\nguessDigest:  ");
  // // if (threadIdx.x == 0) {
  //   for (int i = 0; i < 16; ++i) {
  //     printf("%u  ", (std::uint8_t) guessDigest[i]);
  //   }
  // // }
  // printf("~\n");
  

  do {
    // unsigned char* guessMsg = (unsigned char *) malloc(msgLength * sizeof(unsigned char));
    guessMsg = createGuess(msgLength, d_randState);

    // unsigned char* guessDigest = (unsigned char *) malloc(16 * sizeof(unsigned char));
    guessDigest = fakeMD5(guessMsg, msgLength, paddedLength, d_shiftPerRound, d_sineConstArr);

    /// DEBUG:
    // printf("~=~\ntid: %d\n", tid);
    // printf("~+~\nguessMsg:  ");
    // // if (threadIdx.x == 0) {
    //   for (int i = 0; i < msgLength; ++i) {
    //     printf("%u  ", (std::uint8_t) guessMsg[i]);
    //   }
    // // }
    // printf("~\n");
    // printf("~=~\ntid: %d\n", tid);
    // printf("~+~\nguessDigest:  ");
    // // if (threadIdx.x == 0) {
    //   for (int i = 0; i < 16; ++i) {
    //     printf("%u  ", (std::uint8_t) guessDigest[i]);
    //   }
    // // }
    // printf("~\n");

    compareDigests = compare(d_msgDigest, guessDigest);
    
    /// DEBUG:
    // printf("compareDigests: %s", (compareDigests == 1) ? "true\n" : "false\n");
    // printf("compareDigests: %d\n", compareDigests);

    ++numGuesses;

    if (compareDigests == 1) {
      printf("\nguessFound on thread %u!\n", tid);
      guessFound = true;

      // printf("\n__syncthreads()!\n");
      // __syncthreads();
      // printf("\nbreak!\n");
      // break;
    }

    delete[] guessDigest;
    delete[] guessMsg;
    
  } while (compareDigests != 1 && !guessFound);

  __syncthreads();
  if (tid == 0) printf("\nnumGuesses : %u\n", numGuesses);
  __syncthreads();
  
  return numGuesses;
}



__global__ void run(std::uint64_t* d_numGuessesArr, unsigned char* d_inputMsgChar, unsigned char* d_msgDigest, std::uint64_t msgLength, std::uint64_t paddedLength, std::uint8_t* d_shiftPerRound, std::uint32_t* d_sineConstArr, curandState* d_randState) {
  // printf("\n~~~ run! ~~~\n");
  std::uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ unsigned char inputMsg[];
  extern __shared__ unsigned char shared_msgDigest[16];
  extern __shared__ std::uint8_t shared_shiftPerRound[64];
  extern __shared__ std::uint32_t shared_sineConstArr[64];
  // extern __shared__ curandState shared_randState[NTHREADS*NBLOCKS];
  
  if (threadIdx.x < msgLength) {
    for (int i = threadIdx.x; i < msgLength; i += blockDim.x) {
      inputMsg[i] = d_inputMsgChar[i];
    }
  }
  
  __syncthreads();

  if (tid == 0) {
    d_msgDigest = fakeMD5(inputMsg, msgLength, paddedLength, d_shiftPerRound, d_sineConstArr);
    for (int i = 0; i < 16; ++i) shared_msgDigest[i] = d_msgDigest[i];
  }

  // if (tid == 1) {
  // }
  
  if (tid == 2) {
    for (int i = 0; i < 64; ++i) shared_shiftPerRound[i] = d_shiftPerRound[i];
  }

  if (tid == 3) {
    for (int i = 0; i < 64; ++i) shared_sineConstArr[i] = d_sineConstArr[i];
  }

  // if (tid == 4) {
  //   for (int i = 0; i < NBLOCKS*NTHREADS; ++i) shared_randState[i] = d_randState[i];
  // }

  __syncthreads();

  // make guesses
  // std::uint64_t nGuesses = guess(d_randState, inputMsg, msgLength, paddedLength, d_shiftPerRound, d_sineConstArr, shared_msgDigest, inputMsg);
  std::uint64_t nGuesses = guess(d_randState, inputMsg, msgLength, paddedLength, shared_shiftPerRound, shared_sineConstArr, shared_msgDigest, inputMsg);
  
  // printf("\n~=~tid: %u~\n", tid);
  // printf("\n~+~nGuesses: %u~\n", nGuesses);

  if (tid < NBLOCKS * NTHREADS) d_numGuessesArr[tid] = nGuesses;

  // printf("\n~~~ end run! ~~~\n");
}



int main(int argc, char* argv[]) {
  // Checking argv
  if (argc != 2) {
    // Gather arguments
    std::cerr << "usage: " << argv[0] << " <input file name>" << std::endl;
    return -1;
  }

  // Get file name from command line argument
  std::string fileName(argv[1]);
  std::ifstream msgFile(fileName + ".txt");

  // Grabbing message length
  std::string str, msg;
  while (std::getline(msgFile, str)) {
    msg.append(str);
  } str.clear();
  
  std::uint64_t msgLength = msg.length();
  std::uint64_t paddedLength = 0;
  while (paddedLength <= msgLength * 8) paddedLength += 512;

  std::cout << "msgLength: " << msg.length() << ", paddedLength: " << paddedLength << std::endl;

  unsigned char inputMsgChar[msgLength];
  for (int i = 0; i < msgLength; ++i) {
    inputMsgChar[i] = (unsigned char) msg[i];
  }

  std::uint8_t shiftPerRound[64] = {  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,\
                                      5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,\
                                      4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,\
                                      6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21   };
  
  std::uint32_t sineConstArr[64] = {  0xd76aa478,  0xe8c7b756,  0x242070db,  0xc1bdceee,\
                                      0xf57c0faf,  0x4787c62a,  0xa8304613,  0xfd469501,\
                                      0x698098d8,  0x8b44f7af,  0xffff5bb1,  0x895cd7be,\
                                      0x6b901122,  0xfd987193,  0xa679438e,  0x49b40821,\
                                      0xf61e2562,  0xc040b340,  0x265e5a51,  0xe9b6c7aa,\
                                      0xd62f105d,  0x02441453,  0xd8a1e681,  0xe7d3fbc8,\
                                      0x21e1cde6,  0xc33707d6,  0xf4d50d87,  0x455a14ed,\
                                      0xa9e3e905,  0xfcefa3f8,  0x676f02d9,  0x8d2a4c8a,\
                                      0xfffa3942,  0x8771f681,  0x6d9d6122,  0xfde5380c,\
                                      0xa4beea44,  0x4bdecfa9,  0xf6bb4b60,  0xbebfbc70,\
                                      0x289b7ec6,  0xeaa127fa,  0xd4ef3085,  0x04881d05,\
                                      0xd9d4d039,  0xe6db99e5,  0x1fa27cf8,  0xc4ac5665,\
                                      0xf4292244,  0x432aff97,  0xab9423a7,  0xfc93a039,\
                                      0x655b59c3,  0x8f0ccc92,  0xffeff47d,  0x85845dd1,\
                                      0x6fa87e4f,  0xfe2ce6e0,  0xa3014314,  0x4e0811a1,\
                                      0xf7537e82,  0xbd3af235,  0x2ad7d2bb,  0xeb86d391   };

  std::uint64_t* numGuessesArr = (std::uint64_t *) malloc(NBLOCKS * NTHREADS * sizeof(std::uint64_t));
  for (int i = 0; i < (4 * 64); ++i) {
    numGuessesArr[i] = 0;
  }
  std::uint64_t totalNumGuesses = 0;

  // Synchronize
  cudaDeviceSynchronize();
  
  cudaMalloc((void **) &d_inputMsgChar, msgLength * sizeof(unsigned char));
  cudaMalloc((void **) &d_msgDigest, 16 * sizeof(unsigned char));
  cudaMalloc((void **) &d_shiftPerRound, 64 * sizeof(std::uint8_t));
  cudaMalloc((void **) &d_sineConstArr, 64 * sizeof(std::uint32_t));

  cudaDeviceSynchronize();

  CUDA_err();

  cudaMalloc((void **) &d_randState, NBLOCKS * NTHREADS * sizeof(curandState));
  cudaMalloc((void **) &d_numGuessesArr, NBLOCKS * NTHREADS * sizeof(std::uint64_t));

  cudaDeviceSynchronize();
  
  CUDA_err();
  
  cudaMemcpy(d_inputMsgChar, inputMsgChar, msgLength * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_shiftPerRound, shiftPerRound, 64 * sizeof(std::uint8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sineConstArr, sineConstArr, 64 * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_numGuessesArr, d_numGuessesArr, NBLOCKS * NTHREADS * sizeof(std::uint64_t), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  
  CUDA_err();

  // Time start
  std::chrono::time_point<std::chrono::high_resolution_clock> time_start = std::chrono::high_resolution_clock::now();
  
  init_rand<<<NBLOCKS, NTHREADS>>>(d_randState);

  cudaDeviceSynchronize();

  CUDA_err();
  
  // run<<<1,1, (paddedLength / 8) * sizeof(unsigned char)>>>(d_numGuessesArr, d_inputMsgChar, d_msgDigest, msgLength, paddedLength, d_shiftPerRound, d_sineConstArr, d_randState);
  run<<<NBLOCKS, NTHREADS, (paddedLength / 8) * sizeof(unsigned char)>>>(d_numGuessesArr, d_inputMsgChar, d_msgDigest, msgLength, paddedLength, d_shiftPerRound, d_sineConstArr, d_randState);

  cudaDeviceSynchronize();

  CUDA_err();

  // Time end
  std::chrono::time_point<std::chrono::high_resolution_clock> time_end = std::chrono::high_resolution_clock::now();
  
  cudaMemcpy(numGuessesArr, d_numGuessesArr, NBLOCKS * NTHREADS * sizeof(std::uint64_t), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  CUDA_err();

  cudaFree(d_inputMsgChar);
  cudaFree(d_msgDigest);
  cudaFree(d_shiftPerRound);
  cudaFree(d_sineConstArr);
  cudaFree(d_randState);
  cudaFree(d_numGuessesArr);

  CUDA_err();

  cudaDeviceSynchronize();
  
  // Time elapsed
  std::chrono::duration<double> elapsed_seconds = time_end - time_start;

  // Counting up all guesses
  for (std::uint64_t i = 0; i < NTHREADS * NBLOCKS; ++i) std::cout << "Thread " << i << " made " << numGuessesArr[i] << "guesses!" << std::endl;
  for (std::uint64_t i = 0; i < NTHREADS * NBLOCKS; ++i) totalNumGuesses += numGuessesArr[i];

  // Time duration output
  std::cout << "\n\nTime elapsed (s): " << elapsed_seconds.count() << std::endl;
  std::cout << "# Guesses / sec: " << totalNumGuesses / elapsed_seconds.count() << std::endl;

  // Memory deallocation
  msg.clear();
  delete[] numGuessesArr;
  // delete[] shiftPerRound; delete[] sineConstArr;

  return 0;
}