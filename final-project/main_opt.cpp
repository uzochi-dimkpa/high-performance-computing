#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <string>


#include <thread>
#include <mutex>
#include <omp.h>
#include <immintrin.h>



// Mutex
std::mutex mtx;

// Thread batch
std::vector<std::thread> threadBatch;

// Total number of guesses made initialization
unsigned long long int totalNumGuesses = 0;

// Found match boolean
bool foundMatch = false;

// Number of threads
#define NUM_THREADS 16

// Stringify input
#define STRINGIFY(s) #s

// Bitwise operations
// Rotate hex values left by c bits
#define LEFTROTATE(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

// Functions ONE, TWO, THREE, FOUR
// 
/// TODO: Implement function macros
#define ONE(B, C, D) ((B & C) | ((~B) & D))
#define TWO(B, C, D) ((D & B) | ((~D) & C))
#define THREE(B, C, D) (B ^ C ^ D)
#define FOUR(B, C, D) (C ^ (B | (~D)))

// 1 byte; int
std::uint8_t shiftPerRound[64] = {  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
                                    5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
                                    4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
                                    6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21  };



// Fake-MD5 algorithm
// 
/// TODO: fix calculations; currently produces incorrect result; unsure why
// 
unsigned char* fakeMD5(std::string msg) {
  // Unsigned 32-bit integer values & arrays
  std::uint32_t a0 = 0x01234567, b0 = 0x89abcdef, c0 = 0xfedcba98, d0 = 0x76543210;
  std::uint32_t A, B, C, D;
  std::uint32_t sineConstArr[64];
  std::uint32_t msgUint32Arr[16];

  // Unsigned char digest output array
  unsigned char* digest = new unsigned char[16];

  // Populating sine constants
  for (int i = 0; i < 64; ++i) sineConstArr[i] = (std::uint32_t) std::floor((std::pow(2, 32)) * std::abs(std::sin(i + 1)));


  // Calculating length of message in bits
  unsigned long long int paddedLength = 0, paddedLength2 = 0, paddedLength3 = 0;
  while (paddedLength <= msg.length() * 8) paddedLength += 512; paddedLength -= 64;


  // Populating unsigned 8-bit integer vector with message characters
  // with padding to fill size to 512 bits (64 bytes)
  std::vector<uint8_t> msgUint8Vec(msg.begin(), msg.end());
  msgUint8Vec.push_back((std::uint8_t) 128);
  while (msgUint8Vec.size() != paddedLength / 8) msgUint8Vec.push_back((std::uint8_t) 0);
  std::uint64_t msgLength = msg.length() * 8;
  for (int i = 56; i >= 0; i -= 8) msgUint8Vec.push_back(((msgLength >> i) & 0x00000000000000FF));
  


  /// DEBUG:
  // Visualizing outputs of data
  /*
  std::cout << "\n~~~\npaddedLength:\n";
  std::cout << paddedLength;
  std::cout << "\n~~~\nmsgLength:\n";
  std::cout << msg.length() * 8;
  // ~~
  A = a0, B = b0, C = c0, D = d0;
  std::cout << "\n+++===+++\n";
  std::cout << "A, B, C, D:" << std::endl;
  std::cout << std::hex << A << ", " << B << ", " << C << ", " << D << std::endl;
  std::cout << "a0, b0, c0, d0:" << std::endl;
  std::cout << std::hex << a0 << ", " << b0 << ", " << c0 << ", " << d0 << std::endl;
  std::cout << "(B & C):" << std::endl;
  std::cout << std::hex << (B & C) << std::endl;
  std::cout << "(std::uint32_t) (~B)" << std::endl;
  std::cout << std::hex << (std::uint32_t) (~B) << std::endl;
  std::cout << "(std::uint32_t) (~B) & D):" << std::endl;
  std::cout << std::hex << ((std::uint32_t) (~B) & D) << std::endl;
  std::cout << "OPER. 1:" << std::endl;
  std::cout << "( (B & C) | ( std::uint32_t) (~B) & D) ):" << std::endl;
  std::cout << std::hex << ((B & C) | ((std::uint32_t) (~B) & D)) << std::endl;
  A = 0x799d1352, B = 0x2c34dfa2, C = 0xde1673be, D = 0x4b976282;
  std::cout << "(D & B):" << std::endl;
  std::cout << std::hex << (D & B) << std::endl;
  std::cout << "(~D):" << std::endl;
  std::cout << std::hex << (~D) << std::endl;
  std::cout << "((std::uint32_t) (~D) & C):" << std::endl;
  std::cout << std::hex << ((std::uint32_t) (~D) & C) << std::endl;
  std::cout << "OPER. 2:" << std::endl;
  std::cout << "( (D & B) | ((std::uint32_t) (~D) & C) ):" << std::endl;
  std::cout << std::hex << ((D & B) | ((std::uint32_t) (~D) & C)) << std::endl;
  A = 0xeb160cd0, B = 0xd5071367, C = 0xc058ade2, D = 0x63c603d7;
  std::cout << "OPER. 3:" << std::endl;
  std::cout << "( B ^ C ^ D ):" << std::endl;
  std::cout << std::hex << (B ^ C ^ D) << std::endl;
  A = 0x60cdceb1, B = 0x7d502063, C = 0x8b3d715d, D = 0x1de3a739;
  std::cout << "B | (std::uint32_t) (~D))):" << std::endl;
  std::cout << std::hex << (B | (std::uint32_t) (~D)) << std::endl;
  std::cout << "OPER. 4:" << std::endl;
  std::cout << "( C ^ (B | (std::uint32_t) (~D)) ):" << std::endl;
  std::cout << std::hex << (C ^ (B | (std::uint32_t) (~D))) << std::endl;
  A = 0x60cdceb1, B = 0x7d502063, C = 0x8b3d715d, D = 0x1de3a739;
  std::cout << "( A + a0 ), ( B + b0 ), ( C + c0 ), ( D + d0 ):" << std::endl;
  std::cout << std::hex << (A + a0) << ", " << (B + b0) << ", " << (C + c0) << ", " << (D + d0) << std::endl;
  std::uint32_t F = 0x2bd309f0;
  std::cout << "F: " << std::hex << F << "\n" << STRINGIFY(LEFTROTATE(F, 7)) << ": ";
  std::cout << std::hex << LEFTROTATE(F, 7) << std::endl;
  std::cout << "+++===+++";
  // ~~
  std::cout << "\n~~~\nsineConstant:\n|";
  for (std::uint32_t sineConst : sineConstArr) {
    std::cout << std::hex << sineConst << "|";
  }
  // ~~
  std::cout << "\n~~~\nmsgUint8Vec:\n|";
  for (std::uint8_t i : msgUint8Vec) {
    std::cout << std::hex << (i & 0b11111111) << "|";
  }
  // ~~
  std::cout << "\n~~~\nmsgUint32Arr:\n|";
  for (std::uint32_t i : msgUint32Arr) {
    std::cout << std::hex << (i & 0xFFFFFFFF) << "|";
  }
  // ~~
  std::cout << "\n~~~\nmsgCharVector:\n|";
  for (std::uint8_t i : msgCharVector) {
    std::cout << i << "|";
  }
  // ~~
  std::cout << "\n~~~\n";
  /**/
  /// DEBUG:



  /// DIGEST:
  // Digest message for every 512-bit (64-byte) chunk of the input message
  unsigned long long int loopCounter = 0;
  while (!msgUint8Vec.empty()) {
    // Populating unsigned 32-bit integer message arrays with concatenated 8-bit
    // entries from unsigned 8-bit integer vectors, then clearing respective
    // entries from unsigned 8-bit integer vectors
    for (int i = 0; i < 16; ++i) {
      msgUint32Arr[i] = (msgUint8Vec[(i * 4)] << 24) + (msgUint8Vec[(i * 4) + 1] << 16) + (msgUint8Vec[(i * 4) + 2] << 8) + msgUint8Vec[(i * 4) + 3];
    }
    msgUint8Vec.erase( msgUint8Vec.begin(), msgUint8Vec.size() > 64 ?  msgUint8Vec.begin() + 64 : msgUint8Vec.end() );
    
    A = a0, B = b0, C = c0, D = d0;

    /// DEBUG:
    /*
    std::cout << "\n";
    std::cout << "LOOP " << loopCounter + 1 << std::endl;
    std::cout << "msgUint32Arr:" << std::endl;
    for (std::uint32_t m : msgUint32Arr) std::cout << std::hex << m << " "; std::cout << std::endl;
    std::cout << "a0, b0, c0, d0:" << std::endl;
    std::cout << std::hex << a0 << ", " << b0 << ", " << c0 << ", " << d0 << std::endl;
    /**/
    /// DEBUG:

    for (std::uint32_t i = 0; i < 64; ++i) {
      std::uint32_t F, g = 0;

      if (i >= 0 && i <= 15) {
        // F = ((B & C) | ((std::uint32_t) (~B) & D));
        F = ONE(B, C, D);
        // g = i;
        ++g;
      } else if (i >= 16 && i <= 31) {
        // F = ((D & B) | ((std::uint32_t) (~D) & C));
        F = TWO(B, C, D);
        g = ((5 * i) + 1) % 16;
      } else if (i >= 32 && i <= 47) {
        // F = (B ^ C ^ D);
        F = THREE(B, C, D);
        g = ((3 * i) + 5) % 16;
      } else {
        // F = (C ^ (B | (std::uint32_t) (~D)));
        F = FOUR(B, C, D);
        g = (7 * i) % 16;
      }

      // F = (F + A + sineConstArr[i] + msgUint32Arr[g]);
      F += (A + sineConstArr[i] + msgUint32Arr[g]);
      A = D;
      D = C;
      C = B;
      B = B + LEFTROTATE(F, shiftPerRound[i]);

      /// DEBUG:
      /*
      std::cout << "\n";
      std::cout << "ITERATION " << i + 1 << std::endl;
      if (i >= 0 && i <= 15) {
        std::cout << "F = ((B & C) | ((~B) & D)):" << std::endl;
      } else if (i >= 16 && i <= 31) {
        std::cout << "F = ((D & B) | ((~D) & C)):" << std::endl;
      } else if (i >= 32 && i <= 47) {
        std::cout << "F = (B ^ C ^ D):" << std::endl;
      } else {
        std::cout << "F = (C ^ (B | (~D))):" << std::endl;
      }
      std::cout << "F: " << std::hex << F << std::endl;
      std::cout << "A, B, C, D:" << std::endl;
      std::cout << std::hex << A << ", " << B << ", " << C << ", " << D << std::endl;
      std::cout << "+++===+++\n";
      /**/
      /// DEBUG:
    }

    a0 += A;
    b0 += B;
    c0 += C;
    d0 += D;

    ++loopCounter;
  }
  /// DIGEST:


  // Append hex values into final digest
  for (int i = 0; i <= 3; ++i) {
    digest[i] = ((a0 >> (24 - (8 * i))) & 0x000000FF);
    digest[4 + i] = ((b0 >> (24 - (8 * i))) & 0x000000FF);
    digest[8 + i] = ((c0 >> (24 - (8 * i))) & 0x000000FF);
    digest[12 + i] = ((d0 >> (24 - (8 * i))) & 0x000000FF);
  }

  // Return digest
  return digest;

  // Memory deallocation
  msg.clear(); delete[] digest; delete[] sineConstArr; delete[] msgUint32Arr;
}


// Print message
void printMessage(std::string msg, std::string msgType) {
  std::cout << "\n" << msgType << " message:\n" << msg << std::endl;
}


// Print digest
void printDigest(unsigned char* digest, std::string digestType) {
  std::cout << "\n" << digestType << " digest:\n";
  for (int i = 0; i < 16; ++i) std::cout << std::hex << (unsigned int) digest[i] << " "; std::cout << std::endl;
}


// Compare digests
bool compare(unsigned char* messageDigest, unsigned char* guessDigest) {
  for (int i = 0; i < 16; ++i) {
    if (messageDigest[i] != guessDigest[i]) {
      return false;
    }
  }
  
  foundMatch = true;
  return true;
}


// Hash input message
unsigned char* hashMessage(std::string fileName, std::string msgType) {
  // Open message file
  std::ifstream message (fileName);

  // Grabbing message text as string
  std::string str, msg;
  while (std::getline(message, str)) {
    msg.append(str);
  } str.clear();

  // Gather message digest
  unsigned char* messageDigest = fakeMD5(msg);
  
  /// OUTPUT:
  // Printing input message and output digest
  // printMessage(msg, msgType);
  // printDigest(fakeMD5(msg), msgType);
  
  return messageDigest;

  // Memory deallocation
  delete[] messageDigest;
}


// Create random guess string
std::string createGuess(unsigned long long int msgLength) {
  // Declare guess message string
  std::string guessMsg;
  
  // Declare & initialize random device to generate
  // random integers between the values of 32 and 126
  std::random_device rd;
  std::default_random_engine rng(rd());
  // std::mt19937 rng(rd());
  std::uniform_int_distribution<unsigned int> genRandInt(32, 126);

  // Generate a random integer and assign it to a variable
  unsigned int randNum = genRandInt(rng);

  // Initialize guess string
  for (int i = 0; i < msgLength; ++i) {
    // std::string tempStr(1, (unsigned char) genRandInt(rng));
    guessMsg.append(std::string(1, (unsigned char) genRandInt(rng)));
  }

  return guessMsg;
}


// Make guesses until input message is detected
int guess(std::string fileName) {
  // Open input message file
  std::ifstream message (fileName);

  // Grabbing input message text as string
  std::string str, msg;
  while (std::getline(message, str)) {
    msg.append(str);
  } str.clear();

  // Get input message length and clear input message
  unsigned long long int msgLength = msg.length(); msg.clear();
  
  // Gather input message digest
  unsigned char* messageDigest = hashMessage(fileName, "input");

  // Declare guess message string and digest array
  std::string guessMsg; unsigned char* guessDigest;

  // Declare guess-making parameters
  unsigned long long int numGuesses = 0, guessLength = 0;
  unsigned long long int numPossibilities = 10 * std::pow(96, msgLength);

  // Make & print guesses
  // std::cout << std::endl;
  do {
    // Initialize guess message string;
    guessMsg = createGuess(guessLength);
    if ((numGuesses % numPossibilities) == 0) ++guessLength;

    // Increment number of guesses
    ++numGuesses;

    // Create guess digest
    guessDigest = fakeMD5(guessMsg);
  } while (!compare(messageDigest, guessDigest));

  // Sequential calculation of total number of guesses
  if (foundMatch) {
    mtx.lock(); 
    totalNumGuesses += numGuesses;
    mtx.unlock();
  }

  // Output printing
  // if (compare(messageDigest, guessDigest)) std::cout << "\n" << "Found a working input message!\
                                            \nSecond Pre-Image Resistance has been broken!\
                                            \nHere is an input that produces a matching hash:\
                                            \n!!!   [--->   " << guessMsg << "   <---]   !!!\
                                            \nalong with its matching digest:\
                                            \n+=+   [";
                                            // for (int i = 0; i < 15; ++i) std::cout << std::hex << (unsigned int) guessDigest[i] << " ";
                                            // std::cout << std::hex << (unsigned int) guessDigest[15] << "]   +=+\n";
                                            // std::cout << std::dec << "It took " << numGuesses << " guess(es)" << std::endl;
  
  // Clear guess and free allocated memory
  guessMsg.clear(); delete[] messageDigest; delete[] guessDigest;

  return numGuesses;
}


int main(int argc, char* argv[]) {
  // Checking argv
  if (argc != 2) {
    // Gather arguments
    std::cerr << "usage: " << argv[0] << " <input file name>" << std::endl;
    return -1;
  }

  // mGuesses
  unsigned long long int nGuesses = 0;

  // Threads
  pthread_t threads[NUM_THREADS];

  // Get file name from command line argument
  std::string fileName(argv[1]);

  // Time start
  std::chrono::time_point<std::chrono::high_resolution_clock> time_start = std::chrono::high_resolution_clock::now();
  
  // unsigned long long int nGuesses = guess(fileName + ".txt");

  for (int i = 0; i < NUM_THREADS; i++) {
    std::thread th (
      [fileName, i] () {
        guess(fileName + ".txt");
      }
    );

    threadBatch.push_back(std::move(th));
  }

  // Join threads
  for (std::thread &th : threadBatch) th.join();

  // Time end
  std::chrono::time_point<std::chrono::high_resolution_clock> time_end = std::chrono::high_resolution_clock::now();
  
  // Time elapsed
  std::chrono::duration<double> elapsed_seconds = time_end - time_start;

  // Time duration & results output
  std::cout << std::dec << "It took " << totalNumGuesses << " guess(es)" << std::endl;
  std::cout << "\n\nTime elapsed (s): " << elapsed_seconds.count() << std::endl;
  // std::cout << "# Guesses / sec: " << nGuesses / elapsed_seconds.count() << std::endl;
  std::cout << "# Guesses / sec: " << totalNumGuesses / elapsed_seconds.count() << std::endl;

  return 0;
}