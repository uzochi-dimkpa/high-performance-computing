#include <iostream>
#include <vector>
#include <chrono>
// #include <cstdlib>
// #include <pthread.h>
#include <thread>
#include <mutex>


#define NUM_THREADS 16
bool foundMatch = false;
unsigned long long int guess = 2000000000;
unsigned long long int numGuesses = 0;
std::mutex mtx;

void makeGuess(int i) {
  int threadNumGuesses = 0;
  while (!foundMatch) {
    threadNumGuesses++;
    if (threadNumGuesses == guess) {
      foundMatch = true;
      std::cout << "Guess found on thread " << std::dec << (long) i << "! Ending all threads..." << std::endl;
    }
  }
  
  std::cout << "Thread " << i << " made " << threadNumGuesses << " guesses!" << std::endl;

  mtx.lock();
  numGuesses += threadNumGuesses;
  mtx.unlock();
}

int main () {
  pthread_t threads[NUM_THREADS];
  std::vector<std::thread> allThreads;
  int rc;
  int i;

  for ( i=0; i < NUM_THREADS; i++ ) {
    std::thread th (
      [i]() {
        makeGuess(i);
      }
    );

    allThreads.push_back(std::move(th));
  }

  for (std::thread &t : allThreads) {
    t.join();
  }

  std::cout << "Total # of guesses: " << numGuesses << std::endl;

  return 0;
}