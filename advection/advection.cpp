#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <chrono>
#include <cmath>

int main(int argc, char* argv[]) {
  // checking argc
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <N> <dt>" << std::endl;
    return -1;
  }

  // Time start
  std::chrono::time_point<std::chrono::system_clock> time_start = std::chrono::system_clock::now();

  /// ADVECTION
  // declaring and initializing variables, functions
  int N = atoi(argv[1]);
  float t_max = 2.0, x_min = 0.0, x_max = 1.0, v = 1.0, xc = 0.25, dt = atof(argv[2]);
  float dx = (x_max - x_min) / N, nbsteps = (int) (t_max / dt), alpha = v * (dt / (2 * dx));
  
  // declaring, initializing, and populating arrays
  float* x = new float[N + 2]; float* u_o = new float[N + 2];
  for (int i = 0; i < N + 2; ++i) x[i] = x_min + ((i - 1) * dx);
  for (int i = 0; i < N + 2; ++i) u_o[i] = exp(-200 * pow((x[i] - xc), 2));
  float* u = new float[N + 2]; float* u_new = new float[N + 2]; memcpy(u, u_o, sizeof(u_o)); memcpy(u_new, u_o, sizeof(u_o));
  
  // timestamp for loop
  for (int timestamp = 1; timestamp < nbsteps; ++timestamp) {
    // current timestamp
    double current_time = timestamp * dt;
    
    /// TODO:
    // output u per given timestamp
    // and check for correct output
    
    // Lax-Friedrichs scheme
    for (int i = 1; i < N + 1; ++i) {
      u_new[i] = u[i] - alpha * (u[i + 1] - u[i - 1]) + 0.5 * (u[i + 1] + 2 * (u[i]) + u[i - 1]);
    }

    // set u = u_new
    memcpy(u, u_new, sizeof(u_new));

    // enforcing periodic boundary conditions
    u[0] = u[N]; u[N + 1] = u[1];
  }

  // Time end
  std::chrono::time_point<std::chrono::system_clock> time_end = std::chrono::system_clock::now();

  // Time elapsed
  std::chrono::duration<double> elapsed_seconds = time_end-time_start;
  
  // Time duration & result print out
  std::cout << "+++\n";
  std::cout << "Inputs:\tN = " << N << ", dt = " << dt << std::endl;
  std::cout << "Time elapsed (s): " << elapsed_seconds.count() << std::endl;
  std::cout << "===\n\n";
  
  // memory deallocation
  delete[] x; delete[] u_o; delete[] u; delete[] u_new;
  
  return 0;
}
