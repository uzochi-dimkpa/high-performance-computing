#include <iostream>
#include <stdio.h>

//note that printf in kernels is highly unreliable. only for basic debug

__global__ void myKernel() {
  printf ("threadIdx={%d, %d, %d} blockIdx={%d, %d, %d} blockDim={%d, %d, %d} gridDim={%d, %d, %d}\n",
	  threadIdx.x, threadIdx.y, threadIdx.z,
	  blockIdx.x, blockIdx.y, blockIdx.z,
	  blockDim.x, blockDim.y, blockDim.z,
	  gridDim.x, gridDim.y, gridDim.z);
}

int main () {

  std::cout<<"========================"<<std::endl;
  std::cout<<"1,1"<<std::endl;
  std::cout<<"========================"<<std::endl;
  
  myKernel <<<1,1>>>();
  
  cudaDeviceSynchronize();

  std::cout<<std::endl;
  std::cout<<"========================"<<std::endl;
  std::cout<<"2,3"<<std::endl;
  std::cout<<"========================"<<std::endl;
  
  myKernel <<<2,3>>>();
  
  cudaDeviceSynchronize();

  std::cout<<std::endl;
  std::cout<<"========================"<<std::endl;
  std::cout<<"special"<<std::endl;
  std::cout<<"========================"<<std::endl;

  dim3 gridsize (1,2);
  dim3 blocksize (1,2,3);
  
  myKernel <<<gridsize,blocksize>>>();
  
  cudaDeviceSynchronize();
  return 0;
}