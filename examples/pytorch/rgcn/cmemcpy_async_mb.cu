#include <iostream>
#include <cstdint>
#include <cuda_runtime_api.h>

#define KB 1024
#define GB 1073741824

int main(int argc, char **argv) {

  float *dA, *dB;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int64_t i = 1; i < 10000; i += 1000) {
    int64_t datasize = i * KB;
    cudaMalloc(&dA, datasize);
    cudaMalloc(&dB, datasize);

    // warmup memcpy
    cudaMemcpyAsync(dB, dA, datasize, cudaMemcpyDeviceToDevice);

    // timing memcpy
    cudaEventRecord(start);
    cudaMemcpyAsync(dB, dA, datasize, cudaMemcpyDeviceToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float copytime = 0.0f;
    cudaEventElapsedTime(&copytime, start, stop);
    copytime = copytime / 1000; // seconds

    std::cout << "datasize: " << datasize << " copy_time: " << copytime << " bandwidth GB/s: " << (datasize / copytime / GB) << "\n";

    cudaFree(dA);
    cudaFree(dB);
  }
}
