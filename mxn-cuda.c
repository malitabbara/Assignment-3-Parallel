// Basic matrix multiplication
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// CUDA kernel function
__global__ void matrix_multiplication(float *A, float *B, float *C, int M, int N, int K) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < M && j < N) {
    C[i * N + j] = 0.0f;
    for (int k = 0; k < K; k++) {
      C[i * N + j] += A[i * K + k] * B[k * N + j];
    }
  }
}

// Main function
int main() {
  // Declare variables
  int M = 256;
  int N = 128;  // Adjusted for mxn matrix multiplication
  int K = 256; // Adjusted for mxn matrix multiplication
  int rep = 0;
  float *A, *B, *C;

  while(rep<100){
    // Allocate memory on the host
    A = (float *)malloc(sizeof(float) * M * K);
    B = (float *)malloc(sizeof(float) * K * N);
    C = (float *)malloc(sizeof(float) * M * N);

    // Fill the matrices with random numbers
    for (int i = 0; i < M * K; i++) {
      A[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
      B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeof(float) * M * K);
    cudaMalloc((void **)&d_B, sizeof(float) * K * N);
    cudaMalloc((void **)&d_C, sizeof(float) * M * N);

    // Copy the matrices to the device
    cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matrix_multiplication<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy the result back to the host
    cudaMemcpy(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Print the time taken for execution
    printf("%f ,\n", elapsedTime);

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    rep++;
  }

  return 0;
}
