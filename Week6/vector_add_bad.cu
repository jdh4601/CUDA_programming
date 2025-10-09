#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *A, float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int N = 1000000;
    size_t size = N * sizeof(float);
    
    printf("=== 나쁜 버전 (최적화 안 됨) ===\n");
    printf("벡터 크기: %d 원소\n\n", N);
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    for (int iter = 0; iter < 10; iter++) {
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);
        
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    printf("%.1f, %.1f, %.1f\n", h_A[999999], h_B[999999], h_C[999999]);
    
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
