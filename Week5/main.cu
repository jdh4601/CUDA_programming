#include <stdio.h>
#include <cuda_runtime.h>

// device 전역 변수 -> local에 위치 x
__device__ int d_x;
__device__ int d_y;
__device__ int a;
__device__ int b;

// kernel
__global__ void addKernel(int x, int y, int* result) {
    *result = x + y;
}

__global__ void multiplyKernel(int a, int b, int* result) {
    *result = a * b;
}

int main() {
    int h_x = 2;
    int h_y = 4;    
    int *d_result_add, *d_result_multiply;
    int *h_add, *h_mul;

    cudaMallocHost(&h_add, sizeof(int));
    cudaMallocHost(&h_mul, sizeof(int));

    cudaStream_t s0, s1; // 각 디바이스별 스트림 생성

    // GPU device 1
    cudaSetDevice(0); // GPU 지정하기
    cudaStreamCreate(&s0); // stream 만들기
    cudaMalloc((void**)&d_result_add, sizeof(int)); // 4byte 메모리 할당
    addKernel<<<1, 1, 0, s0>>>(h_x, h_y, d_result_add);
    cudaMemcpyAsync(h_add, d_result_add, sizeof(int), cudaMemcpyDeviceToHost, s0);

    // GPU device 2
    cudaSetDevice(1);
    cudaStreamCreate(&s1);
    cudaMalloc((void**)&d_result_multiply, sizeof(int));
    multiplyKernel<<<1, 1, 0, s1>>>(h_x, h_y, d_result_multiply);
    cudaMemcpyAsync(h_mul, d_result_multiply, sizeof(int), cudaMemcpyDeviceToHost, s1);

    // 마지막에만 기다림
    cudaSetDevice(0); cudaStreamSynchronize(s0);
    cudaSetDevice(1); cudaStreamSynchronize(s1);

    printf("result_add: %d\n", *h_add); // 6
    printf("result multiply: %d\n", *h_mul); // 8

    cudaFree(d_result_add);
    cudaFree(d_result_multiply);

    return 0;
}