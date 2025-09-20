#include <stdio.h>

__global__ void sumWithoutSync(int *data) {
    int tid = threadIdx.x;
    __shared__ int sum;

    if (tid == 0) sum = 0;
    data[tid] = tid;
    sum += data[tid]; 
    if (tid == 0) printf("without syn sum= %d\n", sum); // 0
}

__global__ void sumWithSync(int *data) {
    int tid = threadIdx.x;
    __shared__ int sum; // 공유 메모리

    if (tid == 0) sum = 0;
    __syncthreads(); // 모든 쓰레드 초기화 대기

    data[tid] = tid; 
    __syncthreads(); // d_data[0] ~ d_data[9]까지 입력 준비

    atomicAdd(&sum, data[tid]); // Atomic Operations
    __syncthreads(); // 모든 더하기가 끝날 때까지 기다림

    if (tid == 0) printf("with sync sum= %d\n", sum); // 45
}

int main() {
    const int N = 10; // thread 수
    int *d_data; // device 메모리를 가리키는 포인터

    // GPU 전역 메모리 블록을 확보 후, 그 시작 주소를 
    // 디바이스 포인터에 할당
    cudaMalloc(&d_data, N * sizeof(int)); 

    sumWithoutSync<<<1, N>>>(d_data);
    cudaDeviceSynchronize();

    sumWithSync<<<1, N>>>(d_data);
    cudaDeviceSynchronize();

    printf("The end.");
    cudaFree(d_data);
    return 0;
}