#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const int bytes = N * sizeof(float);

    // Host 메모리
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // device 메모리
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    // 동기
    // 1. H -> D 복사. CPU 쓰레드는 대기
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    // 2. 커널 런치는 비동기
    vectorAdd<<<grid, block>>>(d_A, d_B, d_C, N);
    // 3. 커널 완료 대기
    cudaDeviceSynchronize(); 
    // 4. D -> H 복사
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost); // host로 복사
    printf("결과: c[0] = %f\n", h_C[0]);

    // 비동기
    cudaStream_t s;
    cudaStreamCreate(&s); // stream 생성

    cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, s); // 복사를 stream에 넣는다.
    cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, s);

    vectorAdd<<<grid, block, 0, s>>>(d_A, d_B, d_C, N);

    cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, s);
    
    cudaStreamSynchronize(s); // stream에 넣은 모든 작업 끝날때까지 스레드 대기
    printf("비동기 결과: c[0] = %f\n", h_C[0]);

    cudaStreamDestroy(s);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}