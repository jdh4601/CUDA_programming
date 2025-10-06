#include <stdio.h>
#include <cuda_runtime.h>
#define CHECK(call) do{                                         \
  cudaError_t e=(call);                                         \
  if(e!=cudaSuccess){                                           \
    fprintf(stderr,"CUDA error %s:%d: %s\n",                    \
            __FILE__,__LINE__,cudaGetErrorString(e));           \
    return 1;                                                   \
  }                                                             \
}while(0)

__global__ void divergentKernel(const int* in, int* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride) {
        int v = in[i];
        if ((v & 1) == 0) {
            out[i] = v * 2; // 짝수면 2배
        } else {
            out[i] = v;     // 홀수면 그대로
        }
    }
}

int main() {
    const int N = 10;

    int h_in[N]  = {1,2,3,4,5,6,7,8,9,10};
    int h_out[N] = {0};

    int *d_in, *d_out;
    cudaSetDevice(4); // 4번 gpu 실행

    cudaMalloc((void**)&d_in,  N * sizeof(int));
    cudaMalloc((void**)&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    divergentKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    cudaDeviceSynchronize(); 

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("input:\n");
    for (int i = 0; i < N; i++) printf("%d ", h_in[i]);
    printf("\n");
    printf("output:\n");
    for (int i = 0; i < N; i++) printf("%d ", h_out[i]);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}