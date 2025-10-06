#include <stdio.h>
#include <cuda_runtime.h>

// 자식 커널: 실제 계산
__global__ void childKernel(int level) {
    printf("child kernel launched at level %d, thread %d\n", level, threadIdx.x);
}

// 부모 커널: 자식 커널을 실행
__global__ void parentKernel() {
    int tid = threadIdx.x;
    printf("parent kernel thread %d launching child kernel\n", tid);

    childKernel<<<1, 4>>>(tid); // GPU 내부에서 자식커널 실행
    
    cudaDeviceSynchronize(); // 동기화
}

int main() {
    int dev = 4;
    cudaSetDevice(dev);

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("GPU%d mem free %.2f GiB / %.2f GiB\n", dev, freeB/1024.0/1024/1024, totalB/1024.0/1024/1024);

    printf("launching parent kernel\n");
    parentKernel<<<1, 2>>>();
    cudaDeviceSynchronize();
    
    return 0;
}