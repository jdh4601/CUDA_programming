#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void TaskManager(const double* array, double* newSalaries, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x; // 쓰레드 인덱스
    if (id < size) {
        newSalaries[id] = array[id] + (array[id] * 15.0 / 100.0) + 5000;
    }
}
cudaError_t thehelperfunction(const double* h_in, double* h_out, int size) {
    // 1) 입력 검증
    if (h_in == NULL || h_out == NULL || size < 0) {
        return cudaErrorInvalidValue;
    };

    cudaError_t st = cudaSuccess;
    double *d_in = NULL; // device input
    double *d_out = NULL; // device output
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    size_t nbytes = (size_t)size * sizeof(double); // why?

    // 2) 디바이스 메모리 할당
    // 실패하면? cudaErrorMemoryAllocation 반환 + 바로 메모리 해제
    st = cudaMalloc((void**)&d_in, nbytes);
    if (st != cudaSuccess) goto Cleanup;

    st = cudaMalloc((void**)&d_out, nbytes);
    if (st != cudaSuccess) goto Cleanup;
    
    // 3) host -> device으로 메모리 복사
    st = cudaMemcpy((void**)d_in, h_in, nbytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) goto Cleanup;

    // 4) 커널 런치
    TaskManager<<<blocks, threads>>>(d_in, d_out, size);
    // 4-1) LaunchError
    st = cudaGetLastError();
    if (st != cudaSuccess) goto Cleanup;
    // 4-2) SynchError
    st = cudaDeviceSynchronize();
    if (st != cudaSuccess) goto Cleanup;

    // 5) device -> host 메모리 복사
    st = cudaMemcpy(h_out, d_out, nbytes, cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) goto Cleanup;

Cleanup:
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    return st; // 성공이면 cudaSuccess, 실패하면 에러코드 반환
}