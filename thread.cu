#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "TheEmployeesSalary.h"

__global__ void TaskManager(const double* array, double* newSalaries, int size) {
    // 전역 쓰레드 인덱스
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // 커널에서 경계 검사
    if (id < size) {
        newSalaries[id] = array[id] + (array[id] * 15.0 / 100.0) + 5000;

    }
    // 만약 thread idx가 size를 초과하면? 
    // 아무 동작도 하지 않고 bound check를 통해 건너뛴다.
}

int main(void) {
    // 원소 개수
    int size = sizeof(TheArrayOfSalaries) / sizeof(TheArrayOfSalaries[0]);
    double* deviceArray; // host 배열 포인터
    double* deviceNewSalaries; // device 배열 포인터
    double newSalaries[100]; // 고정 크기 배열

    // 배열을 위한 연속된 메모리 8byte 확보
    cudaMalloc((void**)&deviceArray, size * sizeof(double)); 
    cudaMalloc((void**)&deviceNewSalaries, size * sizeof(double));

    // host 배열 -> device에 복사 -> 다시 host에 복사
    cudaMemcpy(deviceArray, TheArrayOfSalaries, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNewSalaries, newSalaries, size * sizeof(double), cudaMemcpyDeviceToHost);
    
    int threadsPerBlock = 256; // 블록 하나당 쓰레드 256개
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock; 
    // 커널 런치: 커널에 넘기기(디바이스 포인터)
    TaskManager<<<blocksPerGrid, threadsPerBlock>>>(deviceArray, deviceNewSalaries, size);
    cudaDeviceSynchronize(); // gpu가 마무리할 때까지 cpu 기다림.
    
    // newSalaries를 device -> host로 가져오기
    cudaMemcpy(newSalaries, deviceNewSalaries, size * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < size; i++) {
        printf("%f\n", newSalaries[i]);
    }
    
    cudaFree(deviceArray);
    return 0;
}