#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "TheEmployeesSalary.h"

cudaError_t thehelperfunction(const double* h_in, double* h_out, int size);

int main(void) {
    int size = (int)(sizeof(TheArrayOfSalaries) / sizeof(TheArrayOfSalaries[0]));
    double newSalaries[100]; // 여유 버퍼

    if (size > (int)(sizeof(newSalaries)/sizeof(newSalaries[0]))) {
        fprintf(stderr, "size too large for newSalaries buffer\n");
        return 1;
    }

    cudaError_t st = thehelperfunction(TheArrayOfSalaries, newSalaries, size);
    
    if (st != cudaSuccess) {
        fprintf(stderr, "CUDA failed: %s\n", cudaGetErrorString(st));
        return 1;
    }

    for (int i = 0; i < size; ++i) {
        printf("%f\n", newSalaries[i]);
    }
    return 0;
}