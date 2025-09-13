#include <cuda_runtime.h>
#include <iostream>

int main() {
    int count = 0;
    cudaGetDeviceCount(&count); // 실행 가능한 CUDA 5개

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);

        std::cout << "Device " << i << ": " << prop.name << "\n"
                  << "  Compute Capability: " << prop.major << "." << prop.minor << "\n"
                  << "  Global Memory: " << (prop.totalGlobalMem >> 20) << " MB\n"
                  << "  Free memory: " << (freeMem >> 20) << " MB\n"
                  << "  Total memory: " << (totalMem >> 20) << " MB\n"
                  << "  Shared Memory per Block: " << (prop.sharedMemPerBlock >> 10) << " KB\n"
                  << "  Warp Size: " << prop.warpSize << "\n"
                  << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n"
                  << "  Max Grid Size: " << prop.maxGridSize[0] << " x "
                                       << prop.maxGridSize[1] << " x "
                                       << prop.maxGridSize[2] << "\n\n";
    }
    return 0;
}