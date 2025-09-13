#include <cuda_runtime.h>
#include <iostream>

// 최소 에러 체크 매크로 (book.h 대체)
#define HANDLE_ERROR(x) do { \
  cudaError_t _err = (x); \
  if (_err != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(_err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} while (0)


int main(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count); // cuda장치 수를 count에 할당

    if (err != cudaSuccess) {
        // cudaGetErrorString(err): CUDA 에러를 사람이 읽을 수 있게 변환
        std::cerr << "cuda error" << cudaGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "CUDA devices found: " << count << "\n";
    return 0;
}