#include <cstdio>
#include <cuda_runtime.h>

#define DIM 1024

// CPU식 순회
void kernel(unsigned char *ptr) {
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;
            int juliaCalue = julia(x, y);
            ptr[offset*4 + 0] = 255 * juliaValue;
        }
    }
};

// gpu
__global__ void kernel(unsigned char *ptr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= DIM || y >= DIM) return;

    int value = julia(x, y); // 0 or 1
    unsigned char color = (unsigned char)(255 * value);

    int offset = (x + y * DIM) * 4;
    // Memory coalescing -> 4byte 간격
    ptr[offset + 0] = color;
    ptr[offset + 1] = color;
    ptr[offset + 2] = color;
    ptr[offset + 3] = 255;
};

// 구조체: 복소수(r: 실수부, i: 허수부)
struct cuComplex {
    float r;
    float i;

    cuComplex(float a, float b) : r(a), i(b) {}
    float magnitude2(void) { return r*r + i*i; }

    cuComplex operator*(const cuComplex&a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }

    cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

// julia on GPU -> __device__: gpu에서만 호출 가능
__device__ int julia(int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x) / (DIM/2);
    float jy = scale * (float)(DIM/2 - y) / (DIM/2);
    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    for (int i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000) {
            return 0;
        }
    }
    return 1;
}

int main(void) {
    CPUBitmap bitmap(DIM, DIM); // 비트맵 객체 생성
    unsigned char *ptf = bitmap.get_ptr();

    // device memory 할당
    cudaMalloc((void**)&dev_bitmap, bitmap.image_size());
    // 커널 실행
    kernel<<<grid, 1>>>(dev_bitmap); // 쓰레드 생성
    // 결과 host로 가져오기
    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    bitmap.display_and_exit(); // 실행
    cudaFree(dev_bitmap); // 메모리 해제
}

