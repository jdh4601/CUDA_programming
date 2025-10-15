#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INF 2e10f
#define SPHERES 20
#define rnd(x) (x * rand() / RAND_MAX)
#define DIM 1024

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

class CPUBitmap {
private:
    unsigned char *pixels;
    int width;
    int height;

public:
    CPUBitmap(int w, int h) : width(w), height(h) {
        pixels = (unsigned char*)malloc(w * h * 4);
        printf("CPUBitmap created: %dx%d (%ld bytes)\n", w, h, w*h*4L);
    }

    ~CPUBitmap() {
        free(pixels);
    }

    unsigned char* get_ptr() {
        return pixels;
    }

    long image_size() const {
        return width * height * 4;
    }

    void save_ppm(const char* filename) {
        FILE* fp = fopen(filename, "wb");
        if (!fp) {
            printf("Error: Cannot open file %s\n", filename);
            return;
        }
        fprintf(fp, "P6\n%d %d\n255\n", width, height);
        for (int i = 0; i < width * height; i++) {
            fputc(pixels[i*4 + 0], fp);  // R
            fputc(pixels[i*4 + 1], fp);  // G
            fputc(pixels[i*4 + 2], fp);  // B
        }
        fclose(fp);
        printf("Image saved to %s\n", filename);
    }
};

struct Sphere {
    float r, g, b;
    float radius;
    float x, y, z;

    __device__ float hit(float ox, float oy, float *n) {
        float dx = ox - x;
        float dy = oy - y;

        if (dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf(radius*radius - dx*dx - dy*dy);
            *n = dz / sqrtf(radius*radius);
            return dz + z;
        }
        return -INF;
    }
};

__constant__ Sphere s[SPHERES];

__global__ void kernel(unsigned char* ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float ox = x - DIM/2.0f;
    float oy = y - DIM/2.0f;
    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = s[i].hit(ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    }

    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255;
}

int main(void) {
    int device = 3;
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("Available GPUs: %d\n", deviceCount);
    CUDA_CHECK(cudaSetDevice(device));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0)); // 시작 지점 기록

    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    CUDA_CHECK(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
    printf("GPU memory allocated: %ld bytes\n", bitmap.image_size());
    
    Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
    printf("\nGenerating %d random spheres:\n", SPHERES);

    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 1000;
        temp_s[i].radius = rnd(100.0f) + 20;
    }

    // host -> constant memory 복사
    CUDA_CHECK(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));
    free(temp_s);

    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16, 16);
    printf("Launching kernel: (%d, %d) grids × (%d, %d) threads = %d total threads\n",
           grids.x, grids.y, threads.x, threads.y, grids.x * grids.y * threads.x * threads.y);
    
    kernel<<<grids, threads>>>(dev_bitmap); // 측정할 GPU 작업
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), 
                          cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaEventRecord(stop, 0)); // 종료 지점 기록
    CUDA_CHECK(cudaEventSynchronize(stop)); // CPU 대기!!!

    float elapsedTime;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop)); // 정리
    printf("\nTime to generate: %.1f ms\n\n", elapsedTime); // 7.3 ms
    
    bitmap.save_ppm("output.ppm");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dev_bitmap));

    printf("\nRay tracing complete!\n");
    return 0;
}
