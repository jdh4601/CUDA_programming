## 1. 개발 환경 세팅하기

1. NVIDIA의 nvcc 컴파일러를 설치하기

```bash
nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

2. 컴파일 & 실행

```bash
# 컴파일하기(파일 이름을 .cu로 하기)
nvcc main.cu -o main 
# 실행하기
./main
```

> [!NOTE] nvcc(NVIDIA CUDA Compiler)란?
> CUDA 전용 컴파일러. C++과 kernel code로 나눈 후,
> CPU용 C++ 코드는 gcc, clang 컴파일러가 실행되고
> GPU용 커널 코드는 PTX(Parallel Thread Execution) 코드와 GPU 바이너리 코드로 변환해서 하나의 실행 파일로 링크해준다.
## 2. host <-> device

```cpp
// GPU(device)에서 실행되는 코드
__global__ void add(int a, int b, int *c) { 
    // a, b는 값으로 복사됨. c는 주소값으로 전달
    *c = a + b; // c가 가리키는 global memory에 값 할당
}

// 커널에서 호출하는 함수(host에서 실행)
int main(void) {
    int c;
    int *dev_c; // 포인터 변수 선언


    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));
    add<<<1, 1>>>(2, 7, dev_c);

    
    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

    printf("2 + 7 = %d\n", c);
    cudaFree(dev_c); // 디바이스 메모리 해제

    return 0;
}
```

`__global__`:  host(cpu)가 커널을 실행해서 device(GPU)에서 한 번 연산을 한 뒤, 결과값을 device -> host로 가져오는 함수

`int main`: 커널에서 호출하는 함수

`(void**)`: 형변환 cast. 매개변수 타입과 맞출 때 사용함.

`int *dev_c`로 선언한 포인터 변수의 타입을 맞춰준다. (int -> void**)

`cudaMalloc((void**)&dev_c, sizeof(int));`
cudaMalloc으로 만든 포인터는 GPU 글로벌 메모리에 공간을 만들고, 그 공간의 주소를 반환.
device 메모리에 sizeof(int) 만큼 할당(4byte) -> 그 디바이스 주소를 &dev_c에 할당한다.
device메모리를 핸들링하려면 커널 안에서 접근 or cudaMemcpy로 복사해서 다루기

(절대 CPU 코드에서 디바이스 포인터를 역참조(`*dev_c`)해서 사용하면 안된다. 왜냐하면 GPU와CPU는 다른 메모리 공간을 가지므로 CPU입장에서는 가리키는 위치가 존재하지 않는 것처럼 보이기 때문이다.)

`add<<<1, 1>>>(2, 7, dev_c);`
블록 1개, 블록 당 스레드 1개 커널을 비동기로 실행.
dev_c라는 디바이스 주소값(포인터변수의 값)이 kernel로 넘겨준다.

`cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost)`
디바이스에서 호스트로 메모리에 접근할 때 cudaMemcpyDeviceToHost를 사용함.
디바이스 주소 dev_c에서 4byte를 읽어서 호스트 주소 &c로 가져온다.

`cudaMemcpy()`: 
호스트 포인터 -> 호스트 코드로부터 메모리에 접근 가능.
디바이스 포인터 -> 디바이스 코드로부터 메모리 접근 가능.
>호스트 코드로부터 cudaMemcpy()를 통해 *디바이스 메모리에 접근* 가능

`cudaFree(dev_c)`: 디바이스 메모리 해제
## 3. Querying Devices

device가 얼마나 많은 메모리와 capability를 가지고 있는지 알기 어렵다.
>determine which processor is which!

Warp: 

```cpp
int main(void) {
    int count = 0;
    // cuda장치 수를 count에 할당
    cudaError_t err = cudaGetDeviceCount(&count); 

    if (err != cudaSuccess) {
        std::cerr << "cuda error" << cudaGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "CUDA devices found: " << count << "\n";
    return 0;
}
```

`cudaGetErrorString(err)`: CUDA 에러를 사람이 읽을 수 있게 변환
`cudaGetDeviceCount()`: CUDA를 실행하기 전에 얼마나 많은 device가 있는지 파악하기

>결과: CUDA devices found: 5

```cpp
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int count = 0;
    cudaGetDeviceCount(&count); // 실행 가능한 CUDA 5개

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop; // 구조체 변수
        cudaGetDeviceProperties(&prop, i);
		
		// do something with my device's properties
        std::cout << "Device " << i << ": " << prop.name << "\n"
    }
    return 0;
}
```

`cudaGetDeviceProperties(&prop, i);`
i번째 CUDA 디바이스의 property를 가져와서 prop이라는 구조체 변수에 할당한다.

`cudaDeviceProp`
한 개 GPU 장치의 모델명, 메모리 크기, 워프 크기, 최대 스레드 수 등 
실행 환경에 필요한 수많은 속성을 담는 컨테이너(구조체).

```cpp title:cudaDeviceProp
struct cudaDeviceProp {
	char name[256];
	size_t totalGlobalMem;
	int warpSize;
	...
}
```

*출력 결과*
Device 0: NVIDIA L40S
  Compute Capability: 8.9
  Global Memory: 45487 MB
  Shared Memory per Block: 48 KB
  Warp Size: 32 (동시에 실행되는 쓰레드 수)
  Max Threads per Block: 1024 (블록 당 최대 스레드 수)
  Max Grid Size: 2147483647 x 65535 x 65535

| 필드 이름              | 의미                                                                | 예시 출력                                    |
| ------------------ | ----------------------------------------------------------------- | ---------------------------------------- |
| name[256]          | GPU 모델명                                                           | "NVIDIA L40S", "NVIDIA GeForce RTX 4090" |
| major, minor       | **Compute Capability** (지원 기능 세대)                                 | 8.9                                      |
| totalGlobalMem     | GPU의 **전체 글로벌 메모리 용량(바이트)**                                       | 45487 MB, 24111 MB                       |
| sharedMemPerBlock  | 한 **스레드 블록**이 사용할 수 있는 **공유 메모리 크기(바이트)**                         | 48 KB                                    |
| regsPerBlock       | 블록당 사용할 수 있는 **레지스터 수**                                           | (출력 예시는 생략했지만 내부적으로 존재)                  |
| warpSize           | **워프(warp)** 크기 (동시에 실행되는 스레드 수)                                  | 32                                       |
| memPitch           | 2D 메모리 복사 시 최대 pitch(바이트)                                         | –                                        |
| maxThreadsPerBlock | 하나의 **스레드 블록**이 가질 수 있는 **최대 스레드 수**                              | 1024                                     |
| maxThreadsDim[3]   | 한 블록의 x·y·z 방향 **최대 스레드 수**                                       | 예: (1024,1024,64)                        |
| maxGridSize[3]     | 그리드의 x·y·z 방향 **최대 블록 수**                                         | 2147483647 x 65535 x 65535               |
| totalConstMem      | 사용할 수 있는 **상수 메모리 크기(바이트)**                                       | –                                        |
| 추가 필드 (현대 CUDA)    | PCIe 버스 ID, 클럭 속도, L2 캐시 크기, 멀티프로세서 수(multiProcessorCount) 등 더 많음 | –                                        |

```cpp
# 사용 가능한 메모리를 보려면?
size_t freeMem, totalMem;
cudaMemGetInfo(&freeMem, &totalMem);

std::cout << "Free memory: " << (freeMem >> 20) << " MB\n";
std::cout << "Total memory: " << (totalMem >> 20) << " MB\n";
```

*출력 결과*
Device 0: NVIDIA L40S
  Compute Capability: 8.9
  Global Memory: 45487 MB
  Free memory: 2122 MB -> 현재 사용 가능한 메모리
  Total memory: 45487 MB -> 전체 메모리
  Shared Memory per Block: 48 KB -> 쓰레드들이 공유하는 SRAM
  Warp Size: 32
  Max Threads per Block: 1024
  Max Grid Size: 2147483647 x 65535 x 65535 -> 커널 실행 시 지정할 수 있는 블록 배열의 최대 크기

**constant memory**: GPU에 있는 읽기 전용 캐시 메모리 (약 64KB)
**Global Memory / Total Memory**: GPU가 제공하는 전체 글로벌 메모리 용량 (약 48GB)
**Free Memory**: 현재 사용 가능한 메모리 용량



