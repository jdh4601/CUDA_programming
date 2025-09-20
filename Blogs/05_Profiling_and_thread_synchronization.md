## 1. NVIDIA Nsight Systems

*NVIDIA Profiler*: 엔비디아 GPU 성능 분석기.
CUDA 프로그램이 GPU와 CPU를 어떻게 사용할 수 있는지 세밀하게 추적하는 도구.
>CUDA 코드가 어디서 뭘 하느라 시간을 쓰는지 파악한다!

`"nvprof is not supported on devices with compute capability 8.0 and higher."`
> CUDA 11이후 deprecated되고 최신 nsight 도구로 교체됨. -> *NVIDIA Nsight Systems* 사용!

CLI 기반 프로파일러가 수집하는 정보들:
- 각 GPU 커널의 실행 시간
- 메모리 전송 비용
- CPU <-> GPU 동기화 지점
- CUDA API 호출 횟수 & 소요 시간
### Nsight systems 실행 방법

```bash
nsys profile -o profile_report ./main
```

결과:
```bash
Generating '/tmp/nsys-report-b318.qdstrm'
[1/1] [========================100%] profile_report.nsys-rep
Generated:
    /data/donghyun/cuda/profile_report.nsys-rep
```

`profile_report.nsys-rep` 파일 실행해보기

```bash
nsys stats profile_report.nsys-rep
```

결과:
![[Screenshot 2025-09-20 at 5.05.15 PM.png]]

```bash
 ** CUDA API Summary (cudaapisum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)     Min (ns)   Max (ns)     StdDev (ns)            Name         
 --------  ---------------  ---------  -------------  -------------  --------  -----------  -------------  ----------------------
     99.7      236,513,261          2  118,256,630.5  118,256,630.5     7,905  236,505,356  167,228,951.3  cudaMalloc            
      0.2          552,188          1      552,188.0      552,188.0   552,188      552,188            0.0  cudaLaunchKernel      
      0.1          179,755          2       89,877.5       89,877.5     9,249      170,506      114,025.9  cudaFree              
      0.0           63,091          2       31,545.5       31,545.5    28,413       34,678        4,430.0  cudaMemcpy            
      0.0            8,007          1        8,007.0        8,007.0     8,007        8,007            0.0  cudaDeviceSynchronize 
      0.0            2,825          1        2,825.0        2,825.0     2,825        2,825            0.0  cuModuleGetLoadingMode
```
### CUDA API Summary

```
99.7% cudaMalloc
0.2% cudaLaunchKernel
0.1% cudaFree
```

cudaMalloc이 전체 호출 시간 대부분 차지.
>GPU 메모리 할당하는 작업이 비싼 연산이다!

반면 cudaLaunchKernel(커널 실행)은 가벼운 연산이다.

---
## 2. Thread synchronization

*Thread synchronization*이란?
병렬 컴퓨팅에서 여러 쓰레드가 서로 조율된 방식으로 작업하는 기술.
모든 쓰레드가 특정 지점에 도달할 때까지 기다리게 만든다.

*Race Condition*: 2개 이상의 스레드가 동시에 같은 자원에 접근할 때, 실행 순서에 따라 결과가 달라지는 상황.
(ex 같은 자리를 2명이 예약했을 때, 한 명이 찜을 해놓고 결제 전까지 lock을 해서 동시에 예약할 수 없게 만들어야 한다)
### Example

```cpp title:sync.cu
#include <stdio.h>
// 동기화 없이 실행
__global__ void sumWithoutSync(int *data) {
    int tid = threadIdx.x; 
    __shared__ int sum;

    if (tid == 0) sum = 0;
    data[tid] = tid;
    sum += data[tid];
    if (tid == 0) printf("without syn sum=%d\n", sum); // 0
}

// 동기화 실행
__global__ void sumWithSync(int *data) {
    int tid = threadIdx.x;
    __shared__ int sum;

    if (tid == 0) sum = 0;
    __syncthreads();

    data[tid] = tid;
    __syncthreads();

    atomicAdd(&sum, data[tid]);
    __syncthreads();

    if (tid == 0) printf("with sync sum=%d\n", sum); // 45
}

int main() {
    const int N = 10; // thread 수
    int *d_data; // device 메모리를 가리키는 포인터

    // GPU 전역 메모리 블록을 확보 후, 그 시작 주소를 
    // 디바이스 포인터에 할당
    cudaMalloc(&d_data, N * sizeof(int)); // 40byte 요청

    sumWithoutSync<<<1, N>>>(d_data);
    cudaDeviceSynchronize();

    sumWithSync<<<1, N>>>(d_data);
    cudaDeviceSynchronize();

    printf("The end.");
    cudaFree(d_data);
    return 0;
}
```
#### sumWithoutSync

```cpp
if (tid == 0) sum = 0;
data[tid] = tid;
sum += data[tid]; 
if (tid == 0) printf("without syn sum= %d\n", sum);
```

- thread들의 업데이트가 반영되지 않을수도.
- sum+= 이 반영되기 전에 printf가 실행되어버림.
- "0" 출력(결과가 엉켜버림)
#### sumWithSync

```cpp
if (tid == 0) sum = 0;
__syncthreads();
data[tid] = tid;
__syncthreads();
atomicAdd(&sum, data[tid]);
__syncthreads();
```

-> 모든 쓰레드 초기화될때까지 대기.
-> 모든 d_data[] 이 입력될 때까지 대기.
-> atomicAdd를 통해 더하기가 끝날때까지 대기
-> "45" 출력.

| method            | Granularity    | Definition                                |
| ----------------- | -------------- | ----------------------------------------- |
| Locks             | Coarse-grained | 모든 공유 자원에 동시에 한 쓰레드만 접근할 수 있도록 locking한다. |
| Brarriers         | Block / Warp   | 모든 쓰레드가 특정 지점에 도착할 때까지 All stop.          |
| Atomic Operations | Fine-grained   | read-edit-write를 하나의 불가분한 연산으로 처리.        |
### 1. Locks

GPU같은 하드웨어에서는 lock를 사용하면 안된다.
1024개를 동시에 돌리려면 lock을 쓰지 말아야 한다. 병렬로 실행해야 한다.
thread가 끝날 때까지만 기다려야 하기 때문에.
#### 주의할 점
if를 쓸수록 분기가 돼서 하나의 condition을 줄여야 한다. -> CPU는 분기 예측하기도 함. 
device에서 실행하려면 커널 안에 if같은 분기를 조심해야 한다.
여러 쓰레드가 락을 얻기 위해 끝날 때까지 기다려야 하기 때문이다. (병렬성 저하)
(Deadlock: 서로 다른 쓰레드가 서로의 락을 기다리다가 영원히 멈춤)
피해야 하는 건 ‘워프 내에서 서로 갈라지는, *데이터 의존적 분기*다!!!

>GPU 철학: 분기 없기 그냥 단순 무식 연산 많이 하기

> [!NOTE] Warp(워프)란?
> GPU 실행의 최소 단위. "32개의 쓰레드"를 한 묶음으로 만든 것.
> 쓰레드는 동일한 명령어를 한 번에 실행.
> SIMT(Single instruction, Multiple Threads) 방식
> 32개의 쓰레드 군단 하나를 워프라고 보면 됨.
> **그리드(해군 전체) → 블록(11전단) → 워프(1함대) → 스레드(대구함)**

어떤 분기를 말하는 걸까?
### 2. Barriers

```cpp title:예시_코드
__global__ void sumWithSync(int *data) {
    int tid = threadIdx.x;
    __shared__ int sum; // 공유 메모리

    if (tid == 0) sum = 0;
    __syncthreads(); // 모든 쓰레드 초기화 대기

    data[tid] = tid; 
    __syncthreads(); // d_data[0] ~ d_data[9]까지 입력 준비

    atomicAdd(&sum, data[tid]); // 더하기
    __syncthreads(); // 모든 더하기가 끝날 때까지 기다림

    if (tid == 0) printf("with sync sum= %d\n", sum); // 45
}
```

`__syncthreads()`: 컨트롤 동기화. 
중간 체크포인트 기다렸다가 다같이 이동하기 약속함.
(scope: all threads in a block)

```cpp title:예시_코드
__global__ void writer() {
    data = 42;          // 데이터 작성
    __threadfence();    // 전역 메모리에 쓰기가 완료되었음을 보장
    flag = 1;           // 다른 블록이 확인할 신호
}
```

`__threadfence()`: 메모리 동기화. 
한 쓰레드가 GPU전역 메모리에 쓴 결과를 다른 블록의 쓰레드들이 볼 수 있도록 보장하는 메모리 동기화 함수.
커널은 flag가 1이 되는 것을 확인하면, 그때 data를 읽는다.
### 3. Atomic Operations
실행 중간에 다른 스레드가 끼어들 수 없는, 쪼갤 수 없는 단일 연산.
race condition 상태를 예방하기 위해 `sumAtomic, atomicSub`같은 함수 제공

ex) x += 4는 사실 "읽기 → 더하기 → 쓰기" 이렇게 3단계로 나뉘어져 있다.
따라서 여러 쓰레드가 동시에 병렬 연산을 하면 값이 "한 번만" 증가할 수도 있다.

```cpp title:atomic_operations
__global__ void sumAtomic(int *data, int *sum) {
    int tid = threadIdx.x;
    atomicAdd(sum, data[tid]);
}
```

>atomicAdd() 는 더 이상 쪼갤 수 없다.
>`__syncthreads()`만으로는 값 유실을 막기 어렵다.ㅠ

*Atomic Operations의 장점*
- hardware support: 더 빠르게 동작.
- granularity: protect a single memory location
- Contention Behavior: 100개 쓰레드 몰려도 각각 다른 곳에 접근 가능
- GPU 아키텍처 fit: GPU에 맞게 동작.

