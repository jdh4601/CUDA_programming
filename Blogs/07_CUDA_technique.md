## 1. 병렬 처리

```c
#include <stdio.h>
#include <cuda_runtime.h>

// device 전역 변수 -> local에 위치 x
__device__ int d_x;
__device__ int d_y;
__device__ int a;
__device__ int b;

// kernel
__global__ void addKernel(int x, int y, int* result) {
    *result = x + y;
}

__global__ void multiplyKernel(int a, int b, int* result) {
    *result = a * b;
}

int main() {
    int h_x = 2;
    int h_y = 4;
    int result_add, result_multiply;
    
    // GPU device addition
    cudaSetDevice(0); // GPU 지정하기
    
    int* d_result_add;
    cudaMalloc((void**)&d_result_add, sizeof(int));
    cudaMemcpyToSymbol(d_x, &h_x, sizeof(int)); 
    cudaMemcpyToSymbol(d_y, &h_y, sizeof(int));

    addKernel<<<1, 1>>>(h_x, h_y, d_result_add);

    cudaMemcpy(&result_add, d_result_add, sizeof(int), cudaMemcpyDeviceToHost);

    // GPU device multiply
    cudaSetDevice(1);

    int* d_result_multiply;
    cudaMalloc((void**)&d_result_multiply, sizeof(int));

    cudaMemcpyToSymbol(a, &h_x, sizeof(int));
    cudaMemcpyToSymbol(b, &h_y, sizeof(int));

    multiplyKernel<<<1, 1>>>(h_x, h_y, d_result_multiply);

    cudaMemcpy(&result_multiply, d_result_multiply, sizeof(int), cudaMemcpyDeviceToHost);

    printf("result_add: %d\n", result_add); // 6
    printf("result multiply: %d\n", result_multiply); // 8

    cudaFree(d_result_add); 
    cudaFree(d_result_multiply);

    return 0;
}
```


`cudaMemcpy`: 메모리를 host -> device로 복사. 포인터만 인자로 받음.  
result_add는 host의 int변수이므로 &로 주소를 인자로 넣음. 포인터 변수는 그대로 넣음.

`cudaMemcpyToSymbol`: 디바이스 전역 메모리 변수(d_x)에 호스트 데이터 복사

### 이게 병렬 처리가 될까??
두 커널이 동시에 안 돈다.
이유는 cudaMemcpy를 호출해서 GPU0(device)의 작업이 끝날 때까지 host가 기다리기 때문이다.
그 다음에 cudaSetDevice(1)로 넘어가서 multiplyKernel을 실행하므로 "순차적"으로 커널이 실행된다.

#### 무엇을 조심할까?
1. stream의 암묵적 동기화 피하기
2. 블로킹 cudaMemcpy가 바로 동기화되는 것 피하기

## 2. 병렬 처리 하려면?

```c
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addKernel(int x, int y, int* result) {
    *result = x + y;
}

__global__ void multiplyKernel(int a, int b, int* result) {
    *result = a * b;
}

int main() {
    int h_x = 2;
    int h_y = 4;    
    int *d_result_add, *d_result_multiply;
    int *h_add, *h_mul;

    cudaMallocHost(&h_add, sizeof(int));
    cudaMallocHost(&h_mul, sizeof(int));

    cudaStream_t s0, s1; // 각 디바이스별 스트림 생성

    // GPU device 1
    cudaSetDevice(0); // GPU 지정하기
    cudaStreamCreate(&s0); // stream 만들기
    cudaMalloc((void**)&d_result_add, sizeof(int)); // 4byte 메모리 할당
    addKernel<<<1, 1, 0, s0>>>(h_x, h_y, d_result_add);
    cudaMemcpyAsync(h_add, d_result_add, sizeof(int), cudaMemcpyDeviceToHost, s0);

    // GPU device 2
    cudaSetDevice(1);
    cudaStreamCreate(&s1);
    cudaMalloc((void**)&d_result_multiply, sizeof(int));
    multiplyKernel<<<1, 1, 0, s1>>>(h_x, h_y, d_result_multiply);
    cudaMemcpyAsync(h_mul, d_result_multiply, sizeof(int), cudaMemcpyDeviceToHost, s1);

    // 마지막에만 기다림
    cudaSetDevice(0); cudaStreamSynchronize(s0);
    cudaSetDevice(1); cudaStreamSynchronize(s1);

    printf("result_add: %d\n", *h_add); // 6
    printf("result multiply: %d\n", *h_mul); // 8

    cudaFree(d_result_add);
    cudaFree(d_result_multiply);

    return 0;
}
```

각 디바이스별로 스트림을 만들고
커널과 복사를 async로 걸고, 
마지막에 cudaStreamSync로 기다림.
### Stream이란?
GPU에 보낼 작업들의 FIFO(순서 있는 큐)
한 스트림(s0) 안에 들어간 작업들은 순서가 보장된다.
다른 스트림 s0 vs s1은 동시에 진행될 수 있다.
호스트 메모리는 pinned메모리여야 진짜 비동기가 된다.

```c
// GPU 0
cudaSetDevice(0);
cudaStreamCreate(&s0);
addKernel<<<1, 1, 0, s0>>>(h_x, h_y, d_result_add);
cudaMemcpyAsync(h_add, d_result_add, sizeof(int), cudaMemcpyDeviceToHost, s0);

// GPU 1
cudaSetDevice(1);
cudaStreamCreate(&s1);
multiplyKernel<<<1, 1, 0, s1>>>(h_x, h_y, d_result_multiply);
cudaMemcpyAsync(h_mul, d_result_multiply, sizeof(int), cudaMemcpyDeviceToHost, s1);

// 마지막에만 기다림
cudaSetDevice(0); cudaStreamSynchronize(s0);
cudaSetDevice(1); cudaStreamSynchronize(s1);
```

각 커널이 각각 다른 스트림에 들어갔다. -> 커널이 끝나면 cudaMemcpyAsync가 시작됨.
cudaStreamSynchronize 를 통해 출력 직전에만 기다린다!

## 3. CUDA library

### 1. cuBLAS
선형대수 연산을 GPU로 가속화해주는 라이브러리
ex) 행렬 곱, 벡터 덧셈, dot product, LU 분해, 선형 방정식 풀이 같은 수학 연산
### 2. cuRAND
GPU상에서 난수를 빠르게 생성하는 라이브러리
난수 1억개를 동시에 만들 수 있다.
ex) 몬테카를로 시뮬레이션, 데이터 초기화, 딥러닝 weight 초기화 등

```cpp
#include <stdio.h>
#include <curand.h>
#include <cuda_runtime.h>

int main() {
    const int size = 10;
    unsigned int* d_data;
    unsigned int* h_data = (unsigned int*)malloc(size * sizeof(unsigned int));

    cudaMalloc(&d_data, size * sizeof(unsigned int));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // 난수를 GPU 메모리 d_data에 생성
    curandGenerate(gen, d_data, size);

    // GPU → CPU 복사
    cudaMemcpy(h_data, d_data, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++)
        printf("%u\n", h_data[i]);

    curandDestroyGenerator(gen);
    cudaFree(d_data);
    free(h_data);
}
```

결과
1952543333
1768902757
1852797811

- `curandCreateGenerator`: 난수 생성기 초기화. GPU에서 난수 만들 준비
- `curandGenerate`: 실제 난수 생성, 난수 저장할 메모리 주소(디바이스 포인터)에 생성할 개수만큼 생성한다.
### 3. cuDNN
딥러닝 연산 전용 GPU 가속 라이브러리
CNN, Pooling, Conv, Normalization, ReLU 등의 연산 최적화

```python
conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
```

## 4. Dynamic Parallelism (동적 병렬성)

GPU가 GPU를 제어할 수 있게 해주는 기술

1. CPU가 GPU에서 커널을 실행시킴
2. GPU가 병렬로 작업을 처리함.
3. GPU에서 작업이 끝나면 CPU로 제어가 돌아옴.
CPU -> GPU -> CPU -> GPU ... 이렇게 진행된다.

>여기서 동적 병렬성은 GPU에서 실행 중인 커널이 또 다른 커널을 실행할 수 있다.

CPU가 grid A를 실행. -> Grid A가 Grid B를 실행 -> Grid B가 끝나면 Grid A가 실행 -> Grid A가 끝나면 CPU로 넘어감.

```c
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
    printf("launching parent kernel\n");
    parentKernel<<<1, 2>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

CPU -> main()  
*parentKernelI() 실행* -> 
```
parent kernel thread 0 launching child kernel
parent kernel thread 1 launching child kernel
```

*childKernel() 실행* -> 
```
GPU4 mem free 14.52 GiB / 23.55 GiB
launching parent kernel
parent kernel thread 0 launching child kernel
parent kernel thread 1 launching child kernel
child kernel launched at level 0, thread 0
child kernel launched at level 0, thread 1
child kernel launched at level 0, thread 2
child kernel launched at level 0, thread 3
child kernel launched at level 1, thread 0
child kernel launched at level 1, thread 1
child kernel launched at level 1, thread 2
child kernel launched at level 1, thread 3
```

*parentKernel() 동기화* -> CPU

## 5. Warp Divergence

GPU는 SIMT(single instruction, multiple thread) 모델
하드웨어는 warp라는 32개의 묶음으로 실행, 한 워프 내의 쓰레드는 모두 같은 명령을 동시 수행한다.

if, switch같이 분기가 생기면 한 쪽은 활성화, 다른 쪽은 비활성화 시킨 채, 순차적으로 실행한다.
>병렬성이 줄어들어 느려진다!

GPU는 나중에 경로가 합쳐지면 reconvergence를 해서 다시 워프를 돌리지만, 그동안 idle 쓰레드가 생겨 warp execution efficiency가 떨어진다. -> 낭비!!!

### 전형적인 warp divergence branch 코드

```c title:warp_divergence_brance
__global__ void divergentKernel(const int* in, int* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride) {
        int v = in[i];
        if ((v & 1) == 0) {
            out[i] = v * 2; // 짝수면 2배
        } else {
            out[i] = v;     // 홀수면 그대로
        }
    }
}

int main() {
    const int N = 10;

    int h_in[N]  = {1,2,3,4,5,6,7,8,9,10};
    int h_out[N] = {0};

    int *d_in, *d_out;
    cudaSetDevice(4); // 4번 gpu 실행

    cudaMalloc((void**)&d_in,  N * sizeof(int));
    cudaMalloc((void**)&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    divergentKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    cudaDeviceSynchronize(); 

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("input:\n");
    for (int i = 0; i < N; i++) printf("%d ", h_in[i]);
    printf("\n");
    printf("output:\n");
    for (int i = 0; i < N; i++) printf("%d ", h_out[i]);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
```

결과
```
input:
1 2 3 4 5 6 7 8 9 10 
output:
1 4 3 8 5 12 7 16 9 20
```

### 1. Predication
분기로 갈라지지 않고, 두 경로를 모두 계산해두고 조건에 따라 결과만 선택.  
devergence는 사라지지만, 불필요한 계산이 많아짐...(두 경로 모두 계산).  

```c
int evenResult = v * 2; // 1. 짝수 경로 계산
int oddResult = v; // 2. 홀수 경로 계산
bool isEven = ((v & 1) == 0); // 3. 조건 계산

out[i] = isEven ? evenResult : oddResult; // 4. isEven에 따라 결과 선택
```

단일 명령어 프름으로 바꿔서 계산한다!
### 2. Branch divergence hiding
두 경로 결과를 다 만들고, 조건으로 결과만 선택하는 방식.  
predication의 상위 개념이다. 
>Branch divergence hiding = predication + 다른 완화 기법들