## 1. Memery Coalescing

### Background
- GPU는 수천개의 스레드가 동시에 실행되는데, global memory에 어떻게 접근하는지에 따라 성능 다르다.
- 전역 메모리에서 데이터 가져와야 함. 
- but 전역 메모리는 느리고 대역폭 제한적.

한 warp(32 threads)가 global memory를 읽거나 쓸 때, 연속된 주소를 묶어서 차례대로 접근하면, GPU가 하나의 큰 드랜잭션으로 묶어서 읽는다.  
ex) 32개의 작은 요청 -> 1개의 큰 요청으로 묶음 -> bandwidth 효율 증가. latency 감소  
반대로 스레드들이 여기저기 주소를 접근하면, latency, bandwidth 낭비가 발생.  

![img](../img/6_1.png)

L2나 DRAM에서 왔다갔다 할 때, 한 번의 warp로 128byte로 묶어서 데이터를 주고 받는다.  
ex) 32쓰레드 x 4byte = 128byte  
데이터를 전송할 때 효율적으로 세그먼트 단위로 묶어서 "적은 수의 트랜잭션"으로 전송하기 위해 *Memory coalescing*이 중요하다!  

## 2. SoA vs AoS
### 좋은 경우

Structure of Arrays(SoA)
```cpp
struct pt {
  float x[N];
  float y[N];
  float z[N];
};
pt mypts;
```

메모리 배치: (x1 x2 x3 ...) (y1 y2 y3 ...)  
좌표마다 연속적으로 저장됨. -> transaction 1번으로 끝!  

stride를 가능하면 쓰지 않고, 쓰레드 주소가 i에 가까운 패턴일 때.  
시작 주소 정렬이 잘 맞을 때. 연속적 방식으로 접근!  

![img](../img/6_2.png)

### 나쁜 경우

Array of structures(AoS)  
```cpp
struct pt {
  float x;
  float y;
  float z;
};
pt mypts[N];
```

하나의 pt 구조체 안에 데이터가 연속으로 저장됨.    
메모리 배치: (x1, y1, z1) (x2, y2, z2) 등

x만 읽으려면?  
- thread 0 -> x0
- thread 1 -> x1
- thread 2 -> x2

>각 메모리마다 y, z도 있음. 콜리싱이 캐지고 여러 memory transaction 발생

| 구분              | 물리적 위치/크기   | 접근              | 지연 속도     | 활용도                   |
| --------------- | ----------- | --------------- | --------- | --------------------- |
| Global memory   | DRAM, 수 GB  | 모든 쓰레드 블록이 읽고 씀 | 느림        | cudaMalloc/cudaMemcpy |
| Shared memory   | SM 온칩, KB   | 같은 블록 쓰레드만 공유   | 빠름        | cache/convolution 연산  |
| Constant memory | DRAM의 특수 공간 | 모든 쓰레드 읽기 전용    | 브로드캐스트 최적 | 상수, 하이퍼파라미터           |
### bank
공유 메모리는 대역폭 늘리기 위해 여러 채널(bank)로 나누어져 있다.  
보통 32뱅크, 한 워프가 동시에 공유 메모리 접근.  

## 3. Access pattern

### Sequential access
- 워프의 스레드들이 연속 주소 읽고 쓰기.  
- sequential access가 가장 빠르다.
- host <-> device 갈 때 전송속도 빠르게 하려고 NVLink사용한다.

```cpp
// seq 쓰레드 i -> in[i]에 읽고 out[i]에 기록
__global__ void k_sequential(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] + 1.0f;
}
```
### Strided access
- 스레드마다 일정 stride를 두고 접근
- 1이면 잘 묶임. 커지면 비연속 -> 속도 저하

```cpp
// Strided: 스레드 i -> in[i * stride]
__global__ void k_strided(const float* in, float* out, int N, int STRIDE) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int j = (int)((1LL * i * STRIDE) % N);
        out[i] = in[j];
    }
}
```
### Random access
- 접근 위치 불규칙, 콜리싱 불가, 캐시 히트 낮음
### Gather/Scatter
- 불규칙 접근 중 인덱스 배열

> [!NOTE] Stream이란?
> GPU가 수행할 작업이 담긴 작업 큐

## 4. Synchronous vs Asynchronous transfer

### 1. 동기

```c
// 1. H -> D 복사. CPU 쓰레드는 대기
cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
// 2. 커널 런치는 비동기
vectorAdd<<<grid, block>>>(d_A, d_B, d_C, N);
// 3. 커널 완료 대기
cudaDeviceSynchronize(); 
// 4. D -> H 복사
cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost); // host로 복사
printf("결과: c[0] = %f\n", h_C[0]);
```
### 2. 비동기

```c
cudaStream_t s;
cudaStreamCreate(&s); // stream 생성
cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, s); // 복사를 stream에 넣는다.
cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, s);

vectorAdd<<<grid, block, 0, s>>>(d_A, d_B, d_C, N);

cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, s);

cudaStreamSynchronize(s); // stream에 넣은 모든 작업 끝날때까지 스레드 대기
printf("비동기 결과: c[0] = %f\n", h_C[0]);
```

로직 설명
1. 작업 큐 생성(스트림 생성)
2. cudaMemcpyAsync로 H -> D로 비동기 복사 스트림을 s에 enqueue한다.
3. 서로 겹치게 하려면 다른 스트림에 넣기

|     | 동기                                | 비동기                                            |
| --- | --------------------------------- | ---------------------------------------------- |
| 복사  | cudaMemcpy(dst, src, bytes, kind) | cudaMemcpyAsync(dst, src, bytes, kind, stream) |
| 커널  | kernel<<< >>>                     | kernel<<< ,stream >>>                          |
## 5. profiling 해보기

```bash
nsys stats profile_report.nsys-rep

cudaMalloc: 97% (438ms)
cudaMemcpy: 1.5% (7ms)
cudaMemcpyAsync: 0.7% (3ms)
cudaLaunchKernel: 0.5% (2ms)
```

CPU는 GPU를 기다리는 데 시간을 보낸다. (sem_wait, poll 등)  

![img](../img/6_3.png)

호스트(os)에서 CPU가 뭘 하고 시간을 어디에 썼는지 알 수 있다.  

```bash
vectorAdd ... Total 37 µs, 2번 실행

# GPU 메모리 작업 시간 분포
[CUDA memcpy DtoH]: 50%
[CUDA memcpy HtoD]: 50%

# GPU memory 전송 횟수 / 크기
HtoD: 4번, 각각 4MB
DtoH: 2번, 각각 4MB
```

