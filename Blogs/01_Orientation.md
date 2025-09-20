## 스터디 목표

- CUDA 병렬 프로그래밍 모델(스레드, 블록, 그리드)
- CUDA 커널 작성
- 메모리 관리와 스트림 적용
- Nsight를 활용한 간단한 CUDA 프로그램 프로파일링 및 최적화
- 최종 프로젝트 실습
- CUDA by Example(2010) 바이블 정독.
## 초기 GPU 컴퓨팅

gpu는 연구자들에게 openGL, directX기반의 렌더링 용도였다.
연구자들은 색상이 아닌 숫자 데이터로 GPU에게 계산을 수행하도록 했고,
GPU는 자신이 렌더링 작업이 아닌 계산 작업을 수행했다는 사실을 몰랐다.
GPU를 속여서 비렌더링 작업을 수행한 트릭으로 시작했다.
높은 연산 처리량 덕분에 좋았으나, 유연하진 않았다.
메모리 쓰기 위치를 임의로 지정해야 하고 부동소수점 연산을 어떻게 처리할지 예측되지 않았다.
범용 계산에 GPU를 사용하려면 shading language라는 전용 언어를 익혀야 했던 것이 장벽이었다.

## CUDA 아키텍처란?

CUDA아키텍처는 통합 shader pipline을 도입해서 칩의 ALU를 범용 계산에 자유롭게 이용할 수 있게 했다.
엔비디아는 새로운 GPU가 범용 계산에 가능하게 설계하여, IEEE 부동소수점 연산에 부합하게 했다.
단일 그래픽 작업이아니라 shared memory로 알려진 캐시에도 접근할 수 있게 했다.
처음엔 그래픽 문제로 위장하여 셰이딩 언어로 코드를 작성해야 했지만, 지금은 C언어에 특수한 기능을 추가함.
유체 역학, 의료 영상 분야, 3D 초음파 데이터 처리에서 막대한 계산이 필요한 작업에 널리 활용된다.

## reference
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
https://developer.nvidia.com/blog/even-easier-introduction-cuda/