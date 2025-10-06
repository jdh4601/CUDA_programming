#include <stdio.h>
#include <curand.h>

int main() {
    cudaSetDevice(4); // gpu 지정하기
    const int size = 20;
    // create number generator
    curandGenerator_t generator;
    curandStatus_t status = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    if (status != CURAND_STATUS_SUCCESS) {
        printf("failed to create generator\n");
        return 1;
    }

    // set the seed
    status = curandSetPseudoRandomGeneratorSeed(generator, 1234);
    if (status != CURAND_STATUS_SUCCESS) {
        printf("Failed to set generator seed\n");
        curandDestroyGenerator(generator);
        return 1;
    }

    // allocate memory on the host
    unsigned int* hostArray = new unsigned int[size];

    status = curandGenerate(generator, hostArray, size);
    if (status != CURAND_STATUS_SUCCESS) {
        printf("failed to generate random num");
        delete[] hostArray;
        curandDestroyGenerator(generator);
        return 1;
    }

    // print random number
    for (int i = 0; i < size; i++) {
        printf("%u\n", hostArray[i]);
    }
    
    // clean up
    delete[] hostArray;
    curandDestroyGenerator(generator);

    return 0;
}