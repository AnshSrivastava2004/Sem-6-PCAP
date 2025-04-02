#ifndef LOG_EDGE_H
#define LOG_EDGE_H

#include "../headers.h"

__global__ void logOperator(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
        int L[5][5] = {
            {0, 0, -1, 0, 0},
            {0, -1, -2, -1, 0},
            {-1, -2, 16, -2, -1},
            {0, -1, -2, -1, 0},
            {0, 0, -1, 0, 0},
        };
        int sum = 0;
        for(int i = -2; i <= 2; i++) {
            for(int j = -2; j <= 2; j++) {
                int px = input[(y + i) * width + (x + j)];
                sum += L[i + 1][j + 1] * px;
            }
        }
        int magnitude = min(255, max(0, (int)(255.0 * sum / 1024.0) + 128));
        output[y*width + x] = magnitude;
    }
}

unsigned char* log(unsigned char* input, int width, int height) {
    unsigned char *d_input, *output, *d_output;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t size = width * height;  

    output = (unsigned char*)malloc(size);
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    logOperator<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

#endif
