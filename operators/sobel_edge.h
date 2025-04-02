#ifndef SOBEL_EDGE_H
#define SOBEL_EDGE_H

#include "../headers.h"

__global__ void sobelOperator(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int Gx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        int Gy[3][3] = {
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1}
        };
        int sumX = 0, sumY = 0;
        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                int px = input[(y + i) * width + (x + j)];
                sumX += Gx[i + 1][j + 1] * px;
                sumY += Gy[i + 1][j + 1] * px; 
            }
        }
        int magnitude = (int)(255.0 * sqrtf(sumX * sumX + sumY * sumY) / 1024.0);
        output[y*width + x] = magnitude;
    }
}

unsigned char* sobel(unsigned char* input, int width, int height) {
    unsigned char *d_input, *output, *d_output;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t size = width * height;  

    output = (unsigned char*)malloc(size);
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    sobelOperator<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

#endif
