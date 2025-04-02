#ifndef ROBERTS_EDGE_H
#define ROBERTS_EDGE_H

#include "../headers.h"

__global__ void robertsOperator(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x < width - 1 && y < height - 1) {
        int Gx[2][2] = {
            {1, 0},
            {0, -1}
        };
        int Gy[2][2] = {
            {0, 1},
            {-1, 0}
        };
        int sumX = 0, sumY = 0;
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
                int px = input[(y + i) * width + (x + j)];
                sumX += Gx[i][j] * px;
                sumY += Gy[i][j] * px; 
            }
        }
        int magnitude = min(255, (int)(sqrtf(sumX * sumX + sumY * sumY)));
        output[y*width + x] = magnitude;
    }
}

unsigned char* roberts(unsigned char* input, int width, int height) {
    unsigned char *d_input, *output, *d_output;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t size = width * height;  

    output = (unsigned char*)malloc(size);
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    robertsOperator<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

#endif
