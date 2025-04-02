#ifndef GRAYSCALE_FILTER_H
#define GRAYSCALE_FILTER_H

#include "../headers.h"

__global__ void grayscale(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x < width && y < height) {
        int idx = (y * width + x) * channels;
        
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];

        output[y * width + x] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
    
}

unsigned char* gray(unsigned char* input, int width, int height, int channels) {
    unsigned char *d_input, *output, *d_output;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width+BLOCK_SIZE-1)/BLOCK_SIZE, (height+BLOCK_SIZE-1)/BLOCK_SIZE);

    output = (unsigned char*)malloc(width*height);
    cudaMalloc(&d_input, width*height*channels);
    cudaMalloc(&d_output, width*height);

    cudaMemcpy(d_input, input, width*height*channels, cudaMemcpyHostToDevice);
    grayscale<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels);
    cudaMemcpy(output, d_output, width*height, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

#endif