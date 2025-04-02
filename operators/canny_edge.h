#ifndef CANNY_EDGE_H
#define CANNY_EDGE_H

#include "../headers.h"

__global__ void gaussianBlur(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int gaussianKernel[5][5] = {
        {1, 4, 7, 4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1, 4, 7, 4, 1}
    };

    if(x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
        int sum = 0;
        for(int i = -2; i <= 2; i++) {
            for(int j = -2; j <= 2; j++) {
                int pixel = input[(y + i) * width + (x + j)];
                sum += gaussianKernel[i + 2][j + 2] * pixel;
            }
        }

        output[y * width + x] = (unsigned char)(sum / 273.0f); 
    }
}

__global__ void sobel(unsigned char* input, float* gradient, float* direction, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int Gx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        int Gy[3][3] = {
            {-1, -2, -1},
            {0,  0,  0},
            {1,  2,  1}
        };

        int sumX = 0, sumY = 0;

        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                int pixel = input[(y + i) * width + (x + j)];
                sumX += Gx[i + 1][j + 1] * pixel;
                sumY += Gy[i + 1][j + 1] * pixel;
            }
        }

        gradient[y * width + x] = sqrtf(sumX * sumX + sumY * sumY);
        direction[y * width + x] = atan2f(sumY, sumX);
    }
}

__global__ void nonMaximumSuppression(float* gradient, float* direction, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float angle = direction[y * width + x] * 180.0f / M_PI;
        if (angle < 0) angle += 180;

        int pixel1, pixel2;
        if((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
            pixel1 = gradient[y * width + (x - 1)];
            pixel2 = gradient[y * width + (x + 1)];
        } else if(angle >= 22.5 && angle < 67.5) {
            pixel1 = gradient[(y - 1) * width + (x - 1)];
            pixel2 = gradient[(y + 1) * width + (x + 1)];
        } else if(angle >= 67.5 && angle < 112.5) {
            pixel1 = gradient[(y - 1) * width + x];
            pixel2 = gradient[(y + 1) * width + x];
        } else { 
            pixel1 = gradient[(y + 1) * width + (x - 1)];
            pixel2 = gradient[(y - 1) * width + (x + 1)];
        }

        if (gradient[y * width + x] >= pixel1 && gradient[y * width + x] >= pixel2) {
            output[y * width + x] = (unsigned char)gradient[y * width + x];
        } else {
            output[y * width + x] = 0;
        }
    }
}

__global__ void hysteresisThresholding(unsigned char* input, unsigned char* output, int width, int height, float lowThreshold, float highThreshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height) {
        if(input[y * width + x] >= highThreshold) {
            output[y * width + x] = 255;
        } else if(input[y * width + x] >= lowThreshold) {
            if(x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                if(input[(y - 1) * width + (x - 1)] >= highThreshold || 
                    input[(y - 1) * width + x] >= highThreshold ||
                    input[(y - 1) * width + (x + 1)] >= highThreshold ||
                    input[y * width + (x - 1)] >= highThreshold || 
                    input[y * width + (x + 1)] >= highThreshold ||
                    input[(y + 1) * width + (x - 1)] >= highThreshold ||
                    input[(y + 1) * width + x] >= highThreshold ||
                    input[(y + 1) * width + (x + 1)] >= highThreshold) {
                    output[y * width + x] = 255; 
                } else {
                    output[y * width + x] = 0; 
                }
            }
        } else {
            output[y * width + x] = 0; 
        }
    }
}

unsigned char* canny(unsigned char* input, int width, int height) {
    unsigned char *d_input, *output, *d_smoothed, *d_nmsOutput, *d_finalOutput;
    float *d_gradient, *d_direction;
    size_t size = width * height;

    output = (unsigned char*)malloc(size);
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_smoothed, size);
    cudaMalloc(&d_gradient, size * sizeof(float));
    cudaMalloc(&d_direction, size * sizeof(float));
    cudaMalloc(&d_nmsOutput, size);
    cudaMalloc(&d_finalOutput, size);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    gaussianBlur<<<gridDim, blockDim>>>(d_input, d_smoothed, width, height);
    cudaDeviceSynchronize();

    sobel<<<gridDim, blockDim>>>(d_smoothed, d_gradient, d_direction, width, height);
    cudaDeviceSynchronize();

    nonMaximumSuppression<<<gridDim, blockDim>>>(d_gradient, d_direction, d_nmsOutput, width, height);
    cudaDeviceSynchronize();

    int lowThreshold = 25;
    int highThreshold = 100;
    hysteresisThresholding<<<gridDim, blockDim>>>(d_nmsOutput, d_finalOutput, width, height, lowThreshold, highThreshold);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_finalOutput, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_smoothed);
    cudaFree(d_gradient);
    cudaFree(d_direction);
    cudaFree(d_nmsOutput);
    cudaFree(d_finalOutput);

    return output;
}

#endif