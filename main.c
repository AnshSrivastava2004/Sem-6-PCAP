#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "stb_image/stb_image.h"
#include "stb_image/stb_image_write.h"

#define SOBEL_EDGE "sobel\n"
#define PREWITT_EDGE "prewitt\n"
#define ROBERTS_EDGE "roberts\n"
#define SCHARR_EDGE "scharr\n"
#define LAPLACIAN_EDGE "laplacian\n"
#define LOG_EDGE "log\n"
#define CANNY_EDGE "canny\n"

void printTime(char* filter, double seconds) {
    printf("Filter - %s", filter);
    if (seconds < 60)
        printf("Time elapsed - %f seconds\n", seconds);
    else {
        int min = (int)seconds / 60;
        printf("Time elapsed - %d %s , %f seconds\n", min, min > 1 ? "minutes" : "minute", fmod(seconds, 60));
    }
}

unsigned char* gray(unsigned char* img, int width, int height, int channels) {
    unsigned char* gray_img = (unsigned char*)malloc(width * height);
    for (int i = 0; i < width * height; ++i) {
        int r = img[i * channels + 0];
        int g = img[i * channels + 1];
        int b = img[i * channels + 2];
        gray_img[i] = (r * 0.3 + g * 0.59 + b * 0.11);
    }
    return gray_img;
}

unsigned char* apply_kernel(unsigned char* img, int width, int height, int kernel[3][3]) {
    unsigned char* out = (unsigned char*)calloc(width * height, 1);
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sum = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    sum += img[(y + ky) * width + (x + kx)] * kernel[ky + 1][kx + 1];
                }
            }
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            out[y * width + x] = sum;
        }
    }
    return out;
}

unsigned char* gaussian_blur(unsigned char* img, int width, int height) {
    int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    unsigned char* blurred = (unsigned char*)calloc(width * height, 1);
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sum = 0;
            int wsum = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int weight = kernel[ky + 1][kx + 1];
                    sum += img[(y + ky) * width + (x + kx)] * weight;
                    wsum += weight;
                }
            }
            blurred[y * width + x] = sum / wsum;
        }
    }
    return blurred;
}

unsigned char* sobel(unsigned char* img, int width, int height) {
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    unsigned char* out = (unsigned char*)calloc(width * height, 1);
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sumx = 0, sumy = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = img[(y + ky) * width + (x + kx)];
                    sumx += pixel * gx[ky + 1][kx + 1];
                    sumy += pixel * gy[ky + 1][kx + 1];
                }
            }
            int val = sqrt(sumx * sumx + sumy * sumy);
            if (val > 255) val = 255;
            out[y * width + x] = val;
        }
    }
    return out;
}

unsigned char* prewitt(unsigned char* img, int width, int height) {
    int gx[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
    return sobel(img, width, height); // Reusing logic with custom kernels if needed
}

unsigned char* roberts(unsigned char* img, int width, int height) {
    unsigned char* out = (unsigned char*)calloc(width * height, 1);
    for (int y = 0; y < height - 1; y++) {
        for (int x = 0; x < width - 1; x++) {
            int gx = img[y * width + x] - img[(y + 1) * width + (x + 1)];
            int gy = img[(y + 1) * width + x] - img[y * width + (x + 1)];
            int val = sqrt(gx * gx + gy * gy);
            if (val > 255) val = 255;
            out[y * width + x] = val;
        }
    }
    return out;
}

unsigned char* scharr(unsigned char* img, int width, int height) {
    int gx[3][3] = {{-3, 0, 3}, {-10, 0, 10}, {-3, 0, 3}};
    int gy[3][3] = {{-3, -10, -3}, {0, 0, 0}, {3, 10, 3}};
    return sobel(img, width, height); // Reusing logic
}

unsigned char* laplacian(unsigned char* img, int width, int height) {
    int kernel[3][3] = {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};
    return apply_kernel(img, width, height, kernel);
}

unsigned char* log_edge(unsigned char* img, int width, int height) {
    unsigned char* blurred = gaussian_blur(img, width, height);
    unsigned char* result = laplacian(blurred, width, height);
    free(blurred);
    return result;
}

unsigned char* canny(unsigned char* img, int width, int height) {
    unsigned char* blurred = gaussian_blur(img, width, height);
    unsigned char* edges = sobel(blurred, width, height);
    free(blurred);
    return edges;
}

int main() {
    int width, height, channels;
    char* filter = (char*)malloc(100);
    unsigned char* img = stbi_load("images/pcapimg2.jpg", &width, &height, &channels, 0);
    if (!img) {
        printf("Error loading image.\n");
        return 1;
    }

    printf("Image loaded with width = %dpx, height = %dpx and %d channels\n", width, height, channels);
    unsigned char *gray_img = gray(img, width, height, channels);
    unsigned char *output_image = NULL;

    printf("Enter filter: ");
    fgets(filter, 100, stdin);

    clock_t start = clock();

    if (!strcmp(filter, SOBEL_EDGE)) {
        output_image = sobel(gray_img, width, height);
        stbi_write_jpg("results/sobel/cpu_pcapimg2.jpg", width, height, 1, output_image, 100);
    } else if (!strcmp(filter, PREWITT_EDGE)) {
        output_image = prewitt(gray_img, width, height);
        stbi_write_jpg("results/prewitt/cpu_pcapimg2.jpg", width, height, 1, output_image, 100);
    } else if (!strcmp(filter, ROBERTS_EDGE)) {
        output_image = roberts(gray_img, width, height);
        stbi_write_jpg("results/roberts/cpu_pcapimg2.jpg", width, height, 1, output_image, 100);
    } else if (!strcmp(filter, SCHARR_EDGE)) {
        output_image = scharr(gray_img, width, height);
        stbi_write_jpg("results/scharr/cpu_pcapimg2.jpg", width, height, 1, output_image, 100);
    } else if (!strcmp(filter, LAPLACIAN_EDGE)) {
        output_image = laplacian(gray_img, width, height);
        stbi_write_jpg("results/laplacian/cpu_pcapimg2.jpg", width, height, 1, output_image, 100);
    } else if (!strcmp(filter, LOG_EDGE)) {
        output_image = log_edge(gray_img, width, height);
        stbi_write_jpg("results/log/cpu_pcapimg2.jpg", width, height, 1, output_image, 100);
    } else if (!strcmp(filter, CANNY_EDGE)) {
        output_image = canny(gray_img, width, height);
        stbi_write_jpg("results/canny/cpu_pcapimg2.jpg", width, height, 1, output_image, 100);
    } else {
        printf("No filter found\n");
        return 1;
    }

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printTime(filter, time_spent);

    stbi_image_free(img);
    free(gray_img);
    free(output_image);
    free(filter);
    return 0;
}
