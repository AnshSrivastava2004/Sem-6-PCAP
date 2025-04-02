#include "headers.h"
#include "operators/grayscale.h"
#include "operators/sobel_edge.h"
#include "operators/prewitt_edge.h"
#include "operators/roberts_edge.h"
#include "operators/scharr_edge.h"
#include "operators/laplacian_edge.h"
#include "operators/log_edge.h"
#include "operators/canny_edge.h"

const char* SOBEL_EDGE = "sobel\n";
const char* PREWITT_EDGE = "prewitt\n";
const char* ROBERTS_EDGE = "roberts\n";
const char* SCHARR_EDGE = "scharr\n";
const char* LAPLACIAN_EDGE = "laplacian\n";
const char* LOG_EDGE = "log\n";
const char* CANNY_EDGE = "canny\n";

void printTime(char* filter, float milliseconds) {
    float seconds = milliseconds/1000.0;
    printf("Filter - %s", filter);
    if(seconds < 60) {
        printf("Time elapsed - %f seconds\n", seconds);
    } else {
        int min = (int)seconds/60;
        printf("Time elapsed - %d %s , %f seconds\n", min, min > 1 ? "minutes" : "minute", fmod(seconds, 60));
    } 
}

int main() {
    int width, height, channels;
    float milliseconds;
    char* filter = (char*)malloc(100);
    unsigned char* img = stbi_load("images/pcapimg2.jpg", &width, &height, &channels, 0);
    if(!img) {
        printf("Error loading image.\n");
        exit(0);
    }
    printf("Image loaded with width = %dpx, height = %dpx and %d channels\n", width, height, channels);
    unsigned char *gray_img, *output_image;
    gray_img = gray(img, width, height, channels);

    printf("Enter filter: ");
    fgets(filter, 100, stdin);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if(!strcmp(filter, SOBEL_EDGE)) {
        output_image = sobel(gray_img, width, height);
        stbi_write_jpg("results/sobel/pcapimg2.jpg", width, height, 1, output_image, 100);
        printf("Sobel image added\n");
    } else if(!strcmp(filter, PREWITT_EDGE)) {
        output_image = prewitt(gray_img, width, height);
        stbi_write_jpg("results/prewitt/pcapimg2.jpg", width, height, 1, output_image, 100);
        printf("Prewitt image added\n");
    } else if(!strcmp(filter, ROBERTS_EDGE)) {
        output_image = roberts(gray_img, width, height);
        stbi_write_jpg("results/roberts/pcapimg2.jpg", width, height, 1, output_image, 100);
        printf("Roberts image added\n");
    } else if(!strcmp(filter, SCHARR_EDGE)) {
        output_image = scharr(gray_img, width, height);
        stbi_write_jpg("results/scharr/pcapimg2.jpg", width, height, 1, output_image, 100);
        printf("Scharr image added\n");
    } else if(!strcmp(filter, LAPLACIAN_EDGE)) {
        output_image = laplacian(gray_img, width, height);
        stbi_write_jpg("results/laplacian/pcapimg2.jpg", width, height, 1, output_image, 100);
        printf("Laplacian image added\n");
    } else if(!strcmp(filter, LOG_EDGE)) {
        output_image = log(gray_img, width, height);
        stbi_write_jpg("results/log/pcapimg2.jpg", width, height, 1, output_image, 100);
        printf("LoG image added\n");
    } else if(!strcmp(filter, CANNY_EDGE)) {
        output_image = canny(gray_img, width, height);
        stbi_write_jpg("results/canny/pcapimg2.jpg", width, height, 1, output_image, 100);
        printf("Canny image added\n");
    } else {
        printf("No filter found\n");
        exit(1);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printTime(filter, milliseconds);

    stbi_image_free(img);

    return 0;
}