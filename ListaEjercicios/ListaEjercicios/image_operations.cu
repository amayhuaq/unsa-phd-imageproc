#include "image_operations.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;


__device__ unsigned char verify_pixel_value(int pix_val)
{
    unsigned char res;
    if (pix_val < 0)
        res = (unsigned char)0;
    else if (pix_val > 255)
        res = (unsigned char)255;
    else
        res = (unsigned char)pix_val;
    return res;
}

__global__ void apply_linfunc_gpu(unsigned char* d_img, unsigned char* d_img_res, int n_channels, double A, double B)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int pos = (x + y * gridDim.x) * n_channels;
    d_img_res[pos] = verify_pixel_value(A * d_img[pos] + B);
    d_img_res[pos + 1] = verify_pixel_value(A * d_img[pos + 1] + B);
    d_img_res[pos + 2] = verify_pixel_value(A * d_img[pos + 2] + B);
}

void apply_function(unsigned char* img_data, int img_h, int img_w, int n_channels, double A, double B, unsigned char* img_res)
{
    cout << "Function: F(X) = " << A << "X + " << B << endl;

    // declare GPU memory pointers
    unsigned char* d_img = NULL;
    unsigned char* d_img_res = NULL;

    // allocate GPU memory
    cudaMalloc((void**)&d_img, img_h * img_w * n_channels);
    cudaMalloc((void**)&d_img_res, img_h * img_w * n_channels);

    // transfer the arrays to the GPU
    cudaMemcpy(d_img, img_data, img_h * img_w * n_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_res, img_data, img_h * img_w * n_channels, cudaMemcpyHostToDevice);

    //launch the kernel
    dim3 grid_img(img_w, img_h);
    apply_linfunc_gpu << < grid_img, 1 >> > (d_img, d_img_res, n_channels, A, B);

    // copy back from GPU
    cudaMemcpy(img_res, d_img_res, img_h * img_w * n_channels, cudaMemcpyDeviceToHost);

    // free GPU memory allocation
    cudaFree(d_img_res);
    cudaFree(d_img);
}

/****************** ARITMETHIC OPERATIONS *********************/
__global__ void apply_sum_gpu(unsigned char* d_img1, unsigned char* d_img2, unsigned char* d_img_res, int n_channels)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int pos = (x + y * gridDim.x) * n_channels;
    d_img_res[pos] = (d_img1[pos] + d_img2[pos]) / 2;
    d_img_res[pos + 1] = (d_img1[pos + 1] + d_img2[pos + 1]) / 2;
    d_img_res[pos + 2] = (d_img1[pos + 2] + d_img2[pos + 2]) / 2;
}

__global__ void apply_subs_gpu(unsigned char* d_img1, unsigned char* d_img2, unsigned char* d_img_res, int n_channels)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int pos = (x + y * gridDim.x) * n_channels;
    d_img_res[pos] = abs(d_img1[pos] - d_img2[pos]);
    d_img_res[pos + 1] = abs(d_img1[pos + 1] - d_img2[pos + 1]);
    d_img_res[pos + 2] = abs(d_img1[pos + 2] - d_img2[pos + 2]);
}

__global__ void apply_mult_gpu(unsigned char* d_img1, unsigned char* d_img2, unsigned char* d_img_res, int n_channels)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int pos = (x + y * gridDim.x) * n_channels;
    int val;
    val = (d_img1[pos] * d_img2[pos]) / 255;
    d_img_res[pos] = (unsigned char)val;
    val = (d_img1[pos + 1] * d_img2[pos + 1]) / 255;
    d_img_res[pos + 1] = (unsigned char)val;
    val = (d_img1[pos + 2] * d_img2[pos + 2]) / 255;
    d_img_res[pos + 2] = (unsigned char)val;
}

__device__ unsigned char divide_pixel_value(int pix1, int pix2)
{
    return (pix2 == 0) ? 0 : verify_pixel_value((pix1 * 1.0 / pix2) * 255);
}

__global__ void apply_div_gpu(unsigned char* d_img1, unsigned char* d_img2, unsigned char* d_img_res, int n_channels)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int pos = (x + y * gridDim.x) * n_channels;
    d_img_res[pos] = divide_pixel_value(d_img1[pos], d_img2[pos]);
    d_img_res[pos + 1] = divide_pixel_value(d_img1[pos + 1], d_img2[pos + 1]);
    d_img_res[pos + 2] = divide_pixel_value(d_img1[pos + 2], d_img2[pos + 2]);
}

void apply_aritmethic_operation(unsigned char* img1, unsigned char* img2, int img_h, int img_w, int n_channels, unsigned char* img_res, int operation)
{
    // declare GPU memory pointers
    unsigned char* d_img1 = NULL;
    unsigned char* d_img2 = NULL;
    unsigned char* d_img_res = NULL;

    // allocate GPU memory
    cudaMalloc((void**)&d_img1, img_h * img_w * n_channels);
    cudaMalloc((void**)&d_img2, img_h * img_w * n_channels);
    cudaMalloc((void**)&d_img_res, img_h * img_w * n_channels);

    // transfer the arrays to the GPU
    cudaMemcpy(d_img1, img1, img_h * img_w * n_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, img2, img_h * img_w * n_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_res, img1, img_h * img_w * n_channels, cudaMemcpyHostToDevice);

    //launch the kernel
    dim3 grid_img(img_w, img_h);
    switch (operation)
    {
        case 1: // sum
            apply_sum_gpu << < grid_img, 1 >> > (d_img1, d_img2, d_img_res, n_channels);
            break;
        case 2: // subs
            apply_subs_gpu << < grid_img, 1 >> > (d_img1, d_img2, d_img_res, n_channels);
            break;
        case 3: // mult
            apply_mult_gpu << < grid_img, 1 >> > (d_img1, d_img2, d_img_res, n_channels);
            break;
        case 4: // div
            apply_div_gpu << < grid_img, 1 >> > (d_img1, d_img2, d_img_res, n_channels);
            break;
    }

    // copy back from GPU
    cudaMemcpy(img_res, d_img_res, img_h * img_w * n_channels, cudaMemcpyDeviceToHost);

    // free GPU memory allocation
    cudaFree(d_img_res);
    cudaFree(d_img1);
    cudaFree(d_img2);
}

/**************** CONVOLUTION TO APPLY MEDIA AND SOBEL FILTERS **********************/
__global__ void convolution_gpu(unsigned char* d_img, double* d_filter, unsigned char* d_img_res, int n_channels)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int pos = (x + y * gridDim.x) * n_channels;
    /*int val = (255.0 * d_img1[pos]) / d_img2[pos];
    d_img_res[pos] = (unsigned char)val;
    val = (255.0 * d_img1[pos + 1]) / d_img2[pos + 1];
    d_img_res[pos + 1] = (unsigned char)val;
    val = (255.0 * d_img1[pos + 2]) / d_img2[pos + 2];
    d_img_res[pos + 2] = (unsigned char)val;
    */
}

void apply_convolution(unsigned char* img_data, int img_h, int img_w, int n_channels, int win_size, unsigned char* img_media, unsigned char* img_sobel)
{
    // create filter based on the win_size
    double* h_filter_med, * h_filter_sob;
    
    // declare GPU memory pointers
    unsigned char* d_img = NULL;
    unsigned char* d_med = NULL;
    unsigned char* d_sob = NULL;
    double* d_filter_med = NULL;
    double* d_filter_sob = NULL;

    // allocate GPU memory
    cudaMalloc((void**)&d_img, img_h * img_w * n_channels);
    cudaMalloc((void**)&d_med, img_h * img_w * n_channels);
    cudaMalloc((void**)&d_sob, img_h * img_w * n_channels);
    cudaMalloc((void**)&d_filter_med, win_size * win_size);
    cudaMalloc((void**)&d_filter_sob, win_size * win_size);

    // transfer the arrays to the GPU
    cudaMemcpy(d_img, img_data, img_h * img_w * n_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_med, img_media, img_h * img_w * n_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sob, img_sobel, img_h * img_w * n_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter_med, h_filter_med, win_size * win_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter_sob, h_filter_sob, win_size * win_size, cudaMemcpyHostToDevice);

    //launch the kernel
    dim3 grid_img(img_w, img_h);
    convolution_gpu << < grid_img, 1 >> > (d_img, d_filter_med, d_med, n_channels);
    convolution_gpu << < grid_img, 1 >> > (d_img, d_filter_sob, d_sob, n_channels);
    
    // copy back from GPU
    cudaMemcpy(img_sobel, d_sob, img_h * img_w * n_channels, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_media, d_med, img_h * img_w * n_channels, cudaMemcpyDeviceToHost);

    // free GPU memory allocation
    cudaFree(d_filter_med);
    cudaFree(d_filter_sob);
    cudaFree(d_sob);
    cudaFree(d_med);
    cudaFree(d_img);
}