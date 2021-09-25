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

#define TILE_SIZE 4 

__global__ void convolution_media_gpu(unsigned char* d_img, int img_h, int img_w, int win_size, unsigned char* d_img_res)
{
    // Row and colum for the thread
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    int padd = win_size / 2;
    if ((py <  padd) || (px < padd) || (py >= img_h - padd) || (px >= img_w - padd))
        d_img_res[py * img_w + px] = 0; // Assigning 0 for the border
    else {
        for (int w_i = 0; w_i < win_size; w_i++) {
            for (int w_j = 0; w_j < win_size; w_j++) {
                sum += d_img[(py + w_i - 1) * img_w + (px + w_j - 1)];
            }
        }
        d_img_res[py * img_w + px] = sum / (win_size * win_size);
    }
}

void apply_media_convolution(unsigned char* img_data, int img_h, int img_w, int n_channels, int win_size, unsigned char* img_res)
{
    // declare GPU memory pointers
    unsigned char* d_img = NULL;
    unsigned char* d_img_res = NULL;
 
    // allocate GPU memory
    cudaMalloc((void**)&d_img, img_h * img_w * n_channels);
    cudaMalloc((void**)&d_img_res, img_h * img_w * n_channels);
    
    // transfer the arrays to the GPU
    cudaMemcpy(d_img, img_data, img_h * img_w * n_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_res, img_res, img_h * img_w * n_channels, cudaMemcpyHostToDevice);
    
    //launch the kernel
    dim3 dim_block(TILE_SIZE, TILE_SIZE);
    dim3 grid_img((int)ceil(img_w / TILE_SIZE), (int)ceil(img_h / TILE_SIZE));
    convolution_media_gpu << < grid_img, dim_block >> > (d_img, img_h, img_w, win_size, d_img_res);
        
    // copy back from GPU
    cudaMemcpy(img_res, d_img_res, img_h * img_w * n_channels, cudaMemcpyDeviceToHost);

    // free GPU memory allocation
    cudaFree(d_img_res);
    cudaFree(d_img);
}

__global__ void convolution_sobel_gpu(unsigned char* d_img, int img_h, int img_w, unsigned char* d_img_res)
{
    // Row and colum for the thread
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int padd = 1;
    const int win_size = 3;
    int filter_x[9] = { -1,0,1,-2,0,2,-1,0,1 };
    int filter_y[9] = { 1,2,1,0,0,0,-1,-2,-1 };
    if ((py < padd) || (px < padd) || (py >= img_h - padd) || (px >= img_w - padd))
        d_img_res[px + py * img_w] = 0; // Assigning 0 for the border
    else {
        int k = 0;
        float Gx = 0, Gy = 0;
        for (int w_i = 0; w_i < win_size; w_i++) {
            for (int w_j = 0; w_j < win_size; w_j++) {
                Gx += filter_x[k] * d_img[(py + w_i - 1) * img_w + (px + w_j - 1)];
                Gy += filter_y[k] * d_img[(py + w_i - 1) * img_w + (px + w_j - 1)];
                k++;
            }
        }
        d_img_res[px + py * img_w] = sqrt((Gx * Gx) + (Gy * Gy));
    }
}

void apply_sobel_convolution(unsigned char* img_data, int img_h, int img_w, int n_channels, unsigned char* img_res)
{
    // declare GPU memory pointers
    unsigned char* d_img = NULL;
    unsigned char* d_img_res = NULL;

    // allocate GPU memory
    cudaMalloc((void**)&d_img, img_h * img_w * n_channels);
    cudaMalloc((void**)&d_img_res, img_h * img_w * n_channels);

    // transfer the arrays to the GPU
    cudaMemcpy(d_img, img_data, img_h * img_w * n_channels, cudaMemcpyHostToDevice);

    //launch the kernel
    dim3 dim_block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid_img((int)ceil(img_w / TILE_SIZE), (int)ceil(img_h / TILE_SIZE));
    convolution_sobel_gpu << < grid_img, dim_block >> > (d_img, img_h, img_w, d_img_res);

    // copy back from GPU
    cudaMemcpy(img_res, d_img_res, img_h * img_w * n_channels, cudaMemcpyDeviceToHost);
    
    // free GPU memory allocation
    cudaFree(d_img_res);
    cudaFree(d_img);
}

__device__ unsigned char compute_bilinear_val(unsigned char* d_img, int pos)
{

}

__constant__ unsigned char* img_base;

//__global__ void bilinear_interpolation_gpu(unsigned char* d_img, int img_h, int img_w, int n_channels, int zoom, unsigned char* d_img_res)
__global__ void bilinear_interpolation_gpu(unsigned char* d_img, int img_h, int img_w, int n_channels, int zoom, unsigned char* d_img_res)
{
    // Row and colum for the thread
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    //int px = blockIdx.x;
    //int py = blockIdx.y;
    int pos = (px + py * gridDim.x);

    int sum = 0;
    int padd = 1;
    int new_img_h = img_h * zoom;
    int new_img_w = img_w * zoom;
    //if (!((py < padd) || (px < padd) || (py >= new_img_h - padd) || (px >= new_img_w - padd)))
    //{
        int i = px / zoom;
        int d = i + 1;
        int s = py / zoom;
        int r = s + 1;
        int a = px - i;
        int b = py - s;
        //int pos = (py * new_img_w);// +px);
        //d_img_res[py * img_w * zoom + px] = (1 - a) * (1 - b) * d_img[s * img_w + i] + 
        //    a * (1 - b) * d_img[s * img_w + d] + (1 - a) * b * d_img[r * img_w + i] + a * b * d_img[r * img_w + d];
        d_img_res[pos] = 255; // d_img[s * img_w + i]; //d_img[(int)((py / zoom) * img_w + (py / zoom))];
        //d_img_res[pos + 1] = 255; // d_img[s * img_w + i]; //d_img[(int)((py / zoom) * img_w + (py / zoom))];
        d_img_res[450 * new_img_w + px] = 255; //d_img[(int)((py / zoom) * img_w + (py / zoom))];
    //}
}

void apply_bilinear_interpolation(unsigned char* img_data, int img_h, int img_w, int n_channels, int zoom, unsigned char* img_res)
{
    // declare GPU memory pointers
    unsigned char* d_img = NULL;
    unsigned char* d_img_res = NULL;

    cout << "channels: " << n_channels << endl;

    // allocate GPU memory
    cudaMalloc((void**)&d_img, img_h * img_w * n_channels);
    cudaMalloc((void**)&d_img_res, img_h * img_w * zoom * n_channels);
    //cudaMemcpyToSymbol(img_base, &img_data, img_h * img_w);

    // transfer the arrays to the GPU
    cudaMemcpy(d_img, img_data, img_h * img_w * n_channels, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_img_res, img_res, img_h * img_w * zoom * n_channels, cudaMemcpyHostToDevice);

    //launch the kernel MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    //dim3 dim_block(TILE_SIZE, TILE_SIZE);
    dim3 grid_img(img_w * zoom, img_h * zoom);
    //dim3 grid_img((int)ceil(img_w * zoom / TILE_SIZE), (int)ceil(img_h * zoom / TILE_SIZE));
    //bilinear_interpolation_gpu <<< grid_img, dim_block >>> (d_img, img_h, img_w, n_channels, zoom, d_img_res);
    bilinear_interpolation_gpu <<< grid_img, 1 >>> (d_img, img_h, img_w, n_channels, zoom, d_img_res);

    // copy back from GPU
    cudaMemcpy(img_res, d_img_res, img_h * img_w * zoom * n_channels, cudaMemcpyDeviceToHost);

    // free GPU memory allocation
    cudaFree(d_img_res);
    cudaFree(d_img);
}