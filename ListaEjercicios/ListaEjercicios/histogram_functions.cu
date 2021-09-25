#include "histogram_functions.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;


#define NUM_BIN 256

__managed__ int SIZE;
__managed__ int IMG_H;
__managed__ int IMG_W;


__global__ void histogram_atomic_1d(int* d_hist, unsigned char* d_img)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int item = d_img[tid];
    if (tid < SIZE)
    {
        atomicAdd(&(d_hist[item]), 1);
    }
}

void compute_histogram_1d(unsigned char* img_data, int img_h, int img_w, int *histogram)
{
    SIZE = img_h * img_w;
    //static int histogram[NUM_BIN] = { 0 };

    // declare GPU memory pointers
    unsigned char* d_img;
    int* d_hist;

    // allocate GPU memory
    cudaMalloc((void**)&d_img, SIZE);
    cudaMalloc((void**)&d_hist, NUM_BIN * sizeof(int));

    // transfer the arrays to the GPU
    cudaMemcpy(d_img, img_data, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist, histogram, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice);

    //launch the kernel
    histogram_atomic_1d <<< ((SIZE + NUM_BIN - 1) / NUM_BIN), NUM_BIN >>> (d_hist, d_img);

    // copy back the sum from GPU
    cudaMemcpy(histogram, d_hist, NUM_BIN * sizeof(int), cudaMemcpyDeviceToHost);

    // free GPU memory allocation
    cudaFree(d_img);
    cudaFree(d_hist);

    //return histogram;
}

__global__ void histogram_atomic_3d(int* d_hist, unsigned char* d_img_r, unsigned char* d_img_g, unsigned char* d_img_b)
{
    const int NUM_BINS_3D = 32;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int item_r = d_img_r[tid] / NUM_BINS_3D;
    int item_g = d_img_g[tid] / NUM_BINS_3D;
    int item_b = d_img_b[tid] / NUM_BINS_3D;
    int pos;
    if (tid < SIZE)
    {
        //pos = (item_b * IMG_H * IMG_W) + (item_g * IMG_H) + item_r;
        pos = (item_b * 8 * 3) + (item_g * 8) + item_r;
        atomicAdd(&(d_hist[pos]), 1);
    }
}

int* compute_histogram_3d(unsigned char* img_data_r, unsigned char* img_data_g, unsigned char* img_data_b, int img_h, int img_w)
{
    const int NUM_BIN_3D = 512; // 8 * 8 * 8
    SIZE = img_h * img_w;
    IMG_H = img_h;
    IMG_W = img_w;

    static int histogram[NUM_BIN_3D] = { 0 };

    // declare GPU memory pointers
    unsigned char* d_img_r, *d_img_g, *d_img_b;
    int* d_hist;

    // allocate GPU memory
    cudaMalloc((void**)&d_img_r, SIZE);
    cudaMalloc((void**)&d_img_g, SIZE);
    cudaMalloc((void**)&d_img_b, SIZE);
    cudaMalloc((void**)&d_hist, NUM_BIN_3D * sizeof(int));

    // transfer the arrays to the GPU
    cudaMemcpy(d_img_r, img_data_r, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_g, img_data_g, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_b, img_data_b, SIZE, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_hist, histogram, NUM_BIN_3D * sizeof(int), cudaMemcpyHostToDevice);

    //launch the kernel
    histogram_atomic_3d << <((SIZE + NUM_BIN_3D - 1) / NUM_BIN_3D), NUM_BIN_3D >> > (d_hist, d_img_r, d_img_g, d_img_b);

    // copy back the sum from GPU
    cudaMemcpy(histogram, d_hist, NUM_BIN_3D * sizeof(int), cudaMemcpyDeviceToHost);

    // free GPU memory allocation
    cudaFree(d_img_r);
    cudaFree(d_img_g);
    cudaFree(d_img_b);
    cudaFree(d_hist);

    return histogram;
}

/***************** STRETCHING **********************/
void get_hist_min_max(int* hist, int& m, int& M) {
    m = -1;
    M = -1;
    const int NUM_BINS = 256;
    for (int i = 0; i < NUM_BINS; i++) {
        if (hist[i] > 0) {
            m = i;
            break;
        }
    }
    for (int i = NUM_BINS - 1; i >= 0; i--) {
        if (hist[i] > 0) {
            M = i;
            break;
        }
    }
}

__global__ void stretching_gpu(unsigned char* d_img, unsigned char* d_img_eq, int HIST_m, int HIST_M)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int pos = x + y * gridDim.x;
    //if (pos < SIZE)
    d_img_eq[pos] = (d_img[pos] - HIST_m) * 255 / (HIST_M - HIST_m);
}

void stretch(unsigned char* img_data, int img_h, int img_w, int n_channels, unsigned char *img_eq)
{
    int m, M;
    int hist[NUM_BIN] = { 0 };
    compute_histogram_1d(img_data, img_h, img_w, hist);
    get_hist_min_max(hist, m, M);

    for (int i = 0; i < NUM_BIN; i++) {
        printf("bin %d : count %d \n", i, hist[i]);
    }    
    cout << "min = " << m << " --> " << M << endl;

    SIZE = img_h * img_w;
    
    // declare GPU memory pointers
    unsigned char* d_img = NULL;
    unsigned char* d_img_eq = NULL;

    // allocate GPU memory
    cudaMalloc((void**)&d_img, img_h * img_w * n_channels);
    cudaMalloc((void**)&d_img_eq, img_h * img_w * n_channels);

    // transfer the arrays to the GPU
    cudaMemcpy(d_img, img_data, img_h * img_w * n_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_eq, img_data, img_h * img_w * n_channels, cudaMemcpyHostToDevice);

    //launch the kernel
    dim3 grid_img(img_w, img_h);
    stretching_gpu <<< grid_img, 1 >>> (d_img, d_img_eq, m, M);

    // copy back from GPU
    cudaMemcpy(img_eq, d_img_eq, img_h * img_w * n_channels, cudaMemcpyDeviceToHost);

    // free GPU memory allocation
    cudaFree(d_img_eq);
    cudaFree(d_img);
}

/***************** EQUALIZATION **********************/
__global__ void equalization_gpu(unsigned char* d_img, unsigned char* d_img_eq, int *d_func)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int pos = x + y * gridDim.x;
    d_img_eq[pos] = (unsigned char)d_func[d_img[pos]];
}

void compute_equalization_func(int* histogram, int NP, int* hist_f)
{
    int acum = 0;
    for (int i = 0; i < NUM_BIN; i++)
    {
        acum += histogram[i];
        hist_f[i] = (acum * 1.0 / NP) * 255;
        cout << acum << " / " << NP << " = " << hist_f[i] << endl;
    }
}

void equalize(unsigned char* img_data, int img_h, int img_w, int n_channels, unsigned char* img_res)
{
    int hist[NUM_BIN] = { 0 };
    int hist_f[NUM_BIN] = { 0 };
    compute_histogram_1d(img_data, img_h, img_w, hist);
    compute_equalization_func(hist, img_h * img_w, hist_f);

    for (int i = 0; i < NUM_BIN; i++) {
        printf("bin %d : hist %d : f %d \n", i, hist[i], hist_f[i]);
    }
    
    // declare GPU memory pointers
    unsigned char* d_img = NULL;
    unsigned char* d_img_res = NULL;
    int* d_func;

    // allocate GPU memory
    cudaMalloc((void**)&d_img, img_h * img_w * n_channels);
    cudaMalloc((void**)&d_img_res, img_h * img_w * n_channels);
    cudaMalloc((void**)&d_func, NUM_BIN * sizeof(int));

    // transfer the arrays to the GPU
    cudaMemcpy(d_img, img_data, img_h * img_w * n_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_res, img_data, img_h * img_w * n_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_func, hist_f, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice);

    //launch the kernel
    dim3 grid_img(img_w, img_h);
    equalization_gpu << < grid_img, 1 >> > (d_img, d_img_res, d_func);

    // copy back from GPU
    cudaMemcpy(img_res, d_img_res, img_h * img_w * n_channels, cudaMemcpyDeviceToHost);

    // free GPU memory allocation
    cudaFree(d_img_res);
    cudaFree(d_img);
}
