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

/**************** BILINEAR INTERPOLATION **********************/
texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> texRef;

__global__ void bilinear_interpolation_gpu(int img_h, int img_w, int zoom, unsigned char* d_img_res)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float f01 = tex2D(texRef, col, row);
    float f11 = tex2D(texRef, col + 1, row);
    float f00 = tex2D(texRef, col, row + 1);
    float f10 = tex2D(texRef, col + 1, row + 1);

    if (col < img_w && row < img_h) {
        float a, b;
        for (int i = 0; i < zoom; i++) {
            for (int j = 0; j < zoom; j++) {
                a = j * 1.0 / zoom;
                b = i * 1.0 / zoom;
                d_img_res[row * img_w * zoom * zoom + col * zoom + i * img_w * zoom + j] = (1 - a) * (1 - b) * f01 + a * (1 - b) * f11 + (1 - a) * b * f00 + a * b * f10;
            }
        }
    }
}

void apply_bilinear_interpolation(unsigned char* img_data, int img_h, int img_w, int zoom, unsigned char* img_res)
{
    unsigned char* d_img_res = NULL;

    // allocate GPU memory
    cudaMalloc((void**)&d_img_res, img_h * img_w * zoom * zoom);
    
    // creating texture based on the original image
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaArray* cu_arr;
    cudaMallocArray(&cu_arr, &channelDesc, img_w, img_h);
    cudaMemcpyToArray(cu_arr, 0, 0, img_data, img_h * img_w, cudaMemcpyHostToDevice);
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModePoint;
    cudaBindTextureToArray(texRef, cu_arr, channelDesc);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((img_w + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (img_h + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    bilinear_interpolation_gpu << < blocksPerGrid, threadsPerBlock >> > (img_h, img_w, zoom, d_img_res);
        
    // copy back from GPU
    cudaMemcpy(img_res, d_img_res, img_h * img_w * zoom * zoom, cudaMemcpyDeviceToHost);

    // free GPU memory allocation
    cudaFree(d_img_res);
    cudaFreeArray(cu_arr);
}


#define NUM_VARS 8

void gauss_jordan(double M[NUM_VARS][NUM_VARS + 1], int n) 
{
    double may;
    int ind;
    double aux;
    double pivote;

    for (int k = 0; k < n; k++) {
        may = abs(M[k][k]);
        ind = k;
        for (int l = k + 1; l < n; l++) {
            if (may < abs(M[l][k])) {
                may = abs(M[l][k]);
                ind = l;
            }
        }
        // change rows
        if (k != ind) {
            for (int i = 0; i < n + 1; i++) {
                aux = M[k][i];
                M[k][i] = M[ind][i];
                M[ind][i] = aux;
            }
        }
        if (M[k][k] == 0) {
            cout << "There is not a solution";
            break;
        }
        else {
            for (int i = 0; i < n; i++) {
                if (i != k) {
                    pivote = -M[i][k];
                    for (int j = k; j < n + 1; j++) {
                        M[i][j] = M[i][j] + pivote * M[k][j] / M[k][k];
                    }
                }
                else {
                    pivote = M[k][k];
                    for (int j = k; j < n + 1; j++) {
                        M[i][j] = M[i][j] / pivote;
                    }
                }
            }
        }
    }
}

void compute_coefficients(int img_h, int img_w, int* pts, double *coeffs) 
{
    int x1a = 0, y1a = 0, x2a = img_w - 1, y2a = 0;
    int x4a = 0, y4a = img_h - 1, x3a = img_w - 1, y3a = img_h - 1;
    double M[NUM_VARS][NUM_VARS + 1] = {
        {x1a, y1a, x1a * y1a,1,0,0,0,0,pts[0]},
        {0,0,0,0,x1a, y1a, x1a * y1a,1,pts[1]},
        {x2a, y2a, x2a * y2a,1,0,0,0,0,pts[2]},
        {0,0,0,0,x2a, y2a, x2a * y2a,1,pts[3]},
        {x3a, y3a, x3a * y3a,1,0,0,0,0,pts[4]},
        {0,0,0,0,x3a, y3a, x3a * y3a,1,pts[5]},
        {x4a, y4a, x4a * y4a,1,0,0,0,0,pts[6]},
        {0,0,0,0,x4a, y4a, x4a * y4a,1,pts[7]}
    };
    gauss_jordan(M, NUM_VARS);
    // save solution
    for (int i = 0; i < NUM_VARS; i++) {
        coeffs[i] = M[i][NUM_VARS];
    }
}

__global__ void bilinear_transformation_gpu(unsigned char* d_img, int img_h, int img_w, int n_channels, double *d_coeffs, unsigned char* d_img_res)
{
    int px = blockIdx.x;
    int py = blockIdx.y;
    int new_px = (int)(d_coeffs[0] * px + d_coeffs[1] * py + d_coeffs[2] * px * py + d_coeffs[3]);
    int new_py = (int)(d_coeffs[4] * px + d_coeffs[5] * py + d_coeffs[6] * px * py + d_coeffs[7]);
    
    if(px < img_w && py < img_h && new_px < img_w && new_py < img_h) {
        int pos = (px + py * gridDim.x) * n_channels;
        int new_pos = (new_px + new_py * gridDim.x) * n_channels;
        d_img_res[new_pos] = d_img[pos];
        if (n_channels == 3) {
            d_img_res[new_pos + 1] = d_img[pos + 1];
            d_img_res[new_pos + 2] = d_img[pos + 2];
        }
    }
}

void apply_bilinear_transformation(unsigned char* img_data, int img_h, int img_w, int n_channels, int *pts, unsigned char* img_res)
{
    double coeffs[NUM_VARS];
    compute_coefficients(img_h, img_w, pts, coeffs);
    for (int i = 0; i < NUM_VARS; i++)
        cout << coeffs[i] << endl;

    // declare GPU memory pointers
    unsigned char* d_img = NULL;
    double* d_coeffs = NULL;
    unsigned char* d_img_res = NULL;

    // allocate GPU memory
    cudaMalloc((void**)&d_img, img_h * img_w * n_channels);
    cudaMalloc((void**)&d_coeffs, NUM_VARS * sizeof(double));
    cudaMalloc((void**)&d_img_res, img_h * img_w * n_channels);

    // transfer the arrays to the GPU
    cudaMemcpy(d_img, img_data, img_h * img_w * n_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coeffs, coeffs, NUM_VARS * sizeof(double), cudaMemcpyHostToDevice);

    //launch the kernel
    dim3 grid_img(img_w, img_h);
    bilinear_transformation_gpu << < grid_img, 1 >> > (d_img, img_h, img_w, n_channels, d_coeffs, d_img_res);

    // copy back from GPU
    cudaMemcpy(img_res, d_img_res, img_h * img_w * n_channels, cudaMemcpyDeviceToHost);

    // free GPU memory allocation
    cudaFree(d_img_res);
    cudaFree(d_coeffs);
    cudaFree(d_img);
}