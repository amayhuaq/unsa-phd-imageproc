#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include "helpers.h"
#include "histogram_functions.h"
#include "image_operations.h"

using namespace std;
using namespace cv;

void ejercicio_01(String image_name, int color) 
{
    // read image
    Mat img = imread(image_name);
    if (!img.data) {
        cout << "The image was not found\n";
        return;
    }

    Mat bgr_channels[3];
    split(img, bgr_channels);

    imshow("Input image", img);

    int h_hist[256] = { 0 };
    int h_hist_b[256] = { 0 };
    int h_hist_g[256] = { 0 };
    int h_hist_r[256] = { 0 };

    switch (color)
    {
        case 0:
            compute_histogram_1d(bgr_channels[0].data, img.rows, img.cols, h_hist);
            display_histogram(h_hist, "Histogram 1D");
            break;
        case 1:
            compute_histogram_1d(bgr_channels[0].data, img.rows, img.cols, h_hist_b);
            compute_histogram_1d(bgr_channels[1].data, img.rows, img.cols, h_hist_g);
            compute_histogram_1d(bgr_channels[2].data, img.rows, img.cols, h_hist_r);
            display_histogram(h_hist_b, "Histogram 1D-B", 256, 400, Scalar(255, 0, 0));
            display_histogram(h_hist_g, "Histogram 1D-G", 256, 400, Scalar(0, 255, 0));
            display_histogram(h_hist_r, "Histogram 1D-R", 256, 400, Scalar(0, 0, 255));

            display_3histogram(h_hist_r, h_hist_g, h_hist_b, "Histogram 3 colors");

            //int *h_hist_3d = compute_histogram_3d(bgr_channels[2].data, bgr_channels[1].data, bgr_channels[0].data, img.rows, img.cols);
            //display_histogram(h_hist_3d, "Histogram 3D", 512);
            break;
    }

    waitKey(0);
}

void ejercicio_02(String image_name, int color)
{
    // read image
    Mat img = imread(image_name, color);
    if (!img.data) {
        cout << "The image was not found\n";
        return;
    }
    Mat img_eq;
    switch (color)
    {
        case 0:
            img_eq = img.clone();
            equalize(img.data, img.rows, img.cols, img.channels(), img_eq.data);
            break;
        case 1:
            Mat bgr_channels[3];
            split(img, bgr_channels);

            vector<Mat> img_eq_chs;
            img_eq_chs.push_back(bgr_channels[0].clone());
            img_eq_chs.push_back(bgr_channels[1].clone());
            img_eq_chs.push_back(bgr_channels[2].clone());
            
            equalize(bgr_channels[0].data, img.rows, img.cols, img_eq_chs[0].channels(), img_eq_chs[0].data);
            equalize(bgr_channels[1].data, img.rows, img.cols, img_eq_chs[1].channels(), img_eq_chs[1].data);
            equalize(bgr_channels[2].data, img.rows, img.cols, img_eq_chs[2].channels(), img_eq_chs[2].data);
            merge(img_eq_chs, img_eq);
            
            break;
    }

    imshow("Input image", img);
    imshow("Equalized image", img_eq);

    waitKey(0);
}

void ejercicio_03()
{
    // read image
    String image_name;
    cout << "Image name: ";
    cin >> image_name;
    Mat img = imread(image_name);
    if (!img.data) {
        cout << "The image was not found\n";
        return;
    }
    double A, B;
    cout << "Set A value: ";
    cin >> A;
    cout << "Set B value: ";
    cin >> B;

    Mat img_res = img.clone();
    apply_function(img.data, img.rows, img.cols, img.channels(), A, B, img_res.data);

    imshow("Input image", img);
    imshow("Output image", img_res);

    waitKey(0);
}

void ejercicio_04()
{
    // read image
    String image_name1, image_name2;
    cout << "Image 1 name: "; cin >> image_name1;
    cout << "Image 2 name: "; cin >> image_name2;
    Mat img1 = imread(image_name1);
    Mat img2 = imread(image_name2);
    if (!img1.data || !img2.data) {
        cout << "Some image was not found\n";
        return;
    }
    
    Mat img_res = Mat::zeros(img1.rows, img1.cols, img1.type());
    int operation;
    cout << "Select operation:\n 1 - sum\n 2 - subs\n 3 - mult\n 4 - div\n>> ";
    cin >> operation;
    apply_aritmethic_operation(img1.data, img2.data, img1.rows, img1.cols, img1.channels(), img_res.data, operation);

    imshow("Image 1", img1);
    imshow("Image 2", img2);
    imshow("Output image", img_res);

    waitKey(0);
}

void ejercicio_05(String image_name, int color)
{
    // read image
    Mat img = imread(image_name, color);
    if (!img.data) {
        cout << "The image was not found\n";
        return;
    }
    int win_size;
    cout << "Window size: ";
    cin >> win_size;
    
    Mat img_media, img_sobel;
    switch (color)
    {
        case 0:
            img_media = img.clone();
            img_sobel = img.clone();
            apply_media_convolution(img.data, img.rows, img.cols, img.channels(), win_size, img_media.data);
            apply_sobel_convolution(img.data, img.rows, img.cols, img.channels(), img_sobel.data);
            break;
        case 1:
            Mat bgr_channels[3];
            split(img, bgr_channels);

            vector<Mat> img_med_chs;
            img_med_chs.push_back(bgr_channels[0].clone());
            img_med_chs.push_back(bgr_channels[1].clone());
            img_med_chs.push_back(bgr_channels[2].clone());
            apply_media_convolution(bgr_channels[0].data, img.rows, img.cols, img_med_chs[0].channels(), win_size, img_med_chs[0].data);
            apply_media_convolution(bgr_channels[1].data, img.rows, img.cols, img_med_chs[1].channels(), win_size, img_med_chs[1].data);
            apply_media_convolution(bgr_channels[2].data, img.rows, img.cols, img_med_chs[2].channels(), win_size, img_med_chs[2].data);
            merge(img_med_chs, img_media);

            vector<Mat> img_sob_chs;
            img_sob_chs.push_back(bgr_channels[0].clone());
            img_sob_chs.push_back(bgr_channels[1].clone());
            img_sob_chs.push_back(bgr_channels[2].clone());
            apply_sobel_convolution(bgr_channels[0].data, img.rows, img.cols, img_sob_chs[0].channels(), img_sob_chs[0].data);
            apply_sobel_convolution(bgr_channels[1].data, img.rows, img.cols, img_sob_chs[1].channels(), img_sob_chs[1].data);
            apply_sobel_convolution(bgr_channels[2].data, img.rows, img.cols, img_sob_chs[2].channels(), img_sob_chs[2].data);
            merge(img_sob_chs, img_sobel);
            break;
    }

    imshow("Input image", img);
    imshow("Media", img_media);
    imshow("Sobel", img_sobel);
    
    waitKey(0);
}

void ejercicio_06(String image_name)
{
    // read image
    Mat img = imread(image_name, 0);
    if (!img.data) {
        cout << "The image was not found\n";
        return;
    }
    int zoom;
    cout << "Zoom: ";
    cin >> zoom;

    Mat img_zoom = Mat::zeros(Size(img.cols * zoom, img.rows * zoom), img.type());
    apply_bilinear_interpolation(img.data, img.rows, img.cols, img.channels(), zoom, img_zoom.data);
    
    imshow("Input image", img);
    imshow("Output image", img_zoom);

    waitKey(0);
}

int main()
{
    int n_ex, color;
    String img_name;
    
    while (true) 
    {
        cout << "Select exercise: ";
        cin >> n_ex;
        
        switch (n_ex)
        {
            case 1:
                cout << "Image name: "; cin >> img_name;
                cout << "Color: 0 Gray, 1 Color: "; cin >> color;
                ejercicio_01(img_name, color);
                break;
            case 2:
                cout << "Image name: "; cin >> img_name;
                cout << "Color: 0 Gray, 1 Color: "; cin >> color;
                ejercicio_02(img_name, color);
                break;
            case 3:
                ejercicio_03();
                break;
            case 4:
                ejercicio_04();
                break;
            case 5:
                cout << "Image name: "; cin >> img_name;
                cout << "Color: 0 Gray, 1 Color: "; cin >> color;
                ejercicio_05(img_name, color);
                break;
            //case 6:
            //    cout << "Image name: "; cin >> img_name;
            //    ejercicio_06(img_name);
            //    break;
            default:
                return -1;
        }
    }
}
