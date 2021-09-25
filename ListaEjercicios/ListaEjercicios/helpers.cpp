#include "helpers.h"
//#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

//using namespace cv;
using namespace std;


void display_histogram(int histogram[], const char* name, int num_bins, int hist_h, Scalar color)
{
    const int PADDING = 10;
    int *hist = new int[num_bins];
    int hist_w = num_bins * 2;
    for (int i = 0; i < num_bins; i++) {
        hist[i] = histogram[i];
    }
    // draw the histograms
    int bin_w = cvRound((double)hist_w / num_bins);
    Mat histImage(hist_h + 2 * PADDING, hist_w + 2 * PADDING, CV_8UC3, Scalar(255, 255, 255));

    // find the maximum intensity element from histogram
    int max = hist[0];
    for (int i = 1; i < num_bins; i++) {
        if (max < hist[i]) {
            max = hist[i];
        }
    }

    // normalize the histogram between 0 and histImage.rows
    for (int i = 0; i < num_bins; i++) {
        hist[i] = ((double)hist[i] / max) * hist_h; // histImage.rows;
    }

    // draw the intensity line for histogram
    for (int i = 0; i < num_bins; i++) {
        line(histImage, Point(bin_w * (i) + PADDING, hist_h + PADDING), Point(bin_w * (i) + PADDING, hist_h + PADDING - hist[i]), color, 1, 8, 0);
    }

    // display histogram
    namedWindow(name, WND_PROP_AUTOSIZE);
    imshow(name, histImage);
}

void display_3histogram(int hist_r[], int hist_g[], int hist_b[], const char* name, int num_bins, int hist_h)
{
    const int PADDING = 10;
    int* hist_rt = new int[num_bins];
    int* hist_gt = new int[num_bins];
    int* hist_bt = new int[num_bins];
    int hist_w = num_bins * 2;
    for (int i = 0; i < num_bins; i++) {
        hist_rt[i] = hist_r[i];
        hist_gt[i] = hist_g[i];
        hist_bt[i] = hist_b[i];
    }
    // draw the histograms
    int bin_w = cvRound((double)hist_w / num_bins);
    Mat histImage(hist_h + 2 * PADDING, hist_w + 2 * PADDING, CV_8UC3, Scalar(255, 255, 255));

    // find the maximum intensity element from histogram
    int max = -1;
    for (int i = 0; i < num_bins; i++) {
        if (max < hist_rt[i]) {
            max = hist_rt[i];
        }
        if (max < hist_gt[i]) {
            max = hist_gt[i];
        }
        if (max < hist_bt[i]) {
            max = hist_bt[i];
        }
    }

    // normalize the histogram between 0 and histImage.rows
    for (int i = 0; i < num_bins; i++) {
        hist_rt[i] = ((double)hist_rt[i] / max) * hist_h;
        hist_gt[i] = ((double)hist_gt[i] / max) * hist_h;
        hist_bt[i] = ((double)hist_bt[i] / max) * hist_h;
    }

    // draw the intensity line for histogram
    for (int i = 0; i < num_bins; i++) {
        line(histImage, Point(bin_w * (i)+PADDING, hist_h + PADDING), Point(bin_w * (i)+PADDING, hist_h + PADDING - hist_rt[i]), Scalar(0,0,255), 1, 8, 0);
        line(histImage, Point(bin_w * (i)+PADDING, hist_h + PADDING), Point(bin_w * (i)+PADDING, hist_h + PADDING - hist_gt[i]), Scalar(0,255,0), 1, 8, 0);
        line(histImage, Point(bin_w * (i)+PADDING, hist_h + PADDING), Point(bin_w * (i)+PADDING, hist_h + PADDING - hist_bt[i]), Scalar(255,0,0), 1, 8, 0);
    }

    // display histogram
    namedWindow(name, WND_PROP_AUTOSIZE);
    imshow(name, histImage);
}
