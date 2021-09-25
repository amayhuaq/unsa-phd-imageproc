#include <opencv2/core/core.hpp>
using namespace cv;

// Function to display an histogram as image
void display_histogram(int histogram[], const char* name, int num_bins = 256, int hist_h = 400, Scalar color= Scalar(0, 0, 0));
void display_3histogram(int hist_r[], int hist_g[], int hist_b[], const char* name, int num_bins = 256, int hist_h = 400);
