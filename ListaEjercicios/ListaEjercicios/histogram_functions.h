void compute_histogram_1d(unsigned char * img_data, int img_h, int img_w, int *hist);
int* compute_histogram_3d(int* img_data_r, int* img_data_g, int* img_data_b, int img_h, int img_w);

void stretch(unsigned char* img_data, int img_h, int img_w, int n_channels, unsigned char* img_res);
void equalize(unsigned char* img_data, int img_h, int img_w, int n_channels, unsigned char* img_res);
