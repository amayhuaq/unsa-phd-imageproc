void apply_function(unsigned char* img_data, int img_h, int img_w, int n_channels, double A, double B, unsigned char* img_res);
void apply_aritmethic_operation(unsigned char* img1, unsigned char* img2, int img_h, int img_w, int n_channels, unsigned char* img_res, int operation);

void apply_media_convolution(unsigned char* img_data, int img_h, int img_w, int n_channels, int win_size, unsigned char* img_res);
void apply_sobel_convolution(unsigned char* img_data, int img_h, int img_w, int n_channels, unsigned char* img_res);

void apply_bilinear_interpolation(unsigned char* img_data, int img_h, int img_w, int zoom, unsigned char* img_res);
void apply_bilinear_transformation(unsigned char* img_data, int img_h, int img_w, int n_channels, int *pts, unsigned char* img_res);