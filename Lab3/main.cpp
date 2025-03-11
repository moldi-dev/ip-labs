#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <random>
#include <float.h>

using namespace cv;
using namespace std;

// Given functions in the laboratory guide
void show_histogram(const string& name, const int* hist, int hist_cols, int hist_height);
void show_histogram_float_type(const string& name, const float* hist, int hist_cols, int hist_height);

// Practical work 1: compute the histogram for a given grayscale image (in an array of integers having dimension 256).
void compute_histogram(const Mat_<uchar>& image, int* hist);
void practical_work_1();

// Practical work 2: compute the PDF (in an array of floats of dimension 256).
void compute_pdf(const int* hist, int total_pixels, float* pdf);
void practical_work_2();

// Practical work 3: Display the computed histogram using the provided function.
void practical_work_3();

// Practical work 4: Compute the histogram for a given number of bins m ≤ 256.
void compute_histogram(const Mat_<uchar>& image, int* hist, int bins);
void practical_work_4();

// Practical work 5: Implement the multilevel thresholding algorithm from section 3.3.
vector<int> find_local_maxima(const float* pdf, int window_size, float threshold);
Mat_<uchar> apply_thresholding(const Mat& image, const vector<int>& maxima);
void practical_work_5();

// Practical work 6: Enhance the multilevel thresholding algorithm using the Floyd-Steinberg dithering from section 3.4.
bool is_inside(const Mat& img, int i, int j);
Mat_<uchar> apply_thresholding_floyd_steinberg_dithering(const Mat_<uchar>& image, const vector<int>& maxima);
void practical_work_6();

// Practical work 7: Perform multilevel thresholding on a color image by applying the procedure from 3.3 on the Hue channel from the HSV
// color - space representation of the image. Modify only the Hue values, keeping the S and V channels unchanged or setting them to their
// maximum possible value. Transform the result back to RGB color - space for viewing.
array<Mat_<uchar>, 3> compute_h_s_v(Mat_<Vec3b>& img);
Mat_<Vec3b> h_s_v_to_r_g_b(const array<Mat_<uchar>, 3>& h_s_v);
void practical_work_7();

int main() {

    //practical_work_1(); // equivalent to practical work 3
    //practical_work_2();
    //practical_work_3(); // equivalent to practical work 1
    //practical_work_4();
    //practical_work_5();
    //practical_work_6();
    //practical_work_7();

    return 0;
}

void show_histogram(const string& name, const int* hist, const int hist_cols, const int hist_height) {
    int max_hist = 0;
    double scale = 1.0;
    int baseline = hist_height - 1;
    Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));

    for (int i = 0; i < hist_cols; i++) {
        if (hist[i] > max_hist) {
            max_hist = hist[i];
        }
    }

    scale = (double)hist_height / max_hist;

    for (int x = 0; x < hist_cols; x++) {
        Point p1 = Point(x, baseline);
        Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
        line(imgHist, p1, p2, CV_RGB(255, 0, 255));
    }

    imshow(name, imgHist);
}

void show_histogram_float_type(const string& name, const float* hist, const int hist_cols, const int hist_height) {
    float max_data = 0;
    double scale = 1.0;
    int baseline = hist_height - 1;
    Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));

    for (int i = 0; i < hist_cols; i++) {
        if (hist[i] > max_data) {
            max_data = hist[i];
        }
    }

    scale = (double)hist_height / max_data;

    for (int x = 0; x < hist_cols; x++) {
        Point p1 = Point(x, baseline);
        Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
        line(imgHist, p1, p2, CV_RGB(0, 255, 0));
    }

    imshow(name, imgHist);
}

void compute_histogram(const Mat_<uchar>& image, int* hist) {
    for (int i = 0; i < 256; i++) {
        hist[i] = 0;
    }

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            const int intensity = image(i, j);
            hist[intensity]++;
        }
    }
}

void compute_pdf(const int* hist, int total_pixels, float* pdf) {
    for (int i = 0; i < 256; i++) {
        pdf[i] = (float)(hist[i]) / total_pixels;
    }
}

void compute_histogram(const Mat_<uchar>& image, int* hist, int bins) {
    for (int i = 0; i < bins; i++) {
        hist[i] = 0;
    }

    float bin_width = 256.0f / bins;

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int intensity = image(i, j);
            int bin_index = (int)(intensity / bin_width);
            bin_index = min(bin_index, bins - 1);
            hist[bin_index]++;
        }
    }
}

vector<int> find_local_maxima(const float* pdf, int window_size, float threshold) {
    vector<int> maxima;
    int half_window = window_size / 2;

    for (int k = half_window; k < 256 - half_window; k++) {
        float window_avg = 0.0f;

        for (int i = k - half_window; i <= k + half_window; i++) {
            window_avg += pdf[i];
        }

        window_avg /= window_size;

        if (pdf[k] > window_avg + threshold) {
            bool is_maxima = true;

            for (int i = k - half_window; i <= k + half_window; i++) {
                if (pdf[i] > pdf[k]) {
                    is_maxima = false;
                    break;
                }
            }

            if (is_maxima) {
                maxima.push_back(k);
            }
        }
    }

    maxima.insert(maxima.begin(), 0);
    maxima.push_back(255);

    return maxima;
}

bool is_inside(const Mat& img, const int i, const int j) {
    return i >= 0 && i < img.rows && j >= 0 && j < img.cols;
}

Mat_<uchar> apply_thresholding(const Mat_<uchar>& image, const vector<int>& maxima) {
    Mat_<uchar> result = image.clone();

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int intensity = image(i, j);
            int new_intensity = 0;

            for (size_t k = 0; k < maxima.size() - 1; k++) {
                if (intensity >= maxima[k] && intensity <= maxima[k + 1]) {
                    if (abs(intensity - maxima[k]) < abs(intensity - maxima[k + 1])) {
                        new_intensity = maxima[k];
                    }

                    else {
                        new_intensity = maxima[k + 1];
                    }

                    break;
                }
            }

            result(i, j) = (uchar)(new_intensity);
        }
    }

    return result;
}

Mat_<uchar> apply_thresholding_floyd_steinberg_dithering(const Mat_<uchar>& image, const vector<int>& maxima) {
    Mat_<uchar> clone = image.clone();
    Mat_<uchar> result(image.rows, image.cols);

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int intensity = clone(i, j);
            int new_intensity = 0;

            for (size_t k = 0; k < maxima.size() - 1; k++) {
                if (intensity >= maxima[k] && intensity <= maxima[k + 1]) {
                    if (abs(intensity - maxima[k]) < abs(intensity - maxima[k + 1])) {
                        new_intensity = maxima[k];
                    }

                    else {
                        new_intensity = maxima[k + 1];
                    }

                    break;
                }
            }

            result(i, j) = (uchar)(new_intensity);

            int error = intensity - new_intensity;

            if (is_inside(image, i, j + 1)) {
                clone(i, j + 1) = clamp(clone(i, j + 1) + 7 * error / 16, 0, 255);
            }

            if (is_inside(image, i + 1, j - 1)) {
                clone(i + 1, j - 1) = clamp(clone(i + 1, j - 1) + 3 * error / 16, 0, 255);
            }

            if (is_inside(image, i + 1, j)) {
                clone(i + 1, j) = clamp(clone(i + 1, j) + 5 * error / 16, 0, 255);
            }

            if (is_inside(image, i + 1, j + 1)) {
                clone(i + 1, j + 1) = clamp(clone(i + 1, j + 1) + error / 16, 0, 255);
            }
        }
    }

    return result;
}

array<Mat_<uchar>, 3> compute_h_s_v(Mat_<Vec3b>& img) {
    Mat_<uchar> hue(img.rows, img.cols);
    Mat_<uchar> saturation(img.rows, img.cols);
    Mat_<uchar> value(img.rows, img.cols);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float r = (float)img(i, j)[2] / 255;
            float g = (float)img(i, j)[1] / 255;
            float b = (float)img(i, j)[0] / 255;

            float M = max(max(r, g), b);
            float m = min(min(r, g), b);

            float C = M - m;

            // 1. Value
            float v = M;

            // 2. Saturation
            float s = 0;

            if (v != 0) {
                s = C / v;
            }

            // 3. Hue
            float h = 0;

            if (C != 0) {
                if (M == r) {
                    h = 60 * (g - b) / C;
                }

                if (M == g) {
                    h = 120 + 60 * (b - r) / C;
                }

                if (M == b) {
                    h = 240 + 60 * (r - g) / C;
                }
            }

            if (h < 0) {
                h = h + 360;
            }

            v = v * 255;
            s = s * 255;
            h = h * 180 / 360;

            value(i, j) = clamp((int)v, 0, 255);
            hue(i, j) = clamp((int)h, 0, 255);
            saturation(i, j) = clamp((int)s, 0, 255);
        }
    }

    return array<Mat_<uchar>, 3>({hue, saturation, value});
}

Mat_<Vec3b> h_s_v_to_r_g_b(const array<Mat_<uchar>, 3>& h_s_v) {
    Mat hsv_image;
    merge(vector<Mat>{h_s_v[0]*180/255, h_s_v[1], h_s_v[2]}, hsv_image);

    Mat rgb_image;
    cvtColor(hsv_image, rgb_image, COLOR_HSV2BGR);

    return rgb_image;
}

void practical_work_1() {
    const Mat_<uchar> image = imread("Images/kids.bmp", IMREAD_GRAYSCALE);

    int histogram[256] = {0};

    compute_histogram(image, histogram);

    show_histogram("Histogram", histogram, 256, 400);

    waitKey(0);
}

void practical_work_2() {
    Mat_<uchar> image = imread("Images/kids.bmp", IMREAD_GRAYSCALE);

    // Step 1: Compute the histogram
    int histogram[256] = {0};
    compute_histogram(image, histogram);

    // Step 2: Compute the PDF
    float pdf[256] = {0.0f};
    int total_pixels = image.rows * image.cols;
    compute_pdf(histogram, total_pixels, pdf);

    // Step 3: Display the PDF
    show_histogram_float_type("PDF", pdf, 256, 400);

    waitKey(0);
}

void practical_work_3() {
    const Mat_<uchar> image = imread("Images/kids.bmp", IMREAD_GRAYSCALE);

    int histogram[256] = {0};

    compute_histogram(image, histogram);

    show_histogram("Histogram", histogram, 256, 400);

    waitKey(0);
}

void practical_work_4() {
    const Mat_<uchar> image = imread("Images/kids.bmp", IMREAD_GRAYSCALE);

    int histogram[256] = {0};

    compute_histogram(image, histogram, 100);

    show_histogram("Histogram", histogram, 256, 400);

    waitKey(0);
}

void practical_work_5() {
    Mat_<uchar> image = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);

    // Step 1: Compute the histogram
    int histogram[256] = {0};
    compute_histogram(image, histogram);

    // Step 2: Compute the PDF
    float pdf[256] = {0.0f};
    int total_pixels = image.rows * image.cols;
    compute_pdf(histogram, total_pixels, pdf);

    // Step 3: Find local maxima in the PDF
    int window_size = 11; // Window size for local maxima detection (2 * WH + 1, WH = 5)
    float threshold = 0.0003f; // Threshold for local maxima
    vector<int> maxima = find_local_maxima(pdf, window_size, threshold);

    // Step 4: Apply thresholding
    Mat_<uchar> thresholded_image = apply_thresholding(image, maxima);

    // Display the results
    imshow("Original Image", image);
    imshow("Thresholded Image", thresholded_image);

    waitKey(0);
}

void practical_work_6() {
    Mat_<uchar> image = imread("Images/saturn.bmp", IMREAD_GRAYSCALE);

    // Step 1: Compute the histogram
    int histogram[256] = {0};
    compute_histogram(image, histogram);

    // Step 2: Compute the PDF
    float pdf[256] = {0.0f};
    int total_pixels = image.rows * image.cols;
    compute_pdf(histogram, total_pixels, pdf);

    // Step 3: Find local maxima in the PDF
    int window_size = 11; // Window size for local maxima detection (2 * WH + 1, WH = 5)
    float threshold = 0.0003f; // Threshold for local maxima
    vector<int> maxima = find_local_maxima(pdf, window_size, threshold);

    // Step 4: Apply thresholding
    Mat_<uchar> thresholded_image = apply_thresholding_floyd_steinberg_dithering(image, maxima);

    // Display the results
    imshow("Original Image", image);
    imshow("Thresholded Image", thresholded_image);

    waitKey(0);
}

void practical_work_7() {
    Mat_<Vec3b> img = imread("Images/Lena_24bits.bmp");

    array<Mat_<uchar>, 3> hsv_channels = compute_h_s_v(img);

    // Step 1: Compute the histogram of the Hue channel
    int histogram[256] = {0};
    compute_histogram(hsv_channels[0], histogram);

    // Step 2: Compute the PDF
    float pdf[256] = {0.0f};
    int total_pixels = hsv_channels[0].rows * hsv_channels[0].cols;
    compute_pdf(histogram, total_pixels, pdf);

    // Step 3: Find local maxima in the PDF
    int window_size = 11; // Window size for local maxima detection (2 * WH + 1, WH = 5)
    float threshold = 0.0003f; // Threshold for local maxima
    vector<int> maxima = find_local_maxima(pdf, window_size, threshold);

    // Step 4: Apply thresholding to the Hue channel and change the saturation and value to their maximum values
    Mat_<uchar> thresholded_hue = apply_thresholding(hsv_channels[0], maxima);

    hsv_channels[0] = thresholded_hue;

    hsv_channels[1].setTo(255);
    hsv_channels[2].setTo(255);

    // Step 5: Convert the modified HSV image back to RGB color space
    Mat_<Vec3b> result_img = h_s_v_to_r_g_b(hsv_channels);

    // Display the results
    imshow("Original Image", img);
    imshow("Thresholded Hue Image", result_img);

    waitKey(0);
}