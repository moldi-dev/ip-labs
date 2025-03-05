#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <random>
using namespace cv;

std::array<Mat_<uchar>, 3> get_rgb_channels(Mat_<Vec3b>& img);
Mat_<uchar> rgb_to_gray(Mat_<Vec3b>& img);
Mat_<uchar> gray_to_binary(Mat_<uchar>& img, int threshold);
std::array<Mat_<uchar>, 3> compute_h_s_v(Mat_<Vec3b>& img);
bool is_inside(const Mat_<Vec3b>& img, int i, int j);
Mat_<Vec3b> h_s_v_to_r_g_b(const std::array<Mat_<uchar>, 3>& h_s_v);

void practical_work_1_test();
void practical_work_2_test();
void practical_work_3_test();
void practical_work_4_test();
void practical_work_5_test();
void practical_work_6_test();

int main() {
    //practical_work_1_test();
    //practical_work_2_test();
    //practical_work_3_test();
    practical_work_4_test();
    //practical_work_5_test();
    practical_work_6_test();

    waitKey(0);

    return 0;
}

// Practical work 1: Create a function that will copy the R, G and B channels of a color, RGB24 image (CV_8UC3 type)
// into three matrices of type CV_8UC1 (grayscale images). Display these matrices in three distinct windows.
std::array<Mat_<uchar>, 3> get_rgb_channels(Mat_<Vec3b>& img) {
    Mat_<uchar> red(img.rows, img.cols);
    Mat_<uchar> green(img.rows, img.cols);
    Mat_<uchar> blue(img.rows, img.cols);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            blue(i, j) = img(i, j)[0];
            green(i, j) = img(i, j)[1];
            red(i, j) = img(i, j)[2];
        }
    }

    return std::array<Mat_<uchar>, 3>({red, green, blue});
}

// Practical work 2: Create a function that will convert a color RGB24 image (CV_8UC3 type) to a grayscale image (CV_8UC1),
// and display the result image in a destination window.
Mat_<uchar> rgb_to_gray(Mat_<Vec3b>& img) {
    Mat_<uchar> gray(img.rows, img.cols);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int result = (img(i, j)[0] + img(i, j)[1] + img(i, j)[2]) / 3;
            gray(i, j) = result;
        }
    }

    return gray;
}

// Practical work 3: Create a function for converting from grayscale to black and white (binary), using (2.2).
// Read the threshold from the console. Test the operation on multiple images, and using multiple thresholds.
Mat_<uchar> gray_to_binary(Mat_<uchar>& img, int threshold) {
    Mat_<uchar> binary(img.rows, img.cols);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) < threshold) {
                binary(i, j) = 0;
            }

            else {
                binary(i, j) = 255;
            }
        }
    }

    return binary;
}

// Practical work 4: Create a function that will compute the H, S and V values from the R, G, B channels of an image,
// using the equations from 2.6. Store each value(H, S, V) in a CV_8UC1 matrix. Display these matrices in distinct windows.
// Check the correctness of your implementation using the example below.
std::array<Mat_<uchar>, 3> compute_h_s_v(Mat_<Vec3b>& img) {
    Mat_<uchar> hue(img.rows, img.cols);
    Mat_<uchar> saturation(img.rows, img.cols);
    Mat_<uchar> value(img.rows, img.cols);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float r = (float)img(i, j)[2] / 255;
            float g = (float)img(i, j)[1] / 255;
            float b = (float)img(i, j)[0] / 255;

            float M = std::max(std::max(r, g), b);
            float m = std::min(std::min(r, g), b);

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
            h = h * 255 / 360;

            value(i, j) = std::clamp((int)v, 0, 255);
            hue(i, j) = std::clamp((int)h, 0, 255);
            saturation(i, j) = std::clamp((int)s, 0, 255);
        }
    }

    return std::array<Mat_<uchar>, 3>({hue, saturation, value});
}

// Practical work 5: Implement a function called isInside(img, i, j) which checks if the position
// indicated by the pair (i, j) (row, column) is inside the image img.
bool is_inside(const Mat_<Vec3b>& img, const int i, const int j) {
    return i >= 0 && i < img.rows && j >= 0 && j < img.cols;
}

// Bonus practical work: convert HSV to RGB
Mat_<Vec3b> h_s_v_to_r_g_b(const std::array<Mat_<uchar>, 3>& h_s_v) {
    Mat hsv_image;
    merge(std::vector<Mat>{h_s_v[0]*180/255, h_s_v[1], h_s_v[2]}, hsv_image);

    Mat rgb_image;
    cvtColor(hsv_image, rgb_image, COLOR_HSV2BGR);

    return rgb_image;
}

void practical_work_1_test() {
    Mat_<Vec3b> img = imread("Images/Lena_24bits.bmp");

    const std::array<Mat_<uchar>, 3> channels = get_rgb_channels(img);

    imshow("Red channel result", channels[0]);
    imshow("Green channel result", channels[1]);
    imshow("Blue channel result", channels[2]);
}

void practical_work_2_test() {
    Mat_<Vec3b> img = imread("Images/Lena_24bits.bmp");
    imshow("RGB to Grayscale result", rgb_to_gray(img));
}

void practical_work_3_test() {
    int threshold;

    Mat_<uchar> img2 = imread("Images/Lena_24bits.bmp", IMREAD_GRAYSCALE);

    std::cout << "Input the threshold (from 0 to 255): ";
    std::cin >> threshold;

    imshow("Gray to Binary result", gray_to_binary(img2, threshold));
}

void practical_work_4_test() {
    Mat_<Vec3b> img = imread("Images/Lena_24bits.bmp");
    const std::array<Mat_<uchar>, 3> hsv = compute_h_s_v(img);
    imshow("Hue image", hsv[0]);
    imshow("Saturation image", hsv[1]);
    imshow("Value image", hsv[2]);
}

void practical_work_5_test() {
    Mat_<Vec3b> img = imread("Images/Lena_24bits.bmp");

    std::cout << is_inside(img, 1, 10) << std::endl;
    std::cout << is_inside(img, 1, 100) << std::endl;
    std::cout << is_inside(img, 1, 1000) << std::endl;
    std::cout << is_inside(img, 1, 10000) << std::endl;
    std::cout << is_inside(img, 10, 1) << std::endl;
    std::cout << is_inside(img, 100, 1) << std::endl;
    std::cout << is_inside(img, 1000, 1) << std::endl;
    std::cout << is_inside(img, 10000, 1) << std::endl;
}

void practical_work_6_test() {
    Mat_<Vec3b> img = imread("Images/Lena_24bits.bmp");
    std::array<Mat_<uchar>, 3> hsv = compute_h_s_v(img);
    imshow("HSV to RGB result", h_s_v_to_r_g_b(hsv));
}




