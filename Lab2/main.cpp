#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <random>
using namespace cv;

void get_rgb_channels();
void rgb_to_gray();
void gray_to_binary();
void compute_h_s_v();
bool isInside(Mat img, int i, int j);

int main() {
    //get_rgb_channels();
    //rgb_to_gray();
    //gray_to_binary();
    //compute_h_s_v();

    // Mat_<Vec3b> img = imread("Images/Lena_24bits.bmp");
    // std::cout << isInside(img, 1, 10) << std::endl;
    // std::cout << isInside(img, 1, 100) << std::endl;
    // std::cout << isInside(img, 1, 1000) << std::endl;
    // std::cout << isInside(img, 1, 10000) << std::endl;

    return 0;
}

// Practical work 1: Create a function that will copy the R, G and B channels of a color, RGB24 image (CV_8UC3 type)
// into three matrices of type CV_8UC1 (grayscale images). Display these matrices in three distinct windows.
void get_rgb_channels() {
    // 1. First I'll write the code which just separates the RGB channels in colored images 'cause I'm curious how it would look like :P

    // Mat_<Vec3b> img = imread("Images/Lena_24bits.bmp");
    // Mat_<Vec3b> red(img.rows, img.cols);
    // Mat_<Vec3b> green(img.rows, img.cols);
    // Mat_<Vec3b> blue(img.rows, img.cols);
    //
    // imshow("Initial image", img);
    //
    // for (int i = 0; i < img.rows; i++) {
    //     for (int j = 0; j < img.cols; j++) {
    //         blue(i, j) = Vec3b(img(i, j)[0], 0, 0);
    //         green(i, j) = Vec3b(0, img(i, j)[1], 0);
    //         red(i, j) = Vec3b(0, 0, img(i, j)[2]);
    //     }
    // }
    //
    // imshow("Red image", red);
    // imshow("Green image", green);
    // imshow("Blue image", blue);
    //
    // waitKey(0);

    // -------------------------------------------------------------------------------------------------

    // 2. The actual code for the practical work

    Mat_<Vec3b> img = imread("Images/Lena_24bits.bmp");
    Mat_<uchar> red(img.rows, img.cols);
    Mat_<uchar> green(img.rows, img.cols);
    Mat_<uchar> blue(img.rows, img.cols);

    imshow("Initial image", img);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            blue(i, j) = img(i, j)[0];
            green(i, j) = img(i, j)[1];
            red(i, j) = img(i, j)[2];
        }
    }

    imshow("Red image", red);
    imshow("Green image", green);
    imshow("Blue image", blue);

    waitKey(0);
}

// Practical work 2: Create a function that will convert a color RGB24 image (CV_8UC3 type) to a grayscale image (CV_8UC1),
// and display the result image in a destination window.
void rgb_to_gray() {
    Mat_<Vec3b> img = imread("Images/Lena_24bits.bmp");
    Mat_<uchar> gray(img.rows, img.cols);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int result = (img(i, j)[0] + img(i, j)[1] + img(i, j)[2]) / 3;
            result = std::clamp(result, 0, 255); // Force the result in the [0, 255] interval. If result is < 0 then result = 0, else if result > 255 then result = 255, else result = result
            gray(i, j) = result;
        }
    }

    imshow("Initial image", img);
    imshow("Grayscale image", gray);

    waitKey(0);
}

// Practical work 3: Create a function for converting from grayscale to black and white (binary), using (2.2).
// Read the threshold from the console. Test the operation on multiple images, and using multiple thresholds.
void gray_to_binary() {
    Mat_<uchar> img = imread("Images/Lena_24bits.bmp", IMREAD_GRAYSCALE);
    Mat_<uchar> binary(img.rows, img.cols);
    int threshold = -1;

    // A little trick to force the user to input a valid threshold
    while (threshold < 0 || threshold > 255) {
        std::cout << "Input the threshold (from 0 to 255): ";
        std::cin >> threshold;
    }

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

    imshow("Initial image", img);
    imshow("Binary image", binary);

    waitKey(0);
}

// Practical work 4: Create a function that will compute the H, S and V values from the R, G, B channels of an image,
// using the equations from 2.6. Store each value(H, S, V) in a CV_8UC1 matrix. Display these matrices in distinct windows.
// Check the correctness of your implementation using the example below.
void compute_h_s_v() {
    Mat_<Vec3b> img = imread("Images/Lena_24bits.bmp");
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

    imshow("Initial image", img);

    imshow("Hue image", hue);
    imshow("Saturation image", saturation);
    imshow("Value image", value);

    waitKey(0);
}

// Practical work 5: Implement a function called isInside(img, i, j) which checks if the position
// indicated by the pair (i, j) (row, column) is inside the image img.
bool isInside(Mat img, int i, int j) {
    return i >= 0 && i < img.rows && j >= 0 && j < img.cols;
}


