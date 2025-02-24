#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include <float.h>
using namespace cv;

void tutorial();
void negative_image();
void additive_factor(int factor);
void multiplicative_factor(float factor);
void color_image();
void matrix_and_inverse();
void rotate_image();

int main() {

    tutorial();
    //negative_image();
    //additive_factor(150);
    //multiplicative_factor(3.0f);
    //color_image();
    //matrix_and_inverse();
    //rotate_image();

    return 0;
}


// Lab tutorial by the teaching assistant
void tutorial() {
    Mat_<Vec3b> img(300, 400);
    img.setTo(0);

    for (int j = 0; j < img.cols; j++) {
        img(img.rows / 2, j)[1] = 255; // OpenCV stores blue, green, red (BGR) and not red, green, blue (RGB) => green line in the center
    }

    imshow("Image", img);

    waitKey(0);
}

// Practical work 2: Test the negative_image() function.
void negative_image() {
    Mat_<uchar> img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            img(i, j) = 255 - img(i, j);
        }
    }

    imshow("negative image", img);

    waitKey(0);
}

// Practical work 3: Implement a function which changes the gray levels of an image by an additive factor.
void additive_factor(int factor) {
    Mat_<uchar> img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);

    imshow("Initial image", img);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int result = img(i, j) + factor;

            if (result > 255) {
                img(i, j) = 255;
            }

            else if (result < 0) {
                img(i, j) = 0;
            }

            else {
                img(i, j) = result;
            }
        }
    }

    imshow("Additive image", img);

    waitKey(0);
}

// Practical work 4: Implement a function which changes the gray levels of an image by a multiplicative factor. Save the resulting image.
void multiplicative_factor(float factor) {
    Mat_<uchar> img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);

    imshow("Initial image", img);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float result = img(i, j) * factor;

            if (result > 255) {
                img(i, j) = 255;
            }

            else if (result < 0) {
                img(i, j) = 0;
            }

            else {
                img(i, j) = result;
            }
        }
    }

    imshow("Multiplicative image", img);
    imwrite("Images/multiplicative_image.bmp", img);

    waitKey(0);
}

// Practical work 5: Create a color image of dimension 256 x 256. Divide it into 4 squares and color the squares from top to bottom, left to right as: white, red, green, yellow.
void color_image() {
    Mat_<Vec3b> img(256, 256);
    img.setTo(0);

    int midRow = img.rows / 2;
    int midCol = img.cols / 2;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (i < midRow && j < midCol) {
                // Top-left = white
                img(i, j) = Vec3b(255, 255, 255);
            }

            else if (i < midRow && j >= midCol) {
                // Top-right = red
                img(i, j) = Vec3b(0, 0, 255);
            }

            else if (i >= midRow && j < midCol) {
                // Bottom-left = green
                img(i, j) = Vec3b(0, 255, 0);
            }

            else {
                // Bottom-right = yellow
                img(i, j) = Vec3b(0, 255, 255);
            }
        }
    }

    imshow("Colored squares image", img);

    waitKey(0);
}

// Practical work 6: Create a 3x3 float matrix, determine its inverse and print it.
void matrix_and_inverse() {
    float values[9] = {2.0f, -1.0f, 0.0f,
                       -1.0f, 2.0f, -1.0f,
                       0.0f, -1.0f, 2.0f};

    Mat M(3, 3, CV_32FC1, values);
    Mat MInv = M.inv(DECOMP_LU);

    if (!MInv.empty()) {
        std::cout << "M: " << std::endl << M << std::endl << std::endl;
        std::cout << "MInv: " << std::endl << MInv << std::endl << std::endl;
        std::cout << "M * MInv: " << std::endl << M * MInv << std::endl << std::endl; // Ensuring the computed inverse matrix is correct by checking if M * MInv gives us the identity matrix
    }

    else {
        std::cout << "Can't inverse matrix M because its determinant is equal to 0" << std::endl;
    }
}

// Bonus practical work: Rotate any image 90 degrees clockwise
void rotate_image() {
    Mat_<Vec3b> img = imread("Images/cameraman.bmp");
    Mat_<Vec3b> rotatedImg(img.rows, img.cols, img.type());

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            rotatedImg(j, img.rows - 1 - i) = img(i, j);
        }
    }

    imshow("Initial image", img);
    imshow("Rotated image", rotatedImg);

    waitKey(0);
}


