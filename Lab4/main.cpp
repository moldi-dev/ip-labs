#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <float.h>
#include <random>
#include <math.h>

using namespace cv;
using namespace std;

// Practical work 1: For a specific object in a labeled image selected by a mouse click, compute the object’s
// area, center of mass, axis of elongation, perimeter, thinness ratio, aspect ratio and projections.
vector<Point> find_object_pixels(Mat_<uchar>& image, int label);
int compute_object_area(vector<Point>& object_pixels);
Point2d compute_object_center_of_mass(vector<Point>& object_pixels);
double compute_object_axis_of_elongation(vector<Point>& object_pixels, Point2d center_of_mass);
int compute_object_perimeter(Mat_<uchar>& image, int label);
double compute_object_thinness(int object_area, int object_perimeter);
double compute_object_aspect_ratio(vector<Point>& object_pixels);
pair<vector<int>, vector<int>> compute_object_projections(vector<Point>& object_pixels, int rows, int columns);

// Practical work 1.a: Display the results in the standard output
void onMouse_1a(int event, int x, int y, int flags, void* param);
void practical_work_1_a();

// Practical work 1.b: In a separate image (source image clone)
void onMouse_1b(int event, int x, int y, int flags, void* param);
Mat_<Vec3b> draw_object_contour(Mat_<uchar>& image, int label);
void practical_work_1_b();

// Practical work 1.c: Compute and display the projections of the selected object in a separate image
// (source image clone).
void onMouse_1c(int event, int x, int y, int flags, void* param);
Mat_<Vec3b> draw_object_projections(pair<vector<int>, vector<int>>& projections);
void practical_work_1_c();

// Practical work 2: Create a new processing function which takes as input a labeled image and keeps in the
// output image only the objects that:
// a. have their area < TH_area
// b. have a specific orientation phi, where phi_LOW < phi < phi_HIGH
// where TH_area, phi_LOW, phi_HIGH are given by the user
Mat_<uchar> processing_function(Mat_<uchar>& image, int TH_area, int phi_LOW, int phi_HIGH);
void practical_work_2();

int main() {

    //practical_work_1_a();
    //practical_work_1_b();
    //practical_work_1_c();
    practical_work_2();

    return 0;
}

vector<Point> find_object_pixels(Mat_<uchar>& image, int label) {
    vector<Point> result;

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (image(i, j) == label) {
                result.push_back(Point(i, j));
            }
        }
    }

    return result;
}

int compute_object_area(vector<Point>& object_pixels) {
    return object_pixels.size();
}

Point2d compute_object_center_of_mass(vector<Point>& object_pixels) {
    Point2d result(0, 0);

    for (int i = 0; i < object_pixels.size(); i++) {
        result.x += object_pixels[i].x;
        result.y += object_pixels[i].y;
    }

    result.x /= object_pixels.size();
    result.y /= object_pixels.size();

    return result;
}

double compute_object_axis_of_elongation(vector<Point>& object_pixels, Point2d center_of_mass) {
    double sum_i2j2 = 0;
    double sum_i2 = 0;
    double sum_j2 = 0;

    for (int i = 0; i < object_pixels.size(); i++) {
        double dif_i = object_pixels[i].x - center_of_mass.x;
        double dif_j = object_pixels[i].y - center_of_mass.y;

        sum_i2j2 += dif_i * dif_j;
        sum_i2 += dif_i * dif_i;
        sum_j2 += dif_j * dif_j;
    }

    return 0.5 * (atan2(2.0 * sum_i2j2, (sum_j2 - sum_i2)));
}

int compute_object_perimeter(Mat_<uchar>& image, int label) {
    int result = 0;

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (image(i, j) == label) {
                bool on_contour = false;

                if (i == 0 || i == image.rows - 1 || j == 0 || j == image.cols - 1) {
                    on_contour = true;
                }

                else {
                    if (image(i - 1, j) != label || image(i + 1, j) != label ||
                        image(i, j - 1) != label || image(i, j + 1) != label) {
                        on_contour = true;
                        }
                }

                if (on_contour) {
                    result++;
                }
            }
        }
    }

    return result;
}

double compute_object_thinness(int object_area, int object_perimeter) {
    if (object_perimeter <= 0) {
        return 0.0;
    }

    return 4.0 * CV_PI * object_area / (object_perimeter * object_perimeter);
}

double compute_object_aspect_ratio(vector<Point>& object_pixels) {
    int min_i = object_pixels[0].x;
    int max_i = object_pixels[0].x;

    int min_j = object_pixels[0].y;
    int max_j = object_pixels[0].y;

    for (int i = 1; i < object_pixels.size(); i++) {
        min_i = min(min_i, object_pixels[i].x);
        max_i = max(max_i, object_pixels[i].x);

        min_j = min(min_j, object_pixels[i].y);
        max_j = max(max_j, object_pixels[i].y);
    }

    int diff_i = max_i - min_i + 1;
    int diff_j = max_j - min_j + 1;

    if (diff_i > 0 && diff_j > 0) {
        return (double)diff_i / diff_j;
    }

    return -1.0;
}

pair<vector<int>, vector<int>> compute_object_projections(vector<Point>& object_pixels, int rows, int columns) {
    vector<int> horizontal_object_projection(rows, 0);
    vector<int> vertical_object_projection(columns, 0);
    pair<vector<int>, vector<int>> result(horizontal_object_projection, vertical_object_projection);

    for (int i = 0; i < object_pixels.size(); i++) {
        horizontal_object_projection[object_pixels[i].x]++;
        vertical_object_projection[object_pixels[i].y]++;
    }

    return result;
}

void onMouse_1a(int event, int x, int y, int flags, void* param) {
    if (event != EVENT_LBUTTONDOWN) {
        return;
    }

    Mat_<uchar>* image_ptr = (Mat_<uchar>*)param;
    Mat_<uchar> image = *image_ptr;

    int label = image(y, x);
    vector<Point> object_pixels = find_object_pixels(image, label);
    int object_area = compute_object_area(object_pixels);
    Point2d object_center_of_mass = compute_object_center_of_mass(object_pixels);
    int object_perimeter = compute_object_perimeter(image, label);
    double object_thinness = compute_object_thinness(object_area, object_perimeter);
    double object_aspect_ratio = compute_object_aspect_ratio(object_pixels);
    double object_axis_of_elongation = compute_object_axis_of_elongation(object_pixels, object_center_of_mass);

    cout << "Label: " << label << endl;
    cout << "Area: " << object_area << endl;
    cout << "Center of mass: " << object_center_of_mass << endl;
    cout << "Axis of elongation: " << object_axis_of_elongation << endl;
    cout << "Perimeter: " << object_perimeter << endl;
    cout << "Thinness: " << object_thinness << endl;
    cout << "Aspect ratio: " << object_aspect_ratio << endl;

    cout << endl << endl;
}

void practical_work_1_a() {
    Mat_<uchar> image = imread("Images/trasaturi_geom.bmp", IMREAD_GRAYSCALE);

    namedWindow("1a", WINDOW_AUTOSIZE);
    imshow("1a", image);

    setMouseCallback("1a", onMouse_1a, &image);

    waitKey(0);
    destroyWindow("1a");
}

void onMouse_1b(int event, int x, int y, int flags, void* param) {
    if (event != EVENT_LBUTTONDOWN) {
        return;
    }

    Mat_<uchar>* image_ptr = (Mat_<uchar>*)param;
    Mat_<uchar> image = *image_ptr;

    int label = image(y, x);
    Mat_<Vec3b> clone = draw_object_contour(image, label);
    vector<Point> object_pixels = find_object_pixels(image, label);
    Point2d object_center_of_mass = compute_object_center_of_mass(object_pixels);
    double object_axis_of_elongation = compute_object_axis_of_elongation(object_pixels, object_center_of_mass);

    Point center((int)round(object_center_of_mass.y), (int)round(object_center_of_mass.x));
    circle(clone, center, 5, Scalar(0, 0, 255), -1);

    double length = 25.0;

    Point p1(
        (int)round(object_center_of_mass.y - length * cos(object_axis_of_elongation)),
        (int)round(object_center_of_mass.x - length * sin(object_axis_of_elongation))
    );

    Point p2(
        (int)round(object_center_of_mass.y + length * cos(object_axis_of_elongation)),
        (int)round(object_center_of_mass.x + length * sin(object_axis_of_elongation))
    );

    line(clone, p1, p2, Scalar(255, 0, 0), 2);

    imshow("1b clone", clone);

    waitKey(0);
}

Mat_<Vec3b> draw_object_contour(Mat_<uchar>& image, int label) {
    Mat_<Vec3b> result;

    cvtColor(image, result, COLOR_GRAY2BGR);

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (image(i, j) == label) {
                bool on_contour = false;

                if (i == 0 || i == image.rows - 1 || j == 0 || j == image.cols - 1) {
                    on_contour = true;
                }

                else {
                    if (image(i-1, j) != label || image(i+1, j) != label ||
                        image(i, j-1) != label || image(i, j+1) != label) {
                        on_contour = true;
                        }
                }

                if (on_contour) {
                    result(i, j) = Vec3b(0, 255, 0);
                }
            }
        }
    }

    return result;
}

void practical_work_1_b() {
    Mat_<uchar> image = imread("Images/trasaturi_geom.bmp", IMREAD_GRAYSCALE);

    namedWindow("1b", WINDOW_AUTOSIZE);
    imshow("1b", image);

    setMouseCallback("1b", onMouse_1b, &image);

    waitKey(0);
    destroyWindow("1b");
}

void onMouse_1c(int event, int x, int y, int flags, void* param) {
    if (event != EVENT_LBUTTONDOWN) {
        return;
    }

    Mat_<uchar>* image_ptr = (Mat_<uchar>*)param;
    Mat_<uchar> image = *image_ptr;
    Mat_<Vec3b> clone;

    cvtColor(image, clone, COLOR_GRAY2BGR);

    int label = image(y, x);
    vector<Point> object_pixels = find_object_pixels(image, label);
    pair<vector<int>, vector<int>> projections = compute_object_projections(object_pixels, image.rows, image.cols);
    Mat_<Vec3b> projection_image = draw_object_projections(projections);

    imshow("1c clone", projection_image);

    waitKey(0);
}

Mat_<Vec3b> draw_object_projections(pair<vector<int>, vector<int>>& projections) {
    Mat_<Vec3b> projection_image(500, 500, Vec3b(255, 255, 255));

    vector<int> horizontal_projection = projections.first;
    vector<int> vertical_projection = projections.second;

    int max_h = *max_element(horizontal_projection.begin(), horizontal_projection.end());
    int max_v = *max_element(vertical_projection.begin(), vertical_projection.end());

    for (int i = 0; i < horizontal_projection.size(); i++) {
        if (i < projection_image.rows) {
            int len = (max_h > 0) ? (int)(((double)horizontal_projection[i] / max_h) * (projection_image.cols - 1)) : 0;
            Point p1(0, i);
            Point p2(len, i);
            line(projection_image, p1, p2, Scalar(0, 0, 255), 1);
        }
    }

    for (int j = 0; j < vertical_projection.size(); j++) {
        if (j < projection_image.cols) {
            int len = (max_v > 0) ? (int)(((double)vertical_projection[j] / max_v) * (projection_image.rows - 1)) : 0;
            Point p1(j, projection_image.rows - 1);
            Point p2(j, projection_image.rows - 1 - len);
            line(projection_image, p1, p2, Scalar(255, 0, 0), 1);
        }
    }

    return projection_image;
}

void practical_work_1_c() {
    Mat_<uchar> image = imread("Images/trasaturi_geom.bmp", IMREAD_GRAYSCALE);

    namedWindow("1c", WINDOW_AUTOSIZE);
    imshow("1c", image);

    setMouseCallback("1c", onMouse_1c, &image);

    waitKey(0);
    destroyWindow("1c");
}

Mat_<uchar> processing_function(Mat_<uchar>& image, int TH_area, int phi_LOW, int phi_HIGH) {
    Mat_<uchar> result = image.clone();

    for (int label = 1; label < 256; label++) {
        vector<Point> object_pixels = find_object_pixels(result, label);
        int object_area = compute_object_area(object_pixels);
        Point2d object_center_of_mass = compute_object_center_of_mass(object_pixels);
        double axis_of_elongation = compute_object_axis_of_elongation(object_pixels, object_center_of_mass);

        if (axis_of_elongation < 0) {
            axis_of_elongation += 2 * CV_PI;
        }

        if (object_area >= TH_area || axis_of_elongation < phi_LOW || axis_of_elongation > phi_HIGH) {
            for (int i = 0; i < result.rows; i++) {
                for (int j = 0; j < result.cols; j++) {
                    result(i, j) = 255;
                }
            }
        }
    }

    return result;
}

void practical_work_2() {
    Mat_<uchar> image = imread("Images/linie_oblica.bmp", IMREAD_GRAYSCALE);

    int TH_area = 100000;
    double phi_LOW = 0.0;
    double phi_HIGH = 1.0/4.0 * CV_PI;

    Mat_<uchar> result = processing_function(image, TH_area, phi_LOW, phi_HIGH);

    imshow("2", image);
    imshow("2 clone", result);

    waitKey(0);
}