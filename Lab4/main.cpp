#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include <unordered_set>

using namespace cv;
using namespace std;

// Practical work 1: For a specific object in a labeled image selected by a mouse click, compute the object’s
// area, center of mass, axis of elongation, perimeter, thinness ratio, aspect ratio and projections.
vector<Point> find_object_pixels(Mat_<uchar> image, int x, int y);
int compute_object_area(vector<Point> object_pixels);
Point2d compute_object_center_of_mass(vector<Point> object_pixels);
double compute_object_axis_of_elongation(vector<Point> object_pixels, Point2d center_of_mass);
int compute_object_perimeter(Mat_<uchar> image, vector<Point> object_pixels);
double compute_object_thinness(int object_area, int object_perimeter);
double compute_object_aspect_ratio(vector<Point> object_pixels);
pair<vector<int>, vector<int>> compute_object_projections(Mat_<uchar> image, vector<Point> object_pixels);

// a. Display the results in the standard output
// b. In a separate image (source image clone):
// Draw the contour points of the selected object
// Display the center of mass of the selected object
// Display the axis of elongation of the selected object by using the line
// function from OpenCV.
// c. Compute and display the projections of the selected object in a separate image
// (source image clone).
void onMouse(int event, int x, int y, int flags, void* param);
Mat_<Vec3b> draw_object_contour(Mat_<uchar> image, vector<Point> object_pixels, Vec3b color);
Mat_<Vec3b> draw_object_projections(pair<vector<int>, vector<int>> object_projections, pair<Vec3b, Vec3b> projections_colors);
void practical_work_1();

// Practical work 2: Create a new processing function which takes as input a labeled image and keeps in the
// output image only the objects that:
// a. have their area < TH_area
// b. have a specific orientation phi, where phi_LOW < phi < phi_HIGH
// where TH_area, phi_LOW, phi_HIGH are given by the user.
Mat_<uchar> processing_function(Mat_<uchar> image, int TH_area, int phi_LOW, int phi_HIGH);
vector<Point> find_object_pixels(Mat_<uchar> image, int label);
void practical_work_2();

int main() {

    //practical_work_1();
    //practical_work_2();

    return 0;
}

vector<Point> find_object_pixels(Mat_<uchar> image, int x, int y) {
    vector<Point> result;
    int label = image(y, x);
    vector<vector<bool>> visited(image.rows, vector<bool>(image.cols, false));
    queue<Point> q;
    q.push(Point(x, y));
    visited[y][x] = true;

    while (!q.empty()) {
        Point p = q.front();
        q.pop();
        result.push_back(p);

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) {
                    continue;
                }

                int nx = p.x + i;
                int ny = p.y + j;

                if (nx >= 0 && nx < image.cols && ny >= 0 && ny < image.rows && image(ny, nx) == label && !visited[ny][nx]) {
                    q.push(Point(nx, ny));
                    visited[ny][nx] = true;
                }
            }
        }
    }

    return result;
}

int compute_object_area(vector<Point> object_pixels) {
    return object_pixels.size();
}

Point2d compute_object_center_of_mass(vector<Point> object_pixels) {
    Point2d result(0, 0);

    for (int i = 0; i < object_pixels.size(); i++) {
        result.x += object_pixels[i].x;
        result.y += object_pixels[i].y;
    }

    result.x /= object_pixels.size();
    result.y /= object_pixels.size();

    return result;
}

double compute_object_axis_of_elongation(vector<Point> object_pixels, Point2d center_of_mass) {
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

int compute_object_perimeter(Mat_<uchar> image, vector<Point> object_pixels) {
    int result = 0;

    for (const Point& p : object_pixels) {
        bool on_contour = false;

        if (p.x == 0 || p.x == image.rows - 1 || p.y == 0 || p.y == image.cols - 1) {
            on_contour = true;
        }

        else {
            if (image(p.x - 1, p.y) != image(p.x, p.y) || image(p.x + 1, p.y) != image(p.x, p.y) || image(p.x, p.y - 1) != image(p.x, p.y) || image(p.x, p.y + 1) != image(p.x, p.y)) {
                on_contour = true;
            }
        }

        if (on_contour) {
            result++;
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

double compute_object_aspect_ratio(vector<Point> object_pixels) {
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

pair<vector<int>, vector<int>> compute_object_projections(Mat_<uchar> image, vector<Point> object_pixels) {
    vector<int> horizontal_projection(image.rows, 0);
    vector<int> vertical_projection(image.cols, 0);

    for (const Point& p : object_pixels) {
        horizontal_projection[p.y]++;
        vertical_projection[p.x]++;
    }

    return make_pair(horizontal_projection, vertical_projection);
}

void onMouse(int event, int x, int y, int flags, void* param) {
    if (event != EVENT_LBUTTONDOWN) {
        return;
    }

    Mat_<uchar>* image_ptr = (Mat_<uchar>*)param;
    Mat_<uchar> image = *image_ptr;

    int label = image(y, x);
    vector<Point> object_pixels = find_object_pixels(image, x, y);
    Mat_<Vec3b> clone = draw_object_contour(image, object_pixels, Vec3b(0, 0, 255));
    int object_area = compute_object_area(object_pixels);
    Point2d object_center_of_mass = compute_object_center_of_mass(object_pixels);
    int object_perimeter = compute_object_perimeter(image, object_pixels);
    double object_thinness = compute_object_thinness(object_area, object_perimeter);
    double object_aspect_ratio = compute_object_aspect_ratio(object_pixels);
    double object_axis_of_elongation = compute_object_axis_of_elongation(object_pixels, object_center_of_mass);
    pair<vector<int>, vector<int>> object_projections = compute_object_projections(image, object_pixels);
    pair<Vec3b, Vec3b> projections_colors(Vec3b(0, 255, 0), Vec3b(255, 0, 0));
    Mat_<Vec3b> clone_2 = draw_object_projections(object_projections, projections_colors);

    Point center((int)round(object_center_of_mass.x), (int)round(object_center_of_mass.y));
    circle(clone, center, 5, Scalar(0, 0, 255), -1);

    double length = 25.0;

    Point p1(
        (int)round(object_center_of_mass.x - length * cos(object_axis_of_elongation)),
        (int)round(object_center_of_mass.y - length * sin(object_axis_of_elongation))
    );

    Point p2(
        (int)round(object_center_of_mass.x + length * cos(object_axis_of_elongation)),
        (int)round(object_center_of_mass.y + length * sin(object_axis_of_elongation))
    );

    line(clone, p1, p2, Scalar(255, 0, 0), 2);

    imshow("1 clone", clone);
    imshow("1 projections", clone_2);

    cout << "Label: " << label << endl;
    cout << "Area: " << object_area << endl;
    cout << "Center of mass: " << object_center_of_mass << endl;
    cout << "Axis of elongation: " << object_axis_of_elongation << endl;
    cout << "Perimeter: " << object_perimeter << endl;
    cout << "Thinness: " << object_thinness << endl;
    cout << "Aspect ratio: " << object_aspect_ratio << endl;

    cout << endl << endl;
}

Mat_<Vec3b> draw_object_contour(Mat_<uchar> image, vector<Point> object_pixels, Vec3b color) {
    Mat_<Vec3b> result;
    unordered_set<int> object_set;
    cvtColor(image, result, COLOR_GRAY2BGR);

    for (const Point& p : object_pixels) {
        object_set.insert(p.y * image.cols + p.x);
    }

    for (const Point& p : object_pixels) {
        bool on_contour = false;

        if (p.x == 0 || p.x == image.cols - 1 || p.y == 0 || p.y == image.rows - 1) {
            on_contour = true;
        }

        else {
            if (object_set.find((p.y - 1) * image.cols + p.x) == object_set.end() ||
                object_set.find((p.y + 1) * image.cols + p.x) == object_set.end() ||
                object_set.find(p.y * image.cols + (p.x - 1)) == object_set.end() ||
                object_set.find(p.y * image.cols + (p.x + 1)) == object_set.end()) {
                on_contour = true;
                }
        }

        if (on_contour) {
            result(p.y, p.x) = color;
        }
    }

    return result;
}

Mat_<Vec3b> draw_object_projections(pair<vector<int>, vector<int>> object_projections, pair<Vec3b, Vec3b> projections_colors) {
    Mat_<Vec3b> result(500, 500, Vec3b(255, 255, 255));

    vector<int> horizontal_projection = object_projections.first;
    vector<int> vertical_projection = object_projections.second;

    int max_horizontal = *max_element(horizontal_projection.begin(), horizontal_projection.end());
    int max_vertical = *max_element(vertical_projection.begin(), vertical_projection.end());

    for (int x = 0; x < horizontal_projection.size(); x++) {
        int bar_height = (int)(((double)horizontal_projection[x] / max_horizontal) * (500 / 2));

        for (int y = 0; y < bar_height; y++) {
            result(500 - 1 - y, x) = projections_colors.first;
        }
    }

    for (int y = 0; y < vertical_projection.size(); y++) {
        int bar_width = (int)(((double)vertical_projection[y] / max_vertical) * (500 / 2));

        for (int x = 0; x < bar_width; x++) {
            result(y, x) = projections_colors.second;
        }
    }

    return result;
}

void practical_work_1() {
    Mat_<uchar> image = imread("Images/trasaturi_geom.bmp", IMREAD_GRAYSCALE);

    namedWindow("1 initial", WINDOW_AUTOSIZE);
    imshow("1 initial", image);

    setMouseCallback("1 initial", onMouse, &image);

    waitKey(0);

    destroyWindow("1 initial");
}

Mat_<uchar> processing_function(Mat_<uchar> image, int TH_area, int phi_LOW, int phi_HIGH) {
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

vector<Point> find_object_pixels(Mat_<uchar> image, int label) {
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

void practical_work_2() {
    Mat_<uchar> image = imread("Images/linie_oblica.bmp", IMREAD_GRAYSCALE);

    int TH_area = 100000;
    double phi_LOW = 0.0;
    double phi_HIGH = 1.0/2.0 * CV_PI;

    Mat_<uchar> result = processing_function(image, TH_area, phi_LOW, phi_HIGH);

    imshow("2 initial", image);
    imshow("2 clone", result);

    waitKey(0);
}
