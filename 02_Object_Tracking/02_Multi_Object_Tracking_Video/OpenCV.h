#ifndef OPENCV_H_
#define OPENCV_H_
#endif //OPENCV_H_
#pragma once
#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

void getRandomColors(vector<Scalar>& colors, int numColors);
Mat undistortImage(Mat image);
Mat perspectiveTransformImage(Mat image, Point2f srcpoints[4], Point2f dstpoints[4]);
Mat transformImage(Mat image, Point2f srcpoints[3], Point2f dstpoints[3]);
Mat resizeImage(Mat image, Size size = Size(), double cx = 1.0, double cy = 1.0, int interpolation = INTER_LINEAR);
Mat rotateImage(Mat image, double angle = 45, int flag = 0);
Mat translateImage(Mat image, double tx = 0, double ty = 0);
Point interpolate(Point p1, Point p2, int y);
Point MeanVector(vector<Point> v);
Point MedianVector(vector<Point> v);
Mat drawLanes(Mat image, vector<Vec4i> lines);
vector<Vec4i> HoughLinesP(Mat image, double rho = 1.0, double theta = CV_PI / 60.0, int threshold = 20, double minLineLength=10, double maxLineGap=50);
Mat drawHoughLinesP(Mat image, double rho = 1.0, double theta = CV_PI / 60.0, int threshold = 20, double minLineLength=10, double maxLineGap=50);
Mat drawLines(Mat image, vector<Vec4i> lines);
vector<Vec4i> detectHoughLinesP(Mat image, double rho = 2.0, double theta = CV_PI / 180.0, int threshold = 15, double minLineLength=40, double maxLineGap=20);
Mat weightedSum(Mat foregroundImage, Mat backgroundImage, double alpha = 1.0, double beta = 0.8, double gamma = 0.0);
Mat trapezoidalROI(Mat image, double ratio_x1 = 0.4, double ratio_y1 = 0.6, double ratio_x2 = 0.1, double ratio_y2 = 1.0, int lineType=8);
Mat cannyEdge(Mat image, double threshold1 = 50.0, double threshold2 = 150.0, int apertureSize = 3, bool L2gradient = false);
Mat clippedHistogramEqualize(Mat image, double clipLimit = 40.0, Size tileGridSize = Size(4,4));
Mat histogramEqualize(Mat image);
Mat bilateralFilter(Mat image, int d = -1, double sigmaColor = 10.0, double sigmaSpace = 10.0, int borderType = BORDER_DEFAULT);
Mat medianBlur(Mat image, int kernelSize = 5);
Mat gaussianBlur(Mat image, Size kernelSize = Size(5,5), double sigmaX = 2, double sigmaY = 0, int	borderType = BORDER_DEFAULT);
Mat meanBlur(Mat image, Size kernelSize = Size(5,5), Point anchor = Point(-1,-1), int borderType = BORDER_DEFAULT);
Mat addSaltAndPepper(Mat image, double noiseRatio = 0.01);
Mat mergeChannel(Mat ch1, Mat ch2, Mat ch3);
Mat mergeChannel(vector<Mat> channels);
vector<Mat> splitChannel(Mat image);
Mat convertColor(Mat image, int flag = CV_BGR2GRAY);
void showChannels(string imageName, Mat image);
void saveChannels(string path, string ImageName, string channelname, Mat image);
Mat adaptiveThresholdByCV(Mat image, double maxValue = 255, int adaptiveMethod = ADAPTIVE_THRESH_GAUSSIAN_C, int thresholdType = THRESH_BINARY, int blockSize = 3, double C = 0);
Mat thresholdByCV(Mat image, double thresh = 128, double maxval = 255, int type = THRESH_BINARY);
Mat thresholdByAt(Mat image, uchar thresh = 128);
Mat thresholdByPtr(Mat image, uchar thresh = 128);
Mat thresholdByData(Mat image, uchar thresh = 128);
Mat imageRead(string openPath, int flag = IMREAD_UNCHANGED);
void imageShow(string imageName, Mat image, int flag = CV_WINDOW_NORMAL);
