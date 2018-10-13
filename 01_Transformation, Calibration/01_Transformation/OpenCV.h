#pragma once
#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

string path = "../../Data/";
string lennaImage = "Lenna_Images/Lenna.png";
string roadImage = "Lane_Detection_Images/solidYellowCurve.jpg";

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

Point interpolate(Point p1, Point p2, int y) {
    int result;
    result = double((y-p1.y)*(p2.x-p1.x))/double(p2.y-p1.y)+p1.x;
    return Point(result,y);
}
Point MeanVector(vector<Point> v) {
    int sum1 = 0;
    int sum2 = 0;
    int temp = 0;
    for (temp = 0 ; temp < v.size();temp++) {
        sum1 += v[temp].x;
        sum2 += v[temp].y;
    }
    int p_1 = (double)sum1/(double)temp;
    int p_2 = (double)sum2/(double)temp;
    return Point(p_1,p_2);
}
bool comp(Point a, Point b) {
    return (a.x>b.x);
}
Point MedianVector(vector<Point> v) {
    size_t size = v.size();
    sort(v.begin(), v.end(), comp);
    if (size % 2 == 0)
    {
        int x = (v[size/2-1].x+v[size/2].x)/2;
        int y = (v[size/2-1].y+v[size/2].y)/2;
        return Point(x,y);
    }
    else
    {
        return v[size / 2];
    }
}
Mat drawLanes(Mat image, vector<Vec4i> lines) {
    int w = image.cols;
	int h = image.rows;
	Mat result = Mat::zeros(h,w,CV_8UC3);
	size_t i;
    vector<Point> left_x, left_y, right_x, right_y;
	for( i = 0; i < lines.size(); i++ ) {
		Vec4i l = lines[i];
        int x1, x2, y1, y2;
        x1 = l[0];
        y1 = l[1];
        x2 = l[2];
        y2 = l[3];
        double slope = (double)(y2-y1)/(double)(x2-x1);
        if (abs(slope) < 0.5 ) {
            continue;
        }
        if ( slope <= 0 ) {
            left_x.push_back(Point(x1, x2));
            left_y.push_back(Point(y1, y2));
        }
        else {
            right_x.push_back(Point(x1, x2));
            right_y.push_back(Point(y1, y2));
        }
 	}
    Point left_xp = MeanVector(left_x);
    Point right_xp = MeanVector(right_x);
    Point left_yp = MeanVector(left_y);
    Point right_yp = MeanVector(right_y);

    int min_y = image.rows * 0.6;
    int max_y = image.rows;

    Point left_min = interpolate(Point(left_xp.x, left_yp.x), Point(left_xp.y, left_yp.y), min_y);
    Point left_max = interpolate(Point(left_xp.x, left_yp.x), Point(left_xp.y, left_yp.y), max_y);
    Point right_min = interpolate(Point(right_xp.x, right_yp.x), Point(right_xp.y, right_yp.y), min_y);
    Point right_max = interpolate(Point(right_xp.x, right_yp.x), Point(right_xp.y, right_yp.y), max_y);

    line( result, left_min, left_max, Scalar(0,0,255), 3, CV_AA);
    line( result, right_min, right_max, Scalar(0,0,255), 3, CV_AA);

	return result;
}
vector<Vec4i> HoughLinesP(Mat image, double rho, double theta, int threshold, double minLineLength, double maxLineGap){
	vector<Vec4i> lines;
	HoughLinesP(image, lines, rho, theta, threshold, minLineLength, maxLineGap );
	return lines;
}
Mat drawHoughLinesP(Mat image, double rho, double theta, int threshold, double minLineLength, double maxLineGap){
	vector<Vec4i> lines;
	HoughLinesP(image, lines, rho, theta, threshold, minLineLength, maxLineGap );
	Mat result = Mat::zeros(image.rows, image.cols, CV_8UC3);
	result = drawLines(result, lines);
	return result;
}
Mat drawLines(Mat image, vector<Vec4i> lines) {
    int w = image.cols;
	int h = image.rows;
	Mat result = Mat::zeros(h,w,CV_8UC3);
	size_t i;
	for( i = 0; i < lines.size(); i++ ) {
		Vec4i l = lines[i];
		line( result, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
 	}
	return result;
}
vector<Vec4i> detectHoughLinesP(Mat image, double rho, double theta, int threshold, double minLineLength, double maxLineGap){
	vector<Vec4i> lines;
	HoughLinesP(image, lines, rho, theta, threshold, minLineLength, maxLineGap );
	return lines;
}
Mat weightedSum(Mat foregroundImage, Mat backgroundImage, double alpha, double beta, double gamma) {
	Mat result = foregroundImage.clone();
	if(backgroundImage.channels() == 3 && foregroundImage.channels()==1) {
		vector<Mat> channels = splitChannel(backgroundImage);
		addWeighted(result, alpha, channels[2], beta, gamma, result);
		result = mergeChannel(channels[0], channels[1], result);
	}
	else if(backgroundImage.channels() == 1 && foregroundImage.channels()==1) {
		addWeighted(result, alpha, backgroundImage, beta, gamma, result);
	}
	else if(backgroundImage.channels() == 3 && foregroundImage.channels()==3) {
		addWeighted(result, alpha, backgroundImage, beta, gamma, result);
	}

	return result;
}
Mat trapezoidalROI(Mat image, double ratio_x1, double ratio_y1, double ratio_x2, double ratio_y2, int lineType) {
	Point ROI_points[1][4];
	int w = image.cols;
	int h = image.rows;
	Mat result = Mat::zeros(h,w,CV_8UC1);

	ROI_points[0][0] = Point(w * ratio_x1 , h * ratio_y1);
	ROI_points[0][1] = Point(w * (1.0 - ratio_x1) , h * ratio_y1);
	ROI_points[0][2] = Point(w * (1.0 - ratio_x2), h * ratio_y2);
	ROI_points[0][3] = Point(w * ratio_x2, h * ratio_y2);

	const Point* ppt[1] = { ROI_points[0] };
  	int npt[] = { 4 };
	fillPoly(result, ppt, npt, 1, 255, lineType);
	bitwise_and(result, image, result);
	return result;
}
Mat cannyEdge(Mat image, double threshold1, double threshold2, int apertureSize, bool L2gradient) {
	Mat result;
	result = bilateralFilter(image);
	Canny(result, result, threshold1, threshold2, apertureSize, L2gradient);
	return result;
}
Mat clippedHistogramEqualize(Mat image, double clipLimit, Size tileGridSize) {
	Mat result;
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(clipLimit);
	clahe->setTilesGridSize(tileGridSize);
	clahe->apply(image, result);
	return result;
}
Mat histogramEqualize(Mat image) {
	Mat result;
	equalizeHist(image, result);
	return result;
}
Mat bilateralFilter(Mat image, int d, double sigmaColor, double sigmaSpace, int borderType) {
	Mat result;
	bilateralFilter(image, result, d, sigmaColor, sigmaSpace, borderType);
	return result;
}
Mat medianBlur(Mat image, int kernelSize) {
	Mat result;
	medianBlur(image, result, kernelSize);
	return result;
}
Mat gaussianBlur(Mat image, Size kernelSize, double sigmaX, double sigmaY, int borderType) {
	Mat result;
	GaussianBlur(image, result, kernelSize, sigmaX, sigmaY, borderType);
	return result;
}
Mat meanBlur(Mat image, Size kernelSize, Point anchor, int borderType) {
	Mat result;
	blur(image, result, kernelSize, anchor, borderType);
	return result;
}
Mat addSaltAndPepper(Mat image, double noiseRatio) {
	Mat result = image.clone();
	int i;
	int rows = result.rows;
	int cols = result.cols;
	int ch = result.channels();
	int num_of_noise_pixels = (int)((double)(rows*cols*ch)*noiseRatio);
	for ( i = 0 ; i < num_of_noise_pixels ; i++ ){
		int r = rand() % rows;
		int c = rand() % cols;
		int _ch = rand() % ch;

		uchar* pixel = result.ptr<uchar>(r)+(c*ch)+_ch;
		*pixel = (rand()%2==1)?255:0;
	}
	return result;
}
Mat mergeChannel(Mat ch1, Mat ch2, Mat ch3) {
	Mat result;
	vector<Mat> channels;
	channels.push_back(ch1);
	channels.push_back(ch2);
	channels.push_back(ch3);
	merge(channels, result);
	return result;
}
Mat mergeChannel(vector<Mat> channels) {
	Mat result;
	merge(channels,result);
	return result;
}
vector<Mat> splitChannel(Mat image) {
	vector<Mat> channels;
	split(image, channels);
	return channels;
}
Mat convertColor(Mat image, int flag) {
	Mat result;
	cvtColor(image, result, flag);
	return result;
}
void showChannels(string imageName, Mat image) {
	if(image.channels() != 1) {
		vector<Mat> channels;
		channels = splitChannel(image);
		imageShow(imageName+"_0", channels[0]);
		imageShow(imageName+"_1", channels[1]);
		imageShow(imageName+"_2", channels[2]);
	}
	else if(image.channels() == 1){
		imageShow("GrayScale", image);
	}
	return;
}
void saveChannels(string path, string ImageName, string channelname, Mat image) {
	if(image.channels() != 1) {
		vector<Mat> channels;
		channels = splitChannel(image);
		int i;
		for( i = 0 ; i < 3 ; i++) {
			imwrite(path+ImageName+"_"+channelname[i]+".bmp", channels[i]);
		}
	}
	else {
		imwrite(path+ImageName+".bmp", image);
	}
	return;
}
Mat adaptiveThresholdByCV(Mat image, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C ) {
	Mat result;
	adaptiveThreshold(image, result, maxValue, adaptiveMethod, thresholdType, blockSize, C);
	return result;
}
Mat thresholdByCV(Mat image, double thresh, double maxval, int type) {
	Mat result;
	threshold(image, result, thresh, maxval, type);
	return result;
}
Mat thresholdByAt(Mat image, uchar thresh) {
	int i,j;
	Mat result = image.clone();
	if(result.type() == CV_8UC1) {
		for( j = 0 ; j < result.rows ; j++) {
			for ( i = 0 ; i < result.cols ; i++ ) {
				uchar value = result.at<uchar>(j,i);
				value = value > thresh ? 255 : 0;
				result.at<uchar>(j,i) = value;
			}
		}
	}
	else if(result.type() == CV_8UC3) {
		for( j = 0 ; j < result.rows ; j++) {
			for ( i = 0 ; i < result.cols ; i++ ) {
				Vec3b value = result.at<Vec3b>(j, i);
				int c;
				for(c = 0 ; c < 3 ; c++) {
					value[c] = value[c] > thresh ? 255 : 0;
				}
				result.at<Vec3b>(j,i) = value;
			}
		}
	}
	return result;
}
Mat thresholdByPtr(Mat image, uchar thresh) {
	int i,j;
	Mat result = image.clone();
	if(result.type() == CV_8UC1) {
		for( j = 0 ; j < result.rows ; j++) {
			uchar* image_pointer = result.ptr<uchar>(j);
			for ( i = 0 ; i < result.cols ; i++ ) {
				uchar value = image_pointer[i];
				value = value > thresh ? 255 : 0;
				image_pointer[i] = value;
			}
		}
	}
	else if(result.type() == CV_8UC3) {
		for( j = 0 ; j < result.rows ; j++) {
			uchar* image_pointer = result.ptr<uchar>(j);
			for ( i = 0 ; i < result.cols ; i++ ) {
				int c;
				for(c = 0 ; c < 3 ; c++) {
					uchar value = image_pointer[i*3+c];
					value = value > thresh ? 255 : 0;
					image_pointer[i*3+c] = value;
				}
			}
		}
	}
	return result;
}
Mat thresholdByData(Mat image, uchar thresh) {
	int i,j;
	Mat result = image.clone();
	uchar *image_data = result.data;
	if(result.type() == CV_8UC1) {
		for( j = 0 ; j < result.rows ; j++) {
			for ( i = 0 ; i < result.cols ; i++ ) {
				uchar value = image_data[j * result.cols + i];
				value = value > thresh ? 255 : 0;
				image_data[j * result.cols + i] = value;
			}
		}
	}
	else if(result.type() == CV_8UC3) {
		for( j = 0 ; j < result.rows ; j++) {
			for ( i = 0 ; i < result.cols ; i++ ) {
				int c;
				for(c = 0 ; c < 3 ; c++) {
					uchar value = image_data[(j * result.cols + i) * 3 + c];
					value = value > thresh ? 255 : 0;
					image_data[(j * result.cols + i) * 3 + c] = value;
				}
			}
		}
	}
	return result;
}
Mat imageRead(string openPath, int flag) {
	Mat image = imread(openPath, flag);
	if(image.empty()) {
		cout<<"Image Not Opened"<<endl;
		cout<<"Program Abort"<<endl;
		exit(1);
	}
	else {
		cout<<"Image Opened"<<endl;
		return image;
	}
}
void imageShow(string imageName, Mat image, int flag) {
	namedWindow(imageName, flag);
	cout << "Display "<< imageName << " Channel: " << image.channels() << endl;
    imshow(imageName, image);
	waitKey(0);
}
