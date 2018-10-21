#include "OpenCV.h"

using namespace std;
using namespace cv;

string path = "../../Data/";
string lennaImage = "Lenna_Images/Lenna.png";
string roadImage = "Lane_Detection_Images/solidYellowLeft.jpg";
string sudokuImage = "sudoku-original.jpg";
string roadVideo = "Lane_Detection_Videos/solidWhiteRight.mp4";

int main(void) {
    Mat roadBGR = imageRead(path + roadImage, IMREAD_COLOR);
    imageShow("roadBGR", roadBGR);

    Mat translateRoad_50_110 = translateImage(roadBGR, 50, 110);
    imageShow("translateRoad_50_110", translateRoad_50_110);

    Mat translateRoad_m50_m110 = translateImage(roadBGR, -50, -110);
    imageShow("translateRoad_m50_m110", translateRoad_m50_m110);

    Mat rotateRoad30 = rotateImage(roadBGR, 30, 0);
    imageShow("rotateRoad30", rotateRoad30);

    Mat rotateRoad45 = rotateImage(roadBGR, 45, 0);
    imageShow("rotateRoad45", rotateRoad45);

    Mat rotateRoad60 = rotateImage(roadBGR, 60, 1);
    imageShow("rotateRoad60", rotateRoad60);

    Mat rotateRoad90 = rotateImage(roadBGR, 90, 1);
    imageShow("rotateRoad90", rotateRoad90);

    Mat resize_test1 = resizeImage(roadBGR, Size(), 1.0, 2.0, INTER_LINEAR);
    imageShow("resize_test1", resize_test1);

    Mat resize_test1_2 = resizeImage(roadBGR, Size(roadBGR.cols, roadBGR.rows*2.0), 0.0, 0.0, INTER_LINEAR);
    imageShow("resize_test1_2", resize_test1_2);

    Mat resize_test2 = resizeImage(roadBGR, Size(), 2.0, 1.0, INTER_LINEAR);
    imageShow("resize_test2", resize_test2);

    Mat resize_test2_2 = resizeImage(roadBGR, Size(roadBGR.cols*2.0, roadBGR.rows), 0.0, 0.0, INTER_LINEAR);
    imageShow("resize_test2_2", resize_test2_2);

    Point2f srcTri[3];
    Point2f dstTri[3];
    srcTri[0] = Point2f( 0,0 );
    srcTri[1] = Point2f( roadBGR.cols - 1, 0 );
    srcTri[2] = Point2f( 0, roadBGR.rows - 1 );

    dstTri[0] = Point2f( roadBGR.cols*0.0, roadBGR.rows*0.33 );
    dstTri[1] = Point2f( roadBGR.cols*0.85, roadBGR.rows*0.25 );
    dstTri[2] = Point2f( roadBGR.cols*0.15, roadBGR.rows*0.7 );
    Mat transform3p = transformImage(roadBGR, srcTri, dstTri);
    imageShow("transform3p", transform3p);


    Point2f srcQuad[4];
    int w = roadBGR.cols;
    int h = roadBGR.rows;
    float ratio_x1, ratio_y1, ratio_x2, ratio_y2;
    ratio_x1 = 0.4;
    ratio_y1 = 0.65;
    ratio_x2 = 0.0;
    ratio_y2 = 0.9;
    srcQuad[0] = Point2f(w * ratio_x1 , h * ratio_y1);
    srcQuad[1] = Point2f(w * (1.0 - ratio_x1) , h * ratio_y1);
    srcQuad[2] = Point2f(w * (1.0 - ratio_x2), h * ratio_y2);
    srcQuad[3] = Point2f(w * ratio_x2, h * ratio_y2);
    Point2f dstQuad[4];
    dstQuad[0] = Point2f(0.0 , 0.0);
    dstQuad[1] = Point2f(w , 0.0);
    dstQuad[2] = Point2f(w, h);
    dstQuad[3] = Point2f(0, h);

    Mat transform4p = perspectiveTransformImage(roadBGR, srcQuad, dstQuad);
    imageShow("transform4p", transform4p);
    destroyAllWindows();

    return 0;
}

#ifdef OPENCV_H_
Mat perspectiveTransformImage(Mat image, Point2f srcpoints[4], Point2f dstpoints[4]) {
    Mat M = getPerspectiveTransform(srcpoints, dstpoints);
    Mat result;
    warpPerspective(image, result, M, image.size());
    return result;
}
Mat transformImage(Mat image, Point2f srcpoints[3], Point2f dstpoints[3]) {
    Mat warpMat(2,3,CV_32FC1);
    Mat warpResult;
    warpResult = Mat::zeros(image.rows, image.cols, image.type());
    warpMat = getAffineTransform( srcpoints, dstpoints );
    warpAffine( image, warpResult, warpMat, warpResult.size());
    return warpResult;
}
Mat resizeImage(Mat image, Size size, double cx, double cy, int interpolation) {
    Mat result;
    resize(image, result, size, cx, cy, interpolation);
    return result;
}
Mat rotateImage(Mat image, double angle, int flag) {
    Mat result;
    Point2f center((image.cols-1)/2.0, (image.rows-1)/2.0);
    Mat rot = getRotationMatrix2D(center, angle, 1.0);
    if( flag == 0 ) {
        warpAffine(image, result, rot, image.size());
    }
    else if( flag == 1 ) {
        Rect2f bbox = RotatedRect(Point2f(), image.size(), angle).boundingRect2f();
        rot.at<double>(0, 2) += bbox.width / 2.0 - image.cols / 2.0;
        rot.at<double>(1, 2) += bbox.height / 2.0 - image.rows / 2.0;
        warpAffine(image, result, rot, bbox.size());
    }
    return result;
}
Mat translateImage(Mat image, double tx, double ty) {
    Mat result;
    double matData[] = {1, 0, tx, 0, 1, ty};
    Mat M(2, 3, CV_64FC1, matData);
    warpAffine(image, result, M, image.size());
    return result;
}
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
    Ptr<CLAHE> clahe = createCLAHE();
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
#endif

#ifdef OPENCV_VIDEO_H_
void videoProcessor::run() {
	Mat frameInput;
	Mat frameOutput;
	if(!isOpened())
		return;
	stop = false;
	while(!isStopped()){
		if(!readNextFrame(frameInput))
			break;
        if(videoNameInput.length() != 0)
            imshow(videoNameInput, frameInput);

        frameOutput = videoProcess(frameInput);

        if(videoWritePath.length() != 0)
            writeNextFrame(frameOutput);
        if(videoNameOutput.length() != 0)
            imshow(videoNameOutput, frameOutput);
        frameNumber++;
        if(delay >= 0 && waitKey(delay) >= 0)
            stopIt();
        if(frameToStop >= 0 && getFrameNumber() == frameToStop)
            stopIt();
	}
}
bool videoProcessor::setInput(string videoPath) {
	videoReadPath = videoPath;
	frameNumber = 0;
	capture.release();
	return capture.open(videoReadPath);
}
bool videoProcessor::setOutput(string videoPath, int codec, double framerate, bool isColor) {
	videoWritePath = videoPath;
	extension.clear();
	if(framerate == 0.0) {
		framerate = getFrameRate();
	}
	char c[4];
	if(codec == 0) {
		codec = getCodec(c);
	}
	return writer.open(videoWritePath, codec, framerate, getFrameSize(), isColor);
}
void videoProcessor::showInput(string windowName) {
	videoNameInput = windowName;
	namedWindow(videoNameInput, CV_WINDOW_NORMAL);
}
void videoProcessor::showOutput(string windowName) {
	videoNameOutput = windowName;
	namedWindow(videoNameOutput, CV_WINDOW_NORMAL);
}
double videoProcessor::getFrameRate() {
	return frameRate = capture.get(CV_CAP_PROP_FPS);
}
Size videoProcessor::getFrameSize() {
	return frameSize = Size((int)capture.get(CAP_PROP_FRAME_WIDTH),
        (int)capture.get(CAP_PROP_FRAME_HEIGHT));
}
void videoProcessor::stopIt() {
	stop = true;
}
bool videoProcessor::isStopped() {
	return stop;
}
bool videoProcessor::isOpened() {
	return capture.isOpened();
}
void videoProcessor::setDelay(int d) {
	delay = d;
}
bool videoProcessor::readNextFrame(Mat& frame) {
	return capture.read(frame);
}
void videoProcessor::writeNextFrame(Mat& frame) {
	if(extension.length()) {
		stringstream stream;
		stream << videoWritePath << setfill('0') << setw(digits)
		<< currentIndex++ << extension;
	imwrite(stream.str(), frame);
	}
	else {
		writer.write(frame);
	}
}
long videoProcessor::getFrameNumber() {
	long frameNumber = static_cast<long>(capture.get(CV_CAP_PROP_POS_FRAMES));
	return frameNumber;
}
void videoProcessor::stopAtFrameNumber(long frame) {
	frameToStop = frame;
}
int videoProcessor::getCodec(char codec[4]) {
	union {
		int value;
		char code[4];
	} returned;
	returned.value = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));
	int i;
	for( i = 0 ; i < 4 ; i++ ) {
		codec[i] = returned.code[i];
	}
	return returned.value;
}
#endif
