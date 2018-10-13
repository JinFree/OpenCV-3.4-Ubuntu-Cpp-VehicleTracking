#include "OpenCV.h"

using namespace std;
using namespace cv;
Mat videoProcessing(Mat Input) {
    Mat output = Input.clone();
    Mat HSV = convertColor(output, CV_BGR2HSV);
    vector<Mat> HSV_HSV, HSV_Equalized;
    HSV_HSV = splitChannel(HSV);
    HSV_Equalized = splitChannel(HSV);
    HSV_Equalized[1] = histogramEqualize(HSV_Equalized[1]);
    HSV_Equalized[2] = histogramEqualize(HSV_Equalized[2]);
    Mat S_E_T_white = thresholdByCV(HSV_Equalized[1], 50, 255, THRESH_TOZERO_INV);
    Mat HSV_S_E_T_W = mergeChannel(HSV_HSV[0], S_E_T_white, HSV_Equalized[2]);
    Mat RGB_HSV_W = convertColor(HSV_S_E_T_W, CV_HSV2BGR);
    Mat W_gray = convertColor(RGB_HSV_W, CV_BGR2GRAY);
    Mat gray_th = thresholdByCV(W_gray, 240, 255, THRESH_BINARY);
    Mat roadROI2 = trapezoidalROI(gray_th, 0.35, 0.6, -0.1, 0.95);
    Mat roadCanny = cannyEdge(roadROI2, 50, 100);
    Mat roadROI = trapezoidalROI(roadCanny, 0.4, 0.65, 0.0, 0.9);
    vector<Vec4i> lines = HoughLinesP(roadROI, 1.0, CV_PI/60.0, 20, 10, 50);
    Mat roadLines = drawLanes(roadROI, lines);
    output = weightedSum(roadLines, output);
    return output;
}
int main(void) {
    string roadImagePath = "../../Data/Lane_Detection_Images"/;
    string file1 = "solidWhiteCurve.jpg";
    string file2 = "solidWhiteRight.jpg";
    string file3 = "solidYellowCurve.jpg";
    string file4 = "solidYellowCurve2.jpg";
    string file5 = "solidYellowLeft.jpg";
    string file6 = "whiteCarLaneSwitch.jpg";

    Mat roadBGR1 = imageRead(roadImagePath + file1, IMREAD_COLOR);
    imageShow("roadBGR1", roadBGR1);

    Mat roadBGR2 = imageRead(roadImagePath + file2, IMREAD_COLOR);
    imageShow("roadBGR2", roadBGR2);

    Mat roadBGR3 = imageRead(roadImagePath + file3, IMREAD_COLOR);
    imageShow("roadBGR3", roadBGR3);

    Mat roadBGR4 = imageRead(roadImagePath + file4, IMREAD_COLOR);
    imageShow("roadBGR4", roadBGR4);

    Mat roadBGR5 = imageRead(roadImagePath + file5, IMREAD_COLOR);
    imageShow("roadBGR5", roadBGR5);

    Mat roadBGR6 = imageRead(roadImagePath + file6, IMREAD_COLOR);
    imageShow("roadBGR6", roadBGR6);

    Mat Output1 =  videoProcessing(roadBGR1);
    imageShow("Output1", Output1);

    Mat Output2 =  videoProcessing(roadBGR2);
    imageShow("Output2", Output2);

    Mat Output3 =  videoProcessing(roadBGR3);
    imageShow("Output3", Output3);

    Mat Output4 =  videoProcessing(roadBGR4);
    imageShow("Output4", Output4);

    Mat Output5 =  videoProcessing(roadBGR5);
    imageShow("Output5", Output5);

    Mat Output6 =  videoProcessing(roadBGR6);
    imageShow("Output6", Output6);

    destroyAllWindows();

	return 0;
}
