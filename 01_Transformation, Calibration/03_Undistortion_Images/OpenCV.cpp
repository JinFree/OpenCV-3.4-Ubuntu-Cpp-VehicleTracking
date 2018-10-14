#include "OpenCV.h"

using namespace std;
using namespace cv;

int main(void) {
    string roadImagePath = "../../Data/Lane_Detection_Images/";
    string file1 = "solidYellowCurve2.jpg";
    string file2 = "solidWhiteRight.jpg";
    string file3 = "solidYellowCurve.jpg";
    string file4 = "solidWhiteCurve.jpg";
    string file5 = "solidYellowLeft.jpg";
    string file6 = "whiteCarLaneSwitch.jpg";
    string file7 = "test.jpg";

    Mat roadBGR1 = imageRead(roadImagePath + file1, IMREAD_COLOR);
    imageShow("roadBGR1", roadBGR1);

    Mat Output1 =  undistortImage(roadBGR1);
    imageShow("Output1", Output1);

    Mat roadBGR2 = imageRead(roadImagePath + file2, IMREAD_COLOR);
    imageShow("roadBGR2", roadBGR2);

    Mat Output2 =  undistortImage(roadBGR2);
    imageShow("Output2", Output2);

    Mat roadBGR3 = imageRead(roadImagePath + file3, IMREAD_COLOR);
    imageShow("roadBGR3", roadBGR3);

    Mat Output3 =  undistortImage(roadBGR3);
    imageShow("Output3", Output3);

    Mat roadBGR4 = imageRead(roadImagePath + file4, IMREAD_COLOR);
    imageShow("roadBGR4", roadBGR4);

    Mat Output4 =  undistortImage(roadBGR4);
    imageShow("Output4", Output4);

    Mat roadBGR5 = imageRead(roadImagePath + file5, IMREAD_COLOR);
    imageShow("roadBGR5", roadBGR5);

    Mat Output5 =  undistortImage(roadBGR5);
    imageShow("Output5", Output5);

    Mat roadBGR6 = imageRead(roadImagePath + file6, IMREAD_COLOR);
    imageShow("roadBGR6", roadBGR6);

    Mat Output6 =  undistortImage(roadBGR6);
    imageShow("Output6", Output6);

    Mat roadBGR7 = imageRead(roadImagePath + file7, IMREAD_COLOR);
    imageShow("roadBGR7", roadBGR7);

    Mat Output7 =  undistortImage(roadBGR7);
    imageShow("Output7", Output7);

    destroyAllWindows();
    return 0;
}