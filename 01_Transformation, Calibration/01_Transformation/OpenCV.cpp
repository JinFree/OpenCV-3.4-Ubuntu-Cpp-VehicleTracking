#include "OpenCV.h"

using namespace std;
using namespace cv;
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
