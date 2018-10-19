#pragma once
#include "OpenCV.h"
#include <iomanip>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

string roadVideo = "Lane_Detection_Videos/challenge.mp4";

inline cv::Ptr<cv::Tracker> createTrackerByName(cv::String name)
{
    cv::Ptr<cv::Tracker> tracker;
    if (name == "BOOSTING")
        tracker = TrackerBoosting::create();
    else if (name == "MIL")
        tracker = TrackerMIL::create();
    else if (name == "KCF")
        tracker = TrackerKCF::create();
    else if (name == "TLD")
        tracker = TrackerTLD::create();
    else if (name == "MEDIANFLOW")
        tracker = TrackerMedianFlow::create();
    else
        CV_Error(cv::Error::StsBadArg, "Invalid tracking algorithm name\n");

    return tracker;
}

class videoProcessor {
public:
    void run();
    void singleDetectRun();
    void multiDetectRun();
    Mat videoProcess(Mat Input);
    bool setInput(string videoPath);
    bool setOutput(string videoPath, int codec = 0, double framerate = 0.0, bool isColor = true);
    void showInput(string windowName);
    void showOutput(string windowName);
    double getFrameRate();
    Size getFrameSize();
    void stopIt();
    bool isStopped();
    bool isOpened();
    void setDelay(int d);
    bool readNextFrame(Mat& frame);
    void writeNextFrame(Mat& frame);
    long getFrameNumber();
    void stopAtFrameNumber(long frame);
    int getCodec(char codec[4]);
    string trackerType ="BOOSTING";
private:
    VideoCapture capture;
    string videoReadPath;
    string videoNameInput;
    string videoNameOutput;
    int delay;
    long frameNumber;
    long frameToStop;
    bool stop;
    VideoWriter writer;
    string videoWritePath;
    int currentIndex;
    int digits;
    string extension;
    double frameRate;
    Size frameSize;
};
void videoProcessor::multiDetectRun() {
    Mat frameInput;
    Mat frameOutput;
    Ptr<Tracker> tracker;

    if(!isOpened())
        return;
    stop = false;
    readNextFrame(frameInput);
    vector<Rect> bboxes;
    frameInput = resizeImage(frameInput, Size(960, 640));
    frameInput = undistortImage(frameInput);
    bool showCrosshair = true;
    bool fromCenter = false;
    cout << "\n==========================================================\n";
    cout << "OpenCV says press c to cancel objects selection process" << endl;
    cout << "It doesn't work. Press Escape to exit selection process" << endl;
    cout << "\n==========================================================\n";
    cv::selectROIs("MultiTracker", frameInput, bboxes, showCrosshair, fromCenter);
    destroyAllWindows();
    if(bboxes.size() < 1)
        return;
    vector<Scalar> colors;
    getRandomColors(colors, bboxes.size());
    Ptr<MultiTracker> multiTracker = cv::MultiTracker::create();
    for(int i=0; i < bboxes.size(); i++)
        multiTracker->add(createTrackerByName(trackerType), frameInput, Rect2d(bboxes[i]));


    while(!isStopped()){
        if(!readNextFrame(frameInput))
            break;
        if(videoNameInput.length() != 0)
            imshow(videoNameInput, frameInput);

        frameOutput = videoProcess(frameInput);

        multiTracker->update(frameOutput);

        for(unsigned i=0; i<multiTracker->getObjects().size(); i++)
            rectangle(frameOutput, multiTracker->getObjects()[i], colors[i], 2, 1);
        
        putText(frameOutput, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,255,255),2);

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
void videoProcessor::singleDetectRun() {
    Mat frameInput;
    Mat frameOutput;
    Ptr<Tracker> tracker;

    if (trackerType == "BOOSTING")
        tracker = TrackerBoosting::create();
    else if (trackerType == "MIL")
        tracker = TrackerMIL::create();
    else if (trackerType == "KCF")
        tracker = TrackerKCF::create();
    else if (trackerType == "TLD")
        tracker = TrackerTLD::create();
    else if (trackerType == "MEDIANFLOW")
        tracker = TrackerMedianFlow::create();

    if(!isOpened())
        return;
    stop = false;
    readNextFrame(frameInput);
    Rect2d bbox;
    frameInput = resizeImage(frameInput, Size(960, 640));
    frameInput = undistortImage(frameInput);
    bbox = selectROI(frameInput, false);
    destroyAllWindows();

    rectangle(frameInput, bbox, Scalar( 255, 0, 0 ), 2, 1 );
    tracker->init(frameInput, bbox);

    while(!isStopped()){
        if(!readNextFrame(frameInput))
            break;
        if(videoNameInput.length() != 0)
            imshow(videoNameInput, frameInput);

        frameOutput = videoProcess(frameInput);

        bool ok = tracker->update(frameOutput, bbox);

        if (ok)
            rectangle(frameOutput, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        else
            putText(frameOutput, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);

        putText(frameOutput, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,255,255),2);

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

