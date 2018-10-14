#pragma once
#include "OpenCV.h"
#include <iomanip>

string roadVideo = "Lane_Detection_Videos/solidWhiteRight.mp4";

class videoProcessor {
public:
	void run();
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
