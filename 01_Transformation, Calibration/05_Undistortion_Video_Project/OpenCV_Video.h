#ifndef OPENCV_VIDEO_H_
#define OPENCV_VIDEO_H_
#endif //OPENCV_VIDEO_H_
#pragma once
#include "OpenCV.h"
#include <iomanip>

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
