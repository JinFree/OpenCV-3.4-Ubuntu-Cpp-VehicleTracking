#include "OpenCV.h"
#include "OpenCV_Video.h"

using namespace std;
using namespace cv;

Mat videoProcessor::videoProcess(Mat Input) {
    Mat output = Input.clone();
    output = resizeImage(output, Size(960, 640));
    output = undistortImage(output);
    return output;
}
int main(void) {
    string trackerTypes[5] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW"};
    videoProcessor processor;
    processor.trackerType = trackerTypes[0];
    processor.setInput(path+roadVideo);
    processor.setOutput(path+"Lane_Detection_Videos/draw_multi_"+processor.trackerType+".mp4");
    processor.showInput("Input");
    processor.showOutput("Output");
    processor.setDelay(int(1000./processor.getFrameRate()));
    processor.multiDetectRun();
    return 0;
}
