#include "OpenCV.h"
#include "OpenCV_Video.h"

using namespace std;
using namespace cv;

Mat videoProcessor::videoProcess(Mat Input) {
    Mat output = Input.clone(); 
    output = undistortImage(output);
    return output;
}
int main(void) {
    videoProcessor processor;
    processor.setInput(path+roadVideo);
    processor.setOutput(path+"Lane_Detection_Videos/draw_undistorted_test.mp4");
    processor.showInput("Input");
    processor.showOutput("Output");
    processor.setDelay(int(1000./processor.getFrameRate()));
    processor.run();
   
    return 0;
}
