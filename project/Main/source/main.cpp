#include <Detector.h>

int main() {
    // Instantiate detector
    
    Detector det;

    /*cv::namedWindow("Result");
    cv::setWindowProperty("Result", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);*/

    cv::VideoCapture Video("/opt/samples/videos/sample_2.mp4");

    if (!Video.isOpened()) {
        std::cout << "Error opening video" << std::endl;
        return -1;
    }

    cv::VideoWriter Writer;
    // If recording enabled
    cv::Mat DummyFrame;
    Video.read(DummyFrame);
    Writer = cv::VideoWriter("/opt/output.avi", cv::VideoWriter::fourcc('F', 'M', 'P', '4'), Video.get(cv::CAP_PROP_FPS), cv::Size(DummyFrame.cols, DummyFrame.rows));


    while(true){
        cv::Mat Frame;
        bool IsNotLastFrame = Video.read(Frame);
        if(!IsNotLastFrame){
            std::cout << "No more frames to be processed" << std::endl;
            break;
        }
        cv::Mat test = det.run(Frame);
        Writer.write(test);
    }

    return 0;
}