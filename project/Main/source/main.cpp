#include <Detector.h>
#include <Parser.h>

bool validInput(std::string name) {
	return name.find(".json") != -1;
}

int main() {

	// Instantiate detector  

	Detector det;

	// Instantiate JSON Parser and load Config

	Parser jsonParser;

	jsonParser.parseConfig("/opt/configuration/Config.json");

	// Load Vars

	det.device_name = jsonParser.InferenceDevice;
	det.input_model = jsonParser.ModelPath;

	// Instantiate VideoCapturer

	cv::VideoCapture Video(jsonParser.VideoPath);

	if (!Video.isOpened()) {
		std::cout << "Error opening video" << std::endl;
		return -1;
	}

	cv::VideoWriter Writer;

	// Check if recording is enabled

	if (jsonParser.IsRecording) {
		cv::Mat DummyFrame;
		Video.read(DummyFrame);
		Writer = cv::VideoWriter(jsonParser.OutputPath, cv::VideoWriter::fourcc('F', 'M', 'P', '4'), Video.get(cv::CAP_PROP_FPS), cv::Size(DummyFrame.cols, DummyFrame.rows));
	}

	// Main Loop

	while (true) {
		cv::Mat Frame;
		bool IsNotLastFrame = Video.read(Frame);
		if (!IsNotLastFrame) {
			std::cout << "No more frames to be processed" << std::endl;
			break;
		}
		cv::Mat FrameToRecord = det.run(Frame);
		Writer.write(FrameToRecord);
	}

	return 0;
}