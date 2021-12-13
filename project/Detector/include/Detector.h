#pragma once

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

struct GridAndStride
{
	int grid0;
	int grid1;
	int stride;
};

struct Object
{
	cv::Rect_<float> rect;
	int label;
	float prob;
};

class Detector {
public:
	Detector() = default;

	//VARS

	int INPUT_W;
	int INPUT_H;
	const int NUM_CLASSES = 80;
	const float NMS_THRESH = 0.45f;
	const float BBOX_CONF_THRESH = 0.3f;
	std::string input_model;
	std::string device_name;
	const std::vector<int> desired_classes = {0, 1, 2, 3, 5, 7, 9, 11};

	//METHODS
	cv::Mat staticResize(cv::Mat& img);
	void blobFromImage(cv::Mat& img, InferenceEngine::Blob::Ptr& blob);
	void generateGridStride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);
	void generateProposals(std::vector<GridAndStride> grid_strides, const float* feat_ptr, float prob_threshold, std::vector<Object>& objects);
	void qsortDescentInPlace(std::vector<Object>& faceobjects, int left, int right);
	void qsortDescentInPlace(std::vector<Object>& objects);
	inline float intersectionArea(const Object& a, const Object& b);
	void nmsSortedBBoxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);
	void decodeOutputs(const float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h);
	cv::Mat drawObjects(const cv::Mat& bgr, const std::vector<Object>& objects);
	cv::Mat run(cv::Mat& Frame);
};