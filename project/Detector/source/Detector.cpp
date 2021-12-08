#include <Detector.h>

const float color_list[80][3] =
{
	{0.000, 0.447, 0.741},
	{0.850, 0.325, 0.098},
	{0.929, 0.694, 0.125},
	{0.494, 0.184, 0.556},
	{0.466, 0.674, 0.188},
	{0.301, 0.745, 0.933},
	{0.635, 0.078, 0.184},
	{0.300, 0.300, 0.300},
	{0.600, 0.600, 0.600},
	{1.000, 0.000, 0.000},
	{1.000, 0.500, 0.000},
	{0.749, 0.749, 0.000},
	{0.000, 1.000, 0.000},
	{0.000, 0.000, 1.000},
	{0.667, 0.000, 1.000},
	{0.333, 0.333, 0.000},
	{0.333, 0.667, 0.000},
	{0.333, 1.000, 0.000},
	{0.667, 0.333, 0.000},
	{0.667, 0.667, 0.000},
	{0.667, 1.000, 0.000},
	{1.000, 0.333, 0.000},
	{1.000, 0.667, 0.000},
	{1.000, 1.000, 0.000},
	{0.000, 0.333, 0.500},
	{0.000, 0.667, 0.500},
	{0.000, 1.000, 0.500},
	{0.333, 0.000, 0.500},
	{0.333, 0.333, 0.500},
	{0.333, 0.667, 0.500},
	{0.333, 1.000, 0.500},
	{0.667, 0.000, 0.500},
	{0.667, 0.333, 0.500},
	{0.667, 0.667, 0.500},
	{0.667, 1.000, 0.500},
	{1.000, 0.000, 0.500},
	{1.000, 0.333, 0.500},
	{1.000, 0.667, 0.500},
	{1.000, 1.000, 0.500},
	{0.000, 0.333, 1.000},
	{0.000, 0.667, 1.000},
	{0.000, 1.000, 1.000},
	{0.333, 0.000, 1.000},
	{0.333, 0.333, 1.000},
	{0.333, 0.667, 1.000},
	{0.333, 1.000, 1.000},
	{0.667, 0.000, 1.000},
	{0.667, 0.333, 1.000},
	{0.667, 0.667, 1.000},
	{0.667, 1.000, 1.000},
	{1.000, 0.000, 1.000},
	{1.000, 0.333, 1.000},
	{1.000, 0.667, 1.000},
	{0.333, 0.000, 0.000},
	{0.500, 0.000, 0.000},
	{0.667, 0.000, 0.000},
	{0.833, 0.000, 0.000},
	{1.000, 0.000, 0.000},
	{0.000, 0.167, 0.000},
	{0.000, 0.333, 0.000},
	{0.000, 0.500, 0.000},
	{0.000, 0.667, 0.000},
	{0.000, 0.833, 0.000},
	{0.000, 1.000, 0.000},
	{0.000, 0.000, 0.167},
	{0.000, 0.000, 0.333},
	{0.000, 0.000, 0.500},
	{0.000, 0.000, 0.667},
	{0.000, 0.000, 0.833},
	{0.000, 0.000, 1.000},
	{0.000, 0.000, 0.000},
	{0.143, 0.143, 0.143},
	{0.286, 0.286, 0.286},
	{0.429, 0.429, 0.429},
	{0.571, 0.571, 0.571},
	{0.714, 0.714, 0.714},
	{0.857, 0.857, 0.857},
	{0.000, 0.447, 0.741},
	{0.314, 0.717, 0.741},
	{0.50, 0.5, 0}
};

cv::Mat Detector::staticResize(cv::Mat& img) {
	float r = std::min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
	// r = std::min(r, 1.0f);
	int unpad_w = r * img.cols;
	int unpad_h = r * img.rows;
	cv::Mat re(unpad_h, unpad_w, CV_8UC3);
	cv::resize(img, re, re.size());
	//cv::Mat out(INPUT_W, INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
	cv::Mat out(Detector::INPUT_H, Detector::INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
	re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
	return out;
}

void Detector::blobFromImage(cv::Mat& img, InferenceEngine::Blob::Ptr& blob) {
	int channels = 3;
	int img_h = img.rows;
	int img_w = img.cols;
	InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
	if (!mblob)
	{
		THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
			<< "but by fact we were not able to cast inputBlob to MemoryBlob";
	}
	// locked memory holder should be alive all time while access to its buffer happens
	auto mblobHolder = mblob->wmap();

	float* blob_data = mblobHolder.as<float*>();

	for (size_t c = 0; c < channels; c++)
	{
		for (size_t h = 0; h < img_h; h++)
		{
			for (size_t w = 0; w < img_w; w++)
			{
				blob_data[c * img_w * img_h + h * img_w + w] =
					(float)img.at<cv::Vec3b>(h, w)[c];
			}
		}
	}
}

void Detector::generateGridStride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
	for (auto stride : strides)
	{
		int num_grid_w = target_w / stride;
		int num_grid_h = target_h / stride;
		for (int g1 = 0; g1 < num_grid_h; g1++)
		{
			for (int g0 = 0; g0 < num_grid_w; g0++)
			{
				grid_strides.push_back((GridAndStride) { g0, g1, stride });
			}
		}
	}
}

void Detector::generateProposals(std::vector<GridAndStride> grid_strides, const float* feat_ptr, float prob_threshold, std::vector<Object>& objects)
{

	const int num_anchors = grid_strides.size();

	for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
	{
		const int grid0 = grid_strides[anchor_idx].grid0;
		const int grid1 = grid_strides[anchor_idx].grid1;
		const int stride = grid_strides[anchor_idx].stride;

		const int basic_pos = anchor_idx * (NUM_CLASSES + 5);

		float x_center = (feat_ptr[basic_pos + 0] + grid0) * stride;
		float y_center = (feat_ptr[basic_pos + 1] + grid1) * stride;
		float w = exp(feat_ptr[basic_pos + 2]) * stride;
		float h = exp(feat_ptr[basic_pos + 3]) * stride;
		float x0 = x_center - w * 0.5f;
		float y0 = y_center - h * 0.5f;

		float box_objectness = feat_ptr[basic_pos + 4];

		for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
		{
			if (std::find(desired_classes.begin(), desired_classes.end(), class_idx) != desired_classes.end()) {

				float box_cls_score = feat_ptr[basic_pos + 5 + class_idx];
				float box_prob = box_objectness * box_cls_score;
				if (box_prob > prob_threshold)
				{
					Object obj;
					obj.rect.x = x0;
					obj.rect.y = y0;
					obj.rect.width = w;
					obj.rect.height = h;
					obj.label = class_idx;
					obj.prob = box_prob;

					objects.push_back(obj);
				}
			}

		}

	}
}

inline float Detector::intersectionArea(const Object& a, const Object& b)
{
	cv::Rect_<float> inter = a.rect & b.rect;
	return inter.area();
}

void Detector::qsortDescentInPlace(std::vector<Object>& faceobjects, int left, int right)
{
	int i = left;
	int j = right;
	float p = faceobjects[(left + right) / 2].prob;

	while (i <= j)
	{
		while (faceobjects[i].prob > p)
			i++;

		while (faceobjects[j].prob < p)
			j--;

		if (i <= j)
		{
			// swap
			std::swap(faceobjects[i], faceobjects[j]);

			i++;
			j--;
		}
	}

#pragma omp parallel sections
	{
#pragma omp section
		{
			if (left < j) qsortDescentInPlace(faceobjects, left, j);
		}
#pragma omp section
		{
			if (i < right) qsortDescentInPlace(faceobjects, i, right);
		}
	}
}

void Detector::qsortDescentInPlace(std::vector<Object>& objects)
{
	if (objects.empty())
		return;

	qsortDescentInPlace(objects, 0, objects.size() - 1);
}

void Detector::nmsSortedBBoxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
	picked.clear();

	const int n = faceobjects.size();

	std::vector<float> areas(n);
	for (int i = 0; i < n; i++)
	{
		areas[i] = faceobjects[i].rect.area();
	}

	for (int i = 0; i < n; i++)
	{
		const Object& a = faceobjects[i];

		int keep = 1;
		for (int j = 0; j < (int)picked.size(); j++)
		{
			const Object& b = faceobjects[picked[j]];

			// intersection over union
			float inter_area = intersectionArea(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			// float IoU = inter_area / union_area
			if (inter_area / union_area > nms_threshold)
				keep = 0;
		}

		if (keep)
			picked.push_back(i);
	}
}

void Detector::decodeOutputs(const float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h) {
	std::vector<Object> proposals;
	std::vector<int> strides = { 8, 16, 32 };
	std::vector<GridAndStride> grid_strides;

	generateGridStride(INPUT_W, INPUT_H, strides, grid_strides);
	generateProposals(grid_strides, prob, BBOX_CONF_THRESH, proposals);
	qsortDescentInPlace(proposals);

	std::vector<int> picked;
	nmsSortedBBoxes(proposals, picked, NMS_THRESH);
	int count = picked.size();
	objects.resize(count);

	for (int i = 0; i < count; i++)
	{
		objects[i] = proposals[picked[i]];

		// adjust offset to original unpadded
		float x0 = (objects[i].rect.x) / scale;
		float y0 = (objects[i].rect.y) / scale;
		float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
		float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

		// clip
		x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
		y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
		x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
		y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

		objects[i].rect.x = x0;
		objects[i].rect.y = y0;
		objects[i].rect.width = x1 - x0;
		objects[i].rect.height = y1 - y0;
	}
}

cv::Mat Detector::drawObjects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
	static const char* class_names[] = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"
	};

	cv::Mat image = bgr.clone();

	for (size_t i = 0; i < objects.size(); i++)
	{
		const Object& obj = objects[i];

		fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
			obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

		cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
		float c_mean = cv::mean(color)[0];
		cv::Scalar txt_color;
		if (c_mean > 0.5) {
			txt_color = cv::Scalar(0, 0, 0);
		}
		else {
			txt_color = cv::Scalar(255, 255, 255);
		}

		cv::rectangle(image, obj.rect, color * 255, 2);

		char text[256];
		sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

		cv::Scalar txt_bk_color = color * 0.7 * 255;

		int x = obj.rect.x;
		int y = obj.rect.y + 1;
		if (y > image.rows)
			y = image.rows;

		cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
			txt_bk_color, -1);

		cv::putText(image, text, cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
	}

	cv::imshow("image", image);
	cv::waitKey(1);
	return image;
}


cv::Mat Detector::run(cv::Mat& Frame) {

	// Step 1. Initialize inference engine core

	InferenceEngine::Core ie;

	// Step 2. Read a model in OpenVINO Intermediate Representation

	InferenceEngine::CNNNetwork network = ie.ReadNetwork(input_model);
	if (network.getOutputsInfo().size() != 1)
		throw std::logic_error("Sample supports topologies with 1 output only");
	if (network.getInputsInfo().size() != 1)
		throw std::logic_error("Sample supports topologies with 1 input only");

	// Step 3. Configure input & output

	InferenceEngine::InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
	std::string input_name = network.getInputsInfo().begin()->first;

	// Prepare output blobs

	if (network.getOutputsInfo().empty()) {
		std::cerr << "Network outputs info is empty" << std::endl;
	}
	InferenceEngine::DataPtr output_info = network.getOutputsInfo().begin()->second;
	std::string output_name = network.getOutputsInfo().begin()->first;

	output_info->setPrecision(InferenceEngine::Precision::FP32);

	// Step 4. Loading a model to the device

	InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, device_name);

	// Step 5. Create an infer request

	InferenceEngine::InferRequest infer_request = executable_network.CreateInferRequest();

	// Step 6. Prepare input

	cv::Mat pr_img = staticResize(Frame);
	InferenceEngine::Blob::Ptr imgBlob = infer_request.GetBlob(input_name);
	blobFromImage(pr_img, imgBlob);

	// Step 7. Do inference

	infer_request.Infer();

	// Step 8. Process output

	const InferenceEngine::Blob::Ptr output_blob = infer_request.GetBlob(output_name);
	InferenceEngine::MemoryBlob::CPtr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob);
	if (!moutput) {
		throw std::logic_error("We expect output to be inherited from MemoryBlob, "
			"but by fact we were not able to cast output to MemoryBlob");
	}

	auto moutputHolder = moutput->rmap();
	const float* net_pred = moutputHolder.as<const InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

	int img_w = Frame.cols;
	int img_h = Frame.rows;
	float scale = std::min(INPUT_W / (Frame.cols * 1.0), INPUT_H / (Frame.rows * 1.0));
	std::vector<Object> objects;

	decodeOutputs(net_pred, objects, scale, img_w, img_h);
	cv::Mat proc_frame = drawObjects(Frame, objects);
	return(proc_frame);
}


