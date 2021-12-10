#include <Parser.h>

Parser::Parser() = default;

void Parser::parseConfig(const std::string jsonPath) {

	std::ifstream PathToConfig(jsonPath);

	nlohmann::json Config = nlohmann::json::parse(PathToConfig);

	ModelPath = Config["modelPath"];
	VideoPath = Config["inputPath"];
	InferenceDevice = Config["inferenceDevice"];
	OutputPath = Config["outputPath"];
	IsRecording = Config["isRecording"];
}