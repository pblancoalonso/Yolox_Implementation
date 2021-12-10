#pragma once

#include <string>
#include <json.hpp>
#include <fstream>

class Parser {
public:
	Parser();

	void parseConfig(const std::string jsonPath);

	// VARS 
	std::string VideoPath;
	std::string InferenceDevice;
	std::string ModelPath;
	std::string OutputPath;
	bool IsRecording;

};