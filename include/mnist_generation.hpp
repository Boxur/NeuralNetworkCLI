#pragma once
#include "network_data.hpp"
#include "log.hpp"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <ctime>

class MnistGeneration: public NetworkData
{
private:
	std::initializer_list<int> networkLayout_ = {794,30, 784};
	int numberOfLayers_ = 3;

	double noise_ = 0.0f;


	std::vector<double> trainingInputArray_;
	std::vector<double> trainingOutputArray_;
	std::vector<double> testInputArray_;
	std::vector<double> testOutputArray_;

	std::fstream trainingInputs_;
	std::fstream trainingOutputs_;
	std::fstream testInputs_;
	std::fstream testOutputs_;

public:
	MnistGeneration();

	~MnistGeneration();

	bool GetNextTrainingData(std::vector<double> &inputs, std::vector<double> &outputs) override;

	void GetSameTrainingData(std::vector<double> &inputs, std::vector<double> &outputs) override;

	bool GetNextTestData(std::vector<double> &inputs, std::vector<double> &outputs) override;

	void GetSameTestData(std::vector<double> &inputs, std::vector<double> &outputs) override;

 	bool LoadTrainingData() override;

	bool LoadTestData() override;

	std::initializer_list<int> GetNetworkLayout() override;

	int GetNumberOfLayers() override;

	void SetNoise(double noise);
};
