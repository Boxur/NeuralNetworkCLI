#include "mnist_generation.hpp"

MnistGeneration::MnistGeneration()
{ 
	std::srand(std::time({}));
	trainingInputArray_.resize(794);
	trainingOutputArray_.resize(784);
	testInputArray_.resize(794);
	testOutputArray_.resize(784);
}

MnistGeneration::~MnistGeneration()
{
	if(trainingInputs_)
		trainingInputs_.close();
	if(trainingOutputs_)
		trainingOutputs_.close();
	if(testInputs_)
		testInputs_.close();
	if(testOutputs_)
		testOutputs_.close();
}

bool MnistGeneration::GetNextTrainingData(std::vector<double> &inputs, std::vector<double> &outputs)
{

	if(trainingInputs_.eof() || trainingOutputs_.eof() || !trainingInputs_.good()|| !trainingOutputs_.good())
	{
		return false;
	}
	if(inputs.size()<794)
		Log(Log.error,"The input vector is too small");
	if(outputs.size()<784)
		Log(Log.error,"The output vector is too small");
	
	char byte;
	for (int i = 0; i < 784; i++)
	{
		trainingInputs_.read(&byte, 1);
		outputs[i] = (double)((unsigned char)byte)/255;
		inputs[i] = outputs[i]*(1-noise_)+(((double)std::rand()/(double)RAND_MAX)*noise_);
		trainingOutputArray_[i] = outputs[i];
		trainingInputArray_[i] = inputs[i];
	}
	for (int i = 0; i < 10; i++)
		trainingInputArray_[i+784] = inputs[i+784] = 0;
	if(!trainingOutputs_.read(&byte, 1))
		return false;
	inputs[(int)byte+784] = 1;
	trainingInputArray_[(int)byte+784] = 1;
	return true;
	
}

void MnistGeneration::GetSameTrainingData(std::vector<double> &inputs, std::vector<double> &outputs)
{
	for (int i = 0; i < 794; i++)
	{
		inputs[i] = trainingInputArray_[i];
	}
	for (int i = 0; i < 784; i++)
	{
		outputs[i] = trainingOutputArray_[i];
	}
}

bool MnistGeneration::GetNextTestData(std::vector<double> &inputs, std::vector<double> &outputs)
{
	if (!testInputs_ || !testOutputs_)
	{
		return false;
	}

	char byte;
	for (int i = 0; i < 784; i++)
	{
		testInputs_.read(&byte, 1);
		inputs[i] = (double)((unsigned char)byte) / 255.0;
		outputs[i] = inputs[i];
		testInputArray_[i] = inputs[i];
		testOutputArray_[i] = outputs[i];
	}
	for (int i = 0; i < 10; i++)
	{
		inputs[i+784] = 0;
		testInputArray_[i+784] = 0;
	}
	if(!testOutputs_.read(&byte, 1))
		return false;
	inputs[(int)byte+784] = 1;
	testInputArray_[(int)byte+784] = 1;
	return true;
}

void MnistGeneration::GetSameTestData(std::vector<double> &inputs, std::vector<double> &outputs)
{
	for (int i = 0; i < 794; i++)
	{
		inputs[i] = testInputArray_[i];
	}
	for (int i = 0; i < 784; i++)
	{
		outputs[i] = testOutputArray_[i];
	}

}

bool MnistGeneration::LoadTrainingData()
{
	trainingInputs_.close();
	trainingOutputs_.close();
    trainingInputs_.open("data/train-images.ubyte", std::ios::in|std::ios::binary);
	if (!trainingInputs_) return false;
 	trainingOutputs_.open("data/train-labels.ubyte", std::ios::in|std::ios::binary);
	if (!trainingOutputs_) return false;

	char* skip = new char[16];
	trainingInputs_.read(skip, 16);
	trainingOutputs_.read(skip, 8);
	delete[] skip;
	return true;
}

bool MnistGeneration::LoadTestData()
{
	testInputs_.close();
	testOutputs_.close();
	testInputs_.open("data/test-images.ubyte", std::ios::in|std::ios::binary);
	if(!testInputs_) return false;
	testOutputs_.open("data/test-labels.ubyte", std::ios::in|std::ios::binary);
	if(!testOutputs_) return false;
	char* skip = new char[16];
	testInputs_.read(skip, 16);
	testOutputs_.read(skip, 8);
	delete[] skip;
	return true;
}

std::initializer_list<int> MnistGeneration::GetNetworkLayout()
{
	return networkLayout_;
}

int MnistGeneration::GetNumberOfLayers()
{
	return numberOfLayers_;
}
