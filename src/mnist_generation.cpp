#include "mnist_generation.hpp"

MnistGeneration::MnistGeneration()
{ 
	trainingInputArray_.resize(784);
	trainingOutputArray_.resize(11);
	testInputArray_.resize(784);
	testOutputArray_.resize(11);
}

MnistGeneration::~MnistGeneration()
{
	trainingInputs_.close();
	trainingOutputs_.close();
}
bool MnistGeneration::GetNextTrainingData(std::vector<double> &inputs, std::vector<double> &outputs)
{

	if(trainingInputs_.eof() || trainingOutputs_.eof() || !trainingInputs_.good()|| !trainingOutputs_.good())
	{
		return false;
	}
	if(inputs.size()<784)
		Log(Log.error,"The input vector is too small");
	if(outputs.size()<11)
		Log(Log.error,"The output vector is too small");
	//inputs.resize(784);
	//outputs.resize(11);
	if (rand() % 15 == 0)
	{
		for (int i = 0; i < 784; i++)
		{
			inputs[i] = (double)(rand() % 256) / 255.0;
			trainingInputArray_[i] = inputs[i];
		}
		for (int i = 0; i < 11; i++)
		{
			outputs[i] = 0;
			trainingOutputArray_[i] = outputs[i];
		}
		outputs[10] = 1;
		trainingOutputArray_[10] = 1;
	}
	else
	{
		char byte;
		for (int i = 0; i < 784; i++)
		{
			trainingInputs_.read(&byte, 1);
			inputs[i] = (double)((unsigned char)byte+(rand()%50-25)) / 255.0;
			inputs[i] = ((inputs[i] < 0) ? 0 : inputs[i]);
			inputs[i] = ((inputs[i] > 1) ? 1 : inputs[i]);
			trainingInputArray_[i] = inputs[i];
		}
		for (int i = 0; i < 11; i++)
		{
			outputs[i] = 0;
			trainingOutputArray_[i] = outputs[i];
		}
		if(!trainingOutputs_.read(&byte, 1))
			return false;
		outputs[(int)byte] = 1;
		trainingOutputArray_[(int)byte] = 1;
	}
	return true;
	
}

void MnistGeneration::GetSameTrainingData(std::vector<double> &inputs, std::vector<double> &outputs)
{
	inputs.resize(784);
	outputs.resize(11);
	for (int i = 0; i < 784; i++)
	{
		inputs[i] = trainingInputArray_[i];
	}
	for (int i = 0; i < 11; i++)
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
	inputs.resize(784);
	outputs.resize(11);
	if (rand() % 15 == 0)
	{
		for (int i = 0; i < 784; i++)
		{
			inputs[i] = (double)(rand() % 256) / 255.0;
			testInputArray_[i] = inputs[i];
		}
		for (int i = 0; i < 11; i++)
		{
			outputs[i] = 0;
			testOutputArray_[i] = outputs[i];
		}
		outputs[10] = 1;
		testOutputArray_[10] = 1;
	}
	else
	{
		char byte;
		for (int i = 0; i < 784; i++)
		{
			testInputs_.read(&byte, 1);
			inputs[i] = (double)((unsigned char)byte) / 255.0;
			testInputArray_[i] = inputs[i];
		}
		for (int i = 0; i < 11; i++)
		{
			outputs[i] = 0;
			testOutputArray_[i] = outputs[i];
		}
		if(!testOutputs_.read(&byte, 1))
			return false;
		outputs[(int)byte] = 1;
		testOutputArray_[(int)byte] = 1;
	}
	return true;
}

void MnistGeneration::GetSameTestData(std::vector<double> &inputs, std::vector<double> &outputs)
{
	inputs.resize(784);
	outputs.resize(11);
	for (int i = 0; i < 784; i++)
	{
		inputs[i] = testInputArray_[i];
	}
	for (int i = 0; i < 11; i++)
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
