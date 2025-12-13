#include "mnist_generation.hpp"
#include "network.hpp"
#include "log.hpp"
#include <memory>
#include <fstream>
#include <iostream>

int main()
{
	std::shared_ptr<NetworkData> data = std::make_shared<MnistGeneration>();
	{	
		Network network(data, 0.01f);
		network.LoadNetwork("network.nn");
		network.Train(10);
		std::shared_ptr<NetworkData> test = std::make_shared<MnistGeneration>();
		test->LoadTrainingData();
		std::vector<double> inputs(794);
		std::vector<double> outputs(784);
		test->GetNextTrainingData(inputs,outputs);
		//network.LoadNetwork("network.nn");
		outputs = network.Compute(inputs);
		std::ofstream num("num.pgm");
		num<<"P2\n28 28\n255\n";
		for(int i=0;i<784;i++)
			num<<(int)((outputs[i])*255)<<" ";
		num.close();
		network.SaveNetwork("network.nn");
	}
	//Log.Level(Log.error | Log.info);
	//network.Train(5);
	//network.Test();
	//network.SaveNetwork("network.nn");
}
