#include "mnist_generation.hpp"
#include "network.hpp"
#include "log.hpp"
#include "console.hpp"
#include <memory>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>

int main()
{
	Console console;

	std::shared_ptr<NetworkData> data = std::make_shared<MnistGeneration>();
	Network network(data, 0.1f);

	console["exit"].function = 
		[&](const std::vector<std::string>&)
		{
			console.stop = true;
		};

	console["set"]["newLineAutoComplete"]["1"].function = console["set"]["newLineAutoComplete"]["true"].function =
		[&](const std::vector<std::string>&)
		{
			console.settings.newLineAutoComplete = true;
			std::cout<<"newLineAutoComplete was set to true"<<std::endl;
		};

	console["set"]["newLineAutoComplete"]["0"].function = console["set"]["newLineAutoComplete"]["false"].function =
		[&](const std::vector<std::string>&)
		{
			console.settings.newLineAutoComplete = false;
			std::cout<<"newLineAutoComplete was set to false"<<std::endl;
		};

	console["set"]["newLineAutoComplete"]["show"].function =
		[&](const std::vector<std::string>&)
		{
			std::cout<<"newLineAutoComplete is set to "<<((console.settings.newLineAutoComplete) ? "true" : "false")<<std::endl;
		};

	console["set"]["histSize"].function =
		[&](const std::vector<std::string>& args)
		{
			if(args.size()<1||!all_of(args[0].begin(),args[0].end(),::isdigit))
			{
				std::cout<<"Usage: set histSize [integer]";
				return;
			}
			console.settings.histSize = stoi(args[0]);
			std::cout<<"histSize was set to "<<console.settings.histSize<<std::endl;

		};

	console["network"]["load"].function =
		[&](const std::vector<std::string>& args)
		{
			std::string name = "network.nn";
			if(args.size() > 0)
				name = args[0];
			network.LoadNetwork(name);
		};
	
	console["network"]["train"].function = 
		[&](const std::vector<std::string>& args)
		{
			if(args.size()>0&&!all_of(args[0].begin(),args[0].end(),::isdigit))
			{
				std::cout<<"Usage: train <integer>"<<std::endl;
				return;
			}
			int iterations = 1;
			if(args.size()>0)
				iterations = stoi(args[0]);

			network.Train(iterations);
	
		};

	console["network"]["set"]["learningRate"].function =
		[&](const std::vector<std::string>& args)
		{
			if(args.size()<0)
			{
				std::cout<<"Usage: train [integer > 0]"<<std::endl;
				return;
			}
			std::istringstream ss(args[0]);
			double lr=1;
			ss >> lr;
			if(lr <= 0)
			{
				std::cout<<"Usage: train [integer > 0]"<<std::endl;
				return;
			}	
			network.SetLearningRate(lr);
	
		};		

	console["network"]["save"].function =
		[&](const std::vector<std::string>& args)
		{
			std::string name = "network.nn";
			if(args.size() > 0)
				name = args[0];
			network.SaveNetwork(name);
		};
	
	console["network"]["generate"].function =
		[&](const std::vector<std::string>&)
		{
			MnistGeneration gen;
			std::vector<double> input(794);
			std::vector<double> output(784);
			gen.LoadTestData();
			gen.GetNextTestData(input,output);
			std::ofstream correct("correct.pgm");
			correct<<"P2"<<std::endl<<"28 28"<<std::endl<<"255"<<std::endl;
			for(int i=0;i<784;i++)
				correct<<(int)(output[i]*255)<<" ";
			correct.close();
			output = network.Compute(input);
			std::ofstream generated("generated.pgm");
			generated<<"P2"<<std::endl<<"28 28"<<std::endl<<"255"<<std::endl;
			for(int i=0;i<784;i++)
				generated<<(int)(output[i]*255)<<" ";
		};

	runConsole(console);
	//std::shared_ptr<NetworkData> test = std::make_shared<MnistGeneration>();
	//test->LoadTrainingData();
	//std::vector<double> inputs(794);
	//std::vector<double> outputs(784);
	//test->GetNextTrainingData(inputs,outputs);
	//network.LoadNetwork("network.nn");
	//outputs = network.Compute(inputs);
	//std::ofstream num("num.pgm");
	//num<<"P2\n28 28\n255\n";
	//for(int i=0;i<784;i++)
		//num<<(int)((outputs[i])*255)<<" ";
	//num.close();
	//network.SaveNetwork("network.nn");
	//Log.Level(Log.error | Log.info);
	//network.Train(5);
	//network.Test();
	//network.SaveNetwork("network.nn");
}
