#include "mnist_generation.hpp"
#include "network.hpp"
#include "log.hpp"
#include <memory>

int main()
{
	//this is a comment
	std::shared_ptr<NetworkData> data = std::make_shared<MnistGeneration>();
	Network network(data, 0.1f);
	network.LoadNetwork("network.nn");
	Log.Level(Log.error | Log.info);
	network.Train(50);
	network.Test();
	network.SaveNetwork("network.nn");
}
