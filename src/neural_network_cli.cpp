#include "mnist_data.h"
#include "network.h"
#include "log.h"
#include <memory>

int main()
{
	//this is a comment
	std::shared_ptr<NetworkData> data = std::make_shared<MnistData>();
	Network network(data, 0.1f);
	network.LoadNetwork("network.nn");
	Log.Level(Log.error | Log.info);
	network.Train(50);
	network.Test();
	network.SaveNetwork("network.nn");
}
