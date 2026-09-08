#include "console.hpp"
#include "mnist_generation.hpp"
#include "network.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char *argv[]) {
  cli::Console console;

  axon::Network network(0.001f);
  network.Data<MnistGeneration>();

  console["network"]["load"].function =
      [&](const std::vector<std::string> &args) {
        std::string name = "network.nn";
        if (args.size() > 0)
          name = args[0];
        network.LoadNetworkWeights(name);
      };

  console["network"]["train"].function =
      [&](const std::vector<std::string> &args) {
        if (args.size() > 0 &&
            !all_of(args[0].begin(), args[0].end(), ::isdigit)) {
          std::cout << "Usage: train <integer>" << std::endl;
          return;
        }
        int iterations = 1;
        if (args.size() > 0)
          iterations = stoi(args[0]);

        network.Train(iterations);
      };

  console["network"]["set"]["learningRate"].function =
      [&](const std::vector<std::string> &args) {
        if (args.size() < 0) {
          std::cout << "Usage: train [double > 0]" << std::endl;
          return;
        }
        std::istringstream ss(args[0]);
        double lr = 1;
        ss >> lr;
        if (lr < 0) {
          std::cout << "Usage: train [double > 0]" << std::endl;
          return;
        }
        network.SetLearningRate(lr);
      };

  console["network"]["set"]["noise"].function =
      [&](const std::vector<std::string> &args) {
        if (args.size() < 0) {
          std::cout << "Usage: noise [double > 0]" << std::endl;
          return;
        }
        std::istringstream ss(args[0]);
        double noise = 0.0;
        ss >> noise;
        if (noise < 0) {
          std::cout << "Usage: noise [double > 0]" << std::endl;
          return;
        }

        MnistGeneration::SetNoise(noise);
      };

  console["network"]["save"].function =
      [&](const std::vector<std::string> &args) {
        std::string name = "network.nn";
        if (args.size() > 0)
          name = args[0];
        network.SaveNetworkWeights(name);
      };

  console["network"]["generate"].function =
      [&](const std::vector<std::string> &) {
        std::vector<double> input(794);
        std::vector<double> output(784);
        MnistGeneration::GetExample(input, output);
        std::ofstream correct("correct.pgm");
        correct << "P2" << std::endl
                << "28 28" << std::endl
                << "255" << std::endl;
        for (int i = 0; i < 784; i++)
          correct << (int)(input[i] * 255) << " ";
        correct.close();
        output = network.Compute(input);
        std::ofstream generated("generated.pgm");
        generated << "P2" << std::endl
                  << "28 28" << std::endl
                  << "255" << std::endl;
        for (int i = 0; i < 784; i++)
          generated << (int)(output[i] * 255) << " ";
      };

  if (argc == 1)
    console.Run();
  else
    console.Run(argv[1]);
}
