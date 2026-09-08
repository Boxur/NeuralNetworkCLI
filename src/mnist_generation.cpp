#include "mnist_generation.hpp"
#include "logger.hpp"
#include <algorithm>

static double leakyRelu(double x) { return ((x < 0) ? x / 10 : x); }

static double leakyReluDerivative(double x) { return ((x < 0) ? 0.1 : 1); }

static double sigmoidApprox(double x) {
  return ((x < -2.5) ? 0 : ((x < 2.5) ? 0.2 * x + 0.5 : 1));
}

static double sigmoidApproxDerivative(double x) {
  return ((x < -2.5) ? 0.05 : ((x < 2.5) ? 0.2 : 0.05));
}

double MnistGeneration::noise_ = 0.0;

MnistGeneration::MnistGeneration()
    : activationFunctions_({leakyRelu, leakyRelu, leakyRelu, sigmoidApprox}),
      activationFunctionDerivatives_({leakyReluDerivative, leakyReluDerivative,
                                      leakyReluDerivative,
                                      sigmoidApproxDerivative}) {
  std::srand(std::time({}));
  trainingInputArray_.resize(794);
  trainingOutputArray_.resize(784);
  testInputArray_.resize(794);
  testOutputArray_.resize(784);
}

MnistGeneration::~MnistGeneration() {
  if (trainingInputs_)
    trainingInputs_.close();
  if (trainingOutputs_)
    trainingOutputs_.close();
  if (testInputs_)
    testInputs_.close();
  if (testOutputs_)
    testOutputs_.close();
}

bool MnistGeneration::GetNextTrainingData(std::vector<double> &inputs,
                                          std::vector<double> &outputs) {

  if (trainingInputs_.eof() || trainingOutputs_.eof() ||
      !trainingInputs_.good() || !trainingOutputs_.good()) [[unlikely]]
    return false;

  if (inputs.size() < 794) [[unlikely]]
    logger::Logger::Log(logger::Logger::LogLevel::error,
                        "The input vector is too small");
  if (outputs.size() < 784) [[unlikely]]
    logger::Logger::Log(logger::Logger::LogLevel::error,
                        "The output vector is too small");

  char byte;
  double noise;
  for (int i = 0; i < 784; i++) {
    noise = GetNoise_();
    trainingInputs_.read(&byte, 1);
    outputs[i] = (double)((unsigned char)byte) / 255;
    inputs[i] = outputs[i] * (1 - noise) +
                (((double)std::rand() / (double)RAND_MAX) * noise);
    trainingOutputArray_[i] = outputs[i];
    trainingInputArray_[i] = inputs[i];
  }
  for (int i = 0; i < 10; i++)
    trainingInputArray_[i + 784] = inputs[i + 784] = 0;
  if (!trainingOutputs_.read(&byte, 1))
    return false;
  inputs[(int)byte + 784] = 1;
  trainingInputArray_[(int)byte + 784] = 1;
  return true;
}

void MnistGeneration::GetSameTrainingData(std::vector<double> &inputs,
                                          std::vector<double> &outputs) {
  std::copy(trainingInputArray_.cbegin(), trainingInputArray_.cend(),
            inputs.begin());
  std::copy(trainingOutputArray_.cbegin(), trainingOutputArray_.cend(),
            outputs.begin());
}

bool MnistGeneration::GetNextTestData(std::vector<double> &inputs,
                                      std::vector<double> &outputs) {
  if (!testInputs_ || !testOutputs_) [[unlikely]]
    return false;

  if (inputs.size() < 794) [[unlikely]]
    logger::Logger::Log(logger::Logger::LogLevel::error,
                        "The input vector is too small");
  if (outputs.size() < 784) [[unlikely]]
    logger::Logger::Log(logger::Logger::LogLevel::error,
                        "The output vector is too small");

  char byte;
  double noise;
  noise = GetNoise_();
  for (int i = 0; i < 784; i++) {
    testInputs_.read(&byte, 1);
    testOutputArray_[i] = outputs[i] = (double)((unsigned char)byte) / 255.0;
    testInputArray_[i] = inputs[i] =
        outputs[i] * (1 - noise) +
        (((double)std::rand() / (double)RAND_MAX) * noise);
  }
  for (int i = 0; i < 10; i++) {
    inputs[i + 784] = 0;
    testInputArray_[i + 784] = 0;
  }
  if (!testOutputs_.read(&byte, 1))
    return false;
  inputs[(int)byte + 784] = 1;
  testInputArray_[(int)byte + 784] = 1;
  return true;
}

void MnistGeneration::GetSameTestData(std::vector<double> &inputs,
                                      std::vector<double> &outputs) {
  std::copy(testInputArray_.cbegin(), testInputArray_.cend(), inputs.begin());
  std::copy(testOutputArray_.cbegin(), testOutputArray_.cend(),
            outputs.begin());
}

bool MnistGeneration::LoadTrainingData() {
  trainingInputs_.close();
  trainingOutputs_.close();
  trainingInputs_.open("Assets/data/train-images.ubyte",
                       std::ios::in | std::ios::binary);
  if (!trainingInputs_)
    return false;
  trainingOutputs_.open("Assets/data/train-labels.ubyte",
                        std::ios::in | std::ios::binary);
  if (!trainingOutputs_)
    return false;

  char *skip = new char[16];
  trainingInputs_.read(skip, 16);
  trainingOutputs_.read(skip, 8);
  delete[] skip;
  return true;
}

bool MnistGeneration::LoadTestData() {
  testInputs_.close();
  testOutputs_.close();
  testInputs_.open("Assets/data/test-images.ubyte",
                   std::ios::in | std::ios::binary);
  if (!testInputs_)
    return false;
  testOutputs_.open("Assets/data/test-labels.ubyte",
                    std::ios::in | std::ios::binary);
  if (!testOutputs_)
    return false;
  char *skip = new char[16];
  testInputs_.read(skip, 16);
  testOutputs_.read(skip, 8);
  delete[] skip;
  return true;
}

const std::vector<int> MnistGeneration::GetNetworkLayout() const {
  return networkLayout_;
}

const inline int MnistGeneration::GetNumberOfLayers() const {
  return numberOfLayers_;
}

const std::vector<std::function<double(double)>> &
MnistGeneration::GetActivationFunctions() const {
  return activationFunctions_;
}

const std::vector<std::function<double(double)>> &
MnistGeneration::GetActivationFunctionDerivatives() const {
  return activationFunctionDerivatives_;
}

void MnistGeneration::SetNoise(double noise) { noise_ = noise; }

void MnistGeneration::GetExample(std::vector<double> &inputs,
                                 std::vector<double> &outputs) {
  std::fstream input_file("Assets/data/test-images.ubyte",
                          std::ios::in | std::ios::binary);
  std::fstream output_file("Assets/data/test-labels.ubyte",
                           std::ios::in | std::ios::binary);
  input_file.seekg(16, std::ios::cur);
  output_file.seekg(8, std::ios::cur);

  char byte;
  double noise;
  noise = noise_;
  for (int i = 0; i < 784; i++) {
    input_file.read(&byte, 1);
    outputs[i] = (double)((unsigned char)byte) / 255.0;
    inputs[i] = outputs[i] * (1 - noise) +
                (((double)std::rand() / (double)RAND_MAX) * noise);
  }
  for (int i = 0; i < 10; i++) {
    inputs[i + 784] = 0;
  }
  output_file.read(&byte, 1);
  inputs[(int)byte + 784] = 1;
}

double MnistGeneration::GetNoise_() const {
  double ret = (double)std::rand() / (double)RAND_MAX;
  ret /= 2;
  ret *= ret;
  ret = 1 - ret;
  ret *= noise_;
  return ret;
}
