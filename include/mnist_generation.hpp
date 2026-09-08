#pragma once
#include "network_data.hpp"
#include <cstdlib>
#include <ctime>
#include <fstream>

class MnistGeneration : public axon::NetworkData {
private:
  std::vector<int> networkLayout_ = {794, 256, 256, 256, 784};
  int numberOfLayers_ = 5;

  static double noise_;

  std::vector<double> trainingInputArray_;
  std::vector<double> trainingOutputArray_;
  std::vector<double> testInputArray_;
  std::vector<double> testOutputArray_;

  std::fstream trainingInputs_;
  std::fstream trainingOutputs_;
  std::fstream testInputs_;
  std::fstream testOutputs_;

  const std::vector<std::function<double(double)>> activationFunctions_;
  const std::vector<std::function<double(double)>>
      activationFunctionDerivatives_;

public:
  MnistGeneration();

  ~MnistGeneration();

  bool GetNextTrainingData(std::vector<double> &inputs,
                           std::vector<double> &outputs) override;

  void GetSameTrainingData(std::vector<double> &inputs,
                           std::vector<double> &outputs) override;

  bool GetNextTestData(std::vector<double> &inputs,
                       std::vector<double> &outputs) override;

  void GetSameTestData(std::vector<double> &inputs,
                       std::vector<double> &outputs) override;

  bool LoadTrainingData() override;

  bool LoadTestData() override;

  const std::vector<int> GetNetworkLayout() const override;

  const inline int GetNumberOfLayers() const override;

  const std::vector<std::function<double(double)>> &
  GetActivationFunctions() const override;
  const std::vector<std::function<double(double)>> &
  GetActivationFunctionDerivatives() const override;

  static void SetNoise(double noise);

  static void SetHardNoise(bool hn);

  static void GetExample(std::vector<double> &inputs,
                         std::vector<double> &outputs);

private:
  double GetNoise_() const;
};
