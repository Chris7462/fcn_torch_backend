#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// OpenCV includes
#include <opencv2/core.hpp>

// Torch includes
#include <torch/script.h>
#include <torch/torch.h>


namespace fcn_torch_backend
{
class FCNTorchBackend
{
public:
  FCNTorchBackend(const std::string & model_path, torch::Device device = torch::kCPU);
  cv::Mat segment(const cv::Mat & image);

private:
  cv::Mat apply_colormap(const cv::Mat & mask);
  torch::Tensor preprocess(const cv::Mat & image);

private:
  torch::jit::script::Module model_;
  torch::Device device_;
};

} // namespace fcn_torch_backend
