#include <opencv2/imgproc.hpp>

#include "fcn_segmentation_torch/config.hpp"
#include "fcn_segmentation_torch/fcn_segmentation_torch.hpp"

namespace fcn_segmentation_torch
{

FcnSegmentationTorch::FcnSegmentationTorch(const std::string & model_path, bool use_cuda)
: device_(use_cuda ? torch::kCUDA : torch::kCPU)
{
  // Load model
  try {
    model_ = torch::jit::load(model_path);
    model_.to(device_);
    model_.eval();  // Set to evaluation mode

    std::cout << "Model loaded successfully on "
              << (device_.type() == torch::kCUDA ? "GPU" : "CPU") << std::endl;
  } catch (const c10::Error & e) {
    throw std::runtime_error("Error loading the model: " + std::string(e.what()));
  }
}

// Main inference method - always returns colored segmentation
cv::Mat FcnSegmentationTorch::segment(const cv::Mat & image)
{
  if (image.empty()) {
    throw std::invalid_argument("Input image is empty");
  }

  // Preprocess image (automatically moves to correct device)
  torch::Tensor input_tensor = preprocess(image);

  // Disable gradient computation for inference
  torch::NoGradGuard no_grad;

  // Run inference
  std::vector<torch::jit::IValue> inputs{input_tensor};
  torch::Tensor output = model_.forward(inputs).toTensor();  // [1, 21, H, W]

  // Get segmentation mask and move back to CPU for OpenCV processing
  torch::Tensor mask = output.argmax(1).squeeze(0).to(torch::kU8).cpu();

  // Convert tensor to OpenCV Mat
  cv::Mat mask_mat(image.rows, image.cols, CV_8UC1, mask.data_ptr<uint8_t>());

  // Clone the data since the tensor memory might be deallocated
  cv::Mat result_mask = mask_mat.clone();

  // Apply colormap and return colored result
  return apply_colormap(result_mask);
}

cv::Mat FcnSegmentationTorch::apply_colormap(const cv::Mat & mask)
{
  cv::Mat colormap(mask.rows, mask.cols, CV_8UC3, cv::Scalar(0, 0, 0));

  for (int i = 0; i < mask.rows; ++i) {
    for (int j = 0; j < mask.cols; ++j) {
      int label = static_cast<int>(mask.at<uint8_t>(i, j));
      if (label < config::PASCAL_VOC_COLORMAP.size()) {
        // OpenCV uses BGR format, so we need to reverse the RGB values
        colormap.at<cv::Vec3b>(i, j) = cv::Vec3b(
          config::PASCAL_VOC_COLORMAP[label][2],  // B
          config::PASCAL_VOC_COLORMAP[label][1],  // G
          config::PASCAL_VOC_COLORMAP[label][0]   // R
        );
      }
    }
  }

  return colormap;
}

torch::Tensor FcnSegmentationTorch::preprocess(const cv::Mat & image)
{
  cv::Mat float_img;
  image.convertTo(float_img, CV_32FC3, 1.0 / 255);

  // Convert BGR to RGB
  cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);

  // Normalize with ImageNet mean/std
  std::vector<cv::Mat> channels(3);
  cv::split(float_img, channels);
  channels[0] = (channels[0] - config::MEAN[0]) / config::STDDEV[0];
  channels[1] = (channels[1] - config::MEAN[1]) / config::STDDEV[1];
  channels[2] = (channels[2] - config::MEAN[2]) / config::STDDEV[2];
  cv::merge(channels, float_img);

  // Convert to torch tensor
  torch::Tensor tensor_image = torch::from_blob(
    float_img.data, {1, image.rows, image.cols, 3}, torch::kFloat32);
  tensor_image = tensor_image.permute({0, 3, 1, 2}).contiguous();  // [B, C, H, W]

  // Move tensor to the appropriate device (GPU/CPU)
  return tensor_image.to(device_);
}

} // namespace fcn_segmentation_torch
