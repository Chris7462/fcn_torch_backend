#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <memory>

class FcnSegmentation {
private:
  torch::jit::script::Module model;
  torch::Device device;

  // PASCAL VOC colormap
  const std::vector<std::vector<int>> PASCAL_VOC_COLORMAP = {
    {0, 0, 0},       // Background
    {128, 0, 0},     // Aeroplane
    {0, 128, 0},     // Bicycle
    {128, 128, 0},   // Bird
    {0, 0, 128},     // Boat
    {128, 0, 128},   // Bottle
    {0, 128, 128},   // Bus
    {128, 128, 128}, // Car
    {64, 0, 0},      // Cat
    {192, 0, 0},     // Chair
    {64, 128, 0},    // Cow
    {192, 128, 0},   // Dining table
    {64, 0, 128},    // Dog
    {192, 0, 128},   // Horse
    {64, 128, 128},  // Motorbike
    {192, 128, 128}, // Person
    {0, 64, 0},      // Potted plant
    {128, 64, 0},    // Sheep
    {0, 192, 0},     // Sofa
    {128, 192, 0},   // Train
    {0, 64, 128},    // TV monitor
  };

  cv::Mat apply_colormap(const cv::Mat& mask) {
    cv::Mat colormap(mask.rows, mask.cols, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < mask.rows; ++i) {
      for (int j = 0; j < mask.cols; ++j) {
        int label = static_cast<int>(mask.at<uint8_t>(i, j));
        if (label < PASCAL_VOC_COLORMAP.size()) {
          // OpenCV uses BGR format, so we need to reverse the RGB values
          colormap.at<cv::Vec3b>(i, j) = cv::Vec3b(
            PASCAL_VOC_COLORMAP[label][2],  // B
            PASCAL_VOC_COLORMAP[label][1],  // G
            PASCAL_VOC_COLORMAP[label][0]   // R
          );
        }
      }
    }
    return colormap;
  }

  torch::Tensor preprocess(const cv::Mat& image) {
    cv::Mat float_img;
    image.convertTo(float_img, CV_32FC3, 1.0 / 255);

    // Convert BGR to RGB
    cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);

    // Normalize with ImageNet mean/std
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    channels[0] = (channels[0] - 0.485f) / 0.229f;
    channels[1] = (channels[1] - 0.456f) / 0.224f;
    channels[2] = (channels[2] - 0.406f) / 0.225f;
    cv::merge(channels, float_img);

    // Convert to torch tensor
    torch::Tensor tensor_image = torch::from_blob(
      float_img.data, {1, image.rows, image.cols, 3}, torch::kFloat32);
    tensor_image = tensor_image.permute({0, 3, 1, 2}).contiguous();  // [B, C, H, W]

    // Move tensor to the appropriate device (GPU/CPU)
    return tensor_image.clone().to(device);
  }

public:
  FcnSegmentation(const std::string& model_path, bool use_cuda = false)
    : device(use_cuda ? torch::kCUDA : torch::kCPU) {

    // Load model
    try {
      model = torch::jit::load(model_path);
      model.to(device);
      model.eval();  // Set to evaluation mode

      std::cout << "Model loaded successfully on "
                << (device.type() == torch::kCUDA ? "GPU" : "CPU") << std::endl;
    } catch (const c10::Error& e) {
      throw std::runtime_error("Error loading the model: " + std::string(e.what()));
    }
  }

  // Main inference method - always returns colored segmentation
  cv::Mat segment(const cv::Mat& image) {
    if (image.empty()) {
      throw std::invalid_argument("Input image is empty");
    }

    // Preprocess image (automatically moves to correct device)
    torch::Tensor input_tensor = preprocess(image);

    // Disable gradient computation for inference
    torch::NoGradGuard no_grad;

    // Run inference
    std::vector<torch::jit::IValue> inputs{input_tensor};
    torch::Tensor output = model.forward(inputs).toTensor();  // [1, 21, H, W]

    // Get segmentation mask and move back to CPU for OpenCV processing
    torch::Tensor mask = output.argmax(1).squeeze().to(torch::kU8).cpu();  // [H, W]

    // Convert tensor to OpenCV Mat
    cv::Mat mask_mat(image.rows, image.cols, CV_8UC1, mask.data_ptr<uint8_t>());

    // Clone the data since the tensor memory might be deallocated
    cv::Mat result_mask = mask_mat.clone();

    // Apply colormap and return colored result
    return apply_colormap(result_mask);
  }
};

int main(int argc, const char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: ./main <pt_model> <image_path>\n";
    return -1;
  }

  std::string model_path = argv[1];
  std::string image_path = argv[2];

  // Load image
  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
    std::cerr << "Failed to read image from " << image_path << "\n";
    return -1;
  }

  try {
    // Check CUDA availability and create segmentor
    bool use_cuda = torch::cuda::is_available();

    FcnSegmentation segmentor(model_path, use_cuda);

    // Perform segmentation
    cv::Mat colored_mask = segmentor.segment(image);

    // Save result
    std::string output_path = "segmentation_output_colored.png";
    if (cv::imwrite(output_path, colored_mask)) {
      std::cout << "Inference complete. Saved colored output as " << output_path << std::endl;
    } else {
      std::cerr << "Failed to save segmentation result." << std::endl;
      return -1;
    }

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
