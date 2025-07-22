#include <iostream>

#include <opencv2/imgcodecs.hpp>

#include "fcn_segmentation_torch/fcn_segmentation_torch.hpp"



int main(int argc, const char* argv[])
{
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

    fcn_segmentation_torch::FcnSegmentationTorch segmentor(model_path, use_cuda);

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
