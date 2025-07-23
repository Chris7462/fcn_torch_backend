//// C++ standard library includes
#include <chrono>
#include <numeric>
#include <stdexcept>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Google Test includes
#include <gtest/gtest.h>

// Torch includes
#include <torch/torch.h>

// Local includes
#define private public
#include "fcn_torch_backend/fcn_torch_backend.hpp"
#undef private


class FCNTorchBackendTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    try {
      segmentor_ = std::make_unique<fcn_torch_backend::FCNTorchBackend>(model_path_);
    } catch (const std::exception & e) {
      GTEST_SKIP() << "Failed to initialize Pytorch segmentor: " << e.what();
    }
  }

  void TearDown() override
  {
    // Clean up if needed
  }

  cv::Mat load_test_image()
  {
    cv::Mat image = cv::imread(image_path_);
    if (image.empty()) {
      throw std::runtime_error("Failed to load test image: " + image_path_);
    }
    return image;
  }

  cv::Mat create_overlay(const cv::Mat & original, const cv::Mat & segmentation, float alpha = 0.5f)
  {
    cv::Mat overlay;
    cv::addWeighted(original, 1.0f - alpha, segmentation, alpha, 0.0, overlay);
    return overlay;
  }

  void save_results(
    const cv::Mat & original, const cv::Mat & segmentation,
    const cv::Mat & overlay, const std::string & suffix = "")
  {
    cv::imwrite("test_output_original" + suffix + ".png", original);
    cv::imwrite("test_output_segmentation" + suffix + ".png", segmentation);
    cv::imwrite("test_output_overlay" + suffix + ".png", overlay);
  }

  std::unique_ptr<fcn_torch_backend::FCNTorchBackend> segmentor_;

private:
  const std::string model_path_ = "fcn_resnet50_374x1238.pt";
  const std::string image_path_ = "image_000.png";
};


TEST_F(FCNTorchBackendTest, TestBasicInference)
{
  cv::Mat image = load_test_image();

  // Validate input image
  EXPECT_FALSE(image.empty());
  EXPECT_EQ(image.type(), CV_8UC3);
  EXPECT_GT(image.rows, 0);
  EXPECT_GT(image.cols, 0);

  std::cout << "Input image size: " << image.cols << "x" << image.rows << std::endl;
  std::cout << "Using device: " << (segmentor_->device_.is_cuda() ? "CUDA" : "CPU") << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  cv::Mat segmentation = segmentor_->segment(image);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration<double, std::milli>(end - start);
  std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

  // Validate output
  EXPECT_FALSE(segmentation.empty());
  EXPECT_EQ(segmentation.rows, image.rows);
  EXPECT_EQ(segmentation.cols, image.cols);
  EXPECT_EQ(segmentation.type(), CV_8UC3);  // Should be colored output

  // Check if segmentation contains valid colors (not all black)
  cv::Scalar mean_color = cv::mean(segmentation);
  EXPECT_GT(mean_color[0] + mean_color[1] + mean_color[2], 0.0)
    << "Segmentation appears to be all black";

  // Create overlay
  cv::Mat overlay = create_overlay(image, segmentation, 0.5f);
  EXPECT_EQ(overlay.size(), image.size());
  EXPECT_EQ(overlay.type(), CV_8UC3);

  // Save results for visual inspection
  std::string device_suffix = segmentor_->device_.is_cuda() ? "_gpu" : "_cpu";
  save_results(image, segmentation, overlay, device_suffix);

  // Optional: Display results (comment out for automated testing)
  /*
  cv::imshow("Original", image);
  cv::imshow("Segmentation", segmentation);
  cv::imshow("Overlay", overlay);
  cv::waitKey(0);
  cv::destroyAllWindows();
  */
}

//TEST_F(FCNTorchBackendTest, TestMultipleInferences)
//{
//  cv::Mat image = load_test_image();

//  const int num_iterations = 10;
//  std::vector<double> inference_times;
//  cv::Mat first_result;

//  for (int i = 0; i < num_iterations; ++i) {
//    auto start = std::chrono::high_resolution_clock::now();
//    cv::Mat segmentation = inferencer_->segment(image);
//    auto end = std::chrono::high_resolution_clock::now();

//    auto duration = std::chrono::duration<double, std::milli>(end - start);
//    inference_times.push_back(duration.count());

//    // Validate output consistency
//    EXPECT_EQ(segmentation.rows, image.rows);
//    EXPECT_EQ(segmentation.cols, image.cols);
//    EXPECT_EQ(segmentation.type(), CV_8UC3);

//    // Store first result for consistency check
//    if (i == 0) {
//      first_result = segmentation.clone();
//    } else {
//      // Check if results are consistent (should be identical for same input)
//      cv::Mat diff;
//      cv::absdiff(first_result, segmentation, diff);
//      cv::Scalar sum_diff = cv::sum(diff);
//      double total_diff = sum_diff[0] + sum_diff[1] + sum_diff[2];
//      EXPECT_EQ(total_diff, 0.0) << "Results are not consistent across multiple inferences";
//    }
//  }

//  // Calculate statistics
//  double avg_time = std::accumulate(inference_times.begin(), inference_times.end(), 0.0) /
//    inference_times.size();
//  double min_time = *std::min_element(inference_times.begin(), inference_times.end());
//  double max_time = *std::max_element(inference_times.begin(), inference_times.end());

//  std::cout << "Multiple inference statistics (" << (use_cuda_ ? "GPU" : "CPU") << "):" << std::endl;
//  std::cout << "  Average: " << avg_time << " ms" << std::endl;
//  std::cout << "  Min: " << min_time << " ms" << std::endl;
//  std::cout << "  Max: " << max_time << " ms" << std::endl;

//  // Performance expectations (adjust based on your hardware and model complexity)
//  if (use_cuda_) {
//    EXPECT_LT(avg_time, 200.0) << "GPU inference should be reasonably fast";
//  } else {
//    EXPECT_LT(avg_time, 1000.0) << "CPU inference should complete within reasonable time";
//  }
//}

//TEST_F(FCNTorchBackendTest, TestBenchmarkInference)
//{
//  cv::Mat image = load_test_image();

//  const int warmup_iterations = 5;
//  const int benchmark_iterations = 50;

//  std::cout << "Running benchmark on " << (use_cuda_ ? "GPU" : "CPU") << std::endl;

//  // Warmup
//  std::cout << "Warming up..." << std::endl;
//  for (int i = 0; i < warmup_iterations; ++i) {
//    inferencer_->segment(image);
//  }

//  // Benchmark
//  std::cout << "Running benchmark..." << std::endl;
//  auto start = std::chrono::high_resolution_clock::now();

//  for (int i = 0; i < benchmark_iterations; ++i) {
//    inferencer_->segment(image);
//  }

//  auto end = std::chrono::high_resolution_clock::now();
//  auto total_duration = std::chrono::duration<double, std::milli>(end - start);

//  double avg_time = total_duration.count() / benchmark_iterations;
//  double fps = 1000.0 / avg_time;

//  std::cout << "Benchmark Results (" << (use_cuda_ ? "GPU" : "CPU") << "):" << std::endl;
//  std::cout << "Iterations: " << benchmark_iterations << std::endl;
//  std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
//  std::cout << "Average time per inference: " << avg_time << " ms" << std::endl;
//  std::cout << "Throughput: " << fps << " FPS" << std::endl;

//  // Basic performance check
//  EXPECT_GT(fps, 0.1) << "Throughput should be at least 0.1 FPS";
//}

//TEST_F(FCNTorchBackendTest, TestInputValidation)
//{
//  // Test with empty image
//  cv::Mat empty_image;
//  EXPECT_THROW(inferencer_->segment(empty_image), std::invalid_argument);

//  // Test with different image sizes
//  std::vector<cv::Size> test_sizes = {
//    cv::Size(320, 240),
//    cv::Size(640, 480),
//    cv::Size(1024, 768),
//    cv::Size(100, 100)
//  };

//  for (const auto& size : test_sizes) {
//    cv::Mat test_image = cv::Mat::zeros(size, CV_8UC3);
//    // Fill with some pattern to make it non-zero
//    cv::rectangle(test_image, cv::Rect(10, 10, size.width-20, size.height-20),
//                  cv::Scalar(128, 64, 192), -1);

//    EXPECT_NO_THROW({
//      cv::Mat result = inferencer_->segment(test_image);
//      EXPECT_EQ(result.size(), size);
//      EXPECT_EQ(result.type(), CV_8UC3);
//    }) << "Failed for image size: " << size.width << "x" << size.height;
//  }
//}

//TEST_F(FCNTorchBackendTest, TestColormapApplication)
//{
//  cv::Mat image = load_test_image();
//  cv::Mat segmentation = segmentor_->segment(image);

//  // Check that the segmentation uses expected colors from Pascal VOC colormap
//  std::set<cv::Vec3b> unique_colors;
//
//  for (int i = 0; i < segmentation.rows; ++i) {
//    for (int j = 0; j < segmentation.cols; ++j) {
//      unique_colors.insert(segmentation.at<cv::Vec3b>(i, j));
//    }
//  }

//  std::cout << "Found " << unique_colors.size() << " unique colors in segmentation" << std::endl;

//  // Should have at least background color and some object colors
//  EXPECT_GT(unique_colors.size(), 1) << "Segmentation should contain multiple colors";
//  EXPECT_LE(unique_colors.size(), 21) << "Should not exceed Pascal VOC class count";

//  // Check if background color (black) is present
//  cv::Vec3b background_color(0, 0, 0);
//  bool has_background = unique_colors.find(background_color) != unique_colors.end();
//  EXPECT_TRUE(has_background) << "Segmentation should contain background pixels";
//}

//// Test with multiple different images (if available)
//TEST_F(FCNTorchBackendTest, DISABLED_TestMultipleImages)
//{
//  std::vector<std::string> test_images = {
//    "image_000.png",
//    "image_001.png",
//    "image_002.png"
//  };

//  int successful_tests = 0;

//  for (const auto & image_path : test_images) {
//    cv::Mat image = cv::imread(image_path);
//    if (image.empty()) {
//      std::cout << "Skipping missing image: " << image_path << std::endl;
//      continue;
//    }

//    try {
//      auto start = std::chrono::high_resolution_clock::now();
//      cv::Mat segmentation = inferencer_->segment(image);
//      auto end = std::chrono::high_resolution_clock::now();

//      auto duration = std::chrono::duration<double, std::milli>(end - start);
//      std::cout << "Processed " << image_path << " in " << duration.count() << " ms" << std::endl;

//      cv::Mat overlay = create_overlay(image, segmentation);

//      // Save results with image-specific suffix
//      std::string suffix = "_" + std::to_string(successful_tests) +
//                          (use_cuda_ ? "_gpu" : "_cpu");
//      save_results(image, segmentation, overlay, suffix);

//      successful_tests++;

//    } catch (const std::exception & e) {
//      FAIL() << "Failed to process image " << image_path << ": " << e.what();
//    }
//  }

//  EXPECT_GT(successful_tests, 0) << "No test images were successfully processed";
//  std::cout << "Successfully processed " << successful_tests << " test images" << std::endl;
//}

//// Test device switching if both CPU and GPU are available
//TEST_F(FCNTorchBackendTest, DISABLED_TestDeviceSwitching)
//{
//  if (!torch::cuda::is_available()) {
//    GTEST_SKIP() << "CUDA not available, skipping device switching test";
//  }

//  cv::Mat image = load_test_image();

//  // Test CPU inference
//  fcn_torch_backend::FCNTorchBackend cpu_inferencer(model_path_, false);
//  auto start_cpu = std::chrono::high_resolution_clock::now();
//  cv::Mat cpu_result = cpu_inferencer.segment(image);
//  auto end_cpu = std::chrono::high_resolution_clock::now();
//  auto cpu_duration = std::chrono::duration<double, std::milli>(end_cpu - start_cpu);

//  // Test GPU inference
//  fcn_torch_backend::FCNTorchBackend gpu_inferencer(model_path_, true);
//  auto start_gpu = std::chrono::high_resolution_clock::now();
//  cv::Mat gpu_result = gpu_inferencer.segment(image);
//  auto end_gpu = std::chrono::high_resolution_clock::now();
//  auto gpu_duration = std::chrono::duration<double, std::milli>(end_gpu - start_gpu);

//  std::cout << "CPU inference time: " << cpu_duration.count() << " ms" << std::endl;
//  std::cout << "GPU inference time: " << gpu_duration.count() << " ms" << std::endl;

//  // Results should be identical (allowing for minor floating point differences)
//  EXPECT_EQ(cpu_result.size(), gpu_result.size());
//  EXPECT_EQ(cpu_result.type(), gpu_result.type());

//  // Compare results (they should be very similar, if not identical)
//  cv::Mat diff;
//  cv::absdiff(cpu_result, gpu_result, diff);
//  cv::Scalar sum_diff = cv::sum(diff);
//  double total_diff = sum_diff[0] + sum_diff[1] + sum_diff[2];

//  // Allow for some small differences due to different precision handling
//  double tolerance = cpu_result.rows * cpu_result.cols * 3 * 5; // Allow up to 5 difference per pixel per channel
//  EXPECT_LE(total_diff, tolerance) << "CPU and GPU results differ significantly";

//  std::cout << "Total pixel difference between CPU and GPU: " << total_diff << std::endl;
//}
