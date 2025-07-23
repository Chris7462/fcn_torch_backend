#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <array>


namespace config
{
// ImageNet normalization constants
constexpr std::array<float, 3> MEAN = {0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> STDDEV = {0.229f, 0.224f, 0.225f};

// Pascal VOC colors for visualization
constexpr std::array<std::array<unsigned char, 3>, 21> PASCAL_VOC_COLORMAP = {{
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
}};

} // namespace config

/*
 * Here's a C++ implementation of the PASCAL_VOC_COLORMAP—the standard 21-class
 * color map used for semantic segmentation in the PASCAL VOC dataset. Each
 * class is assigned a unique RGB triplet.
 */

//using Color = std::array<unsigned char, 3>;

//// Generate the PASCAL VOC colormap
//std::vector<Color> get_pascal_voc_colormap() {
//    std::vector<Color> colormap(256); // supports up to 256 classes
//    for (int i = 0; i < 256; ++i) {
//        unsigned char r = 0, g = 0, b = 0;
//        int cid = i;
//        for (int j = 0; j < 8; ++j) {
//            r |= ((cid >> 0) & 1) << (7 - j);
//            g |= ((cid >> 1) & 1) << (7 - j);
//            b |= ((cid >> 2) & 1) << (7 - j);
//            cid >>= 3;
//        }
//        colormap[i] = {r, g, b};
//    }
//    return colormap;
//}
