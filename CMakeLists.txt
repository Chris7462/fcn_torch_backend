cmake_minimum_required(VERSION 3.8)
project(fcn_segmentation)

# Find required packages
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Add executable
add_executable(fcn_segmentation
  src/fcn_segmentation_torch.cpp
  src/main.cpp)

# Include directories
target_include_directories(fcn_segmentation
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

# Link libraries
target_link_libraries(fcn_segmentation
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS})

# Set properties for torch
set_property(TARGET fcn_segmentation PROPERTY CXX_STANDARD 17)
