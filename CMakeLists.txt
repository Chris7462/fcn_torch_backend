cmake_minimum_required(VERSION 3.8)
project(fcn_segmentation_torch VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Define the library target
add_library(fcn_segmentation_torch STATIC
  src/fcn_segmentation_torch.cpp)

# Specify include directories for the target
target_include_directories(fcn_segmentation_torch
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

# Link libraries
target_link_libraries(fcn_segmentation_torch
  PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS})

target_compile_features(fcn_segmentation_torch PUBLIC cxx_std_17)

# Optionally install the library
install(TARGETS fcn_segmentation_torch
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Install header files
install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Export targets and dependencies
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(OpenCV Torch)

# Export include directories for downstream packages
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_fcn_segmentation_torch
    test/test_fcn_segmentation_torch.cpp)

  target_link_libraries(test_fcn_segmentation_torch
    fcn_segmentation_torch
    gtest_main)

endif()

ament_package()
