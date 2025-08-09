# Config.cmake
# Shared configuration for model generation and testing
# This file contains all model-related configuration variables

# Model configuration - change these variables to use different models
set(MODEL_NAME "fcn_resnet101_374x1238" CACHE STRING "Base name of the model")
set(EXPORT_SCRIPT "export_fcn_to_pt.py" CACHE STRING "Python script model")

# Derived file names (automatically generated from MODEL_NAME)
set(MODEL_FILE "${MODEL_NAME}.pt")

# Common directory paths
set(MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/models)
set(SCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/script)
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)

# Full file paths
set(MODEL_PATH ${MODELS_DIR}/${MODEL_FILE})
set(EXPORT_SCRIPT_PATH ${SCRIPTS_DIR}/${EXPORT_SCRIPT})

# Test configuration
set(TEST_IMAGE_FILES
  image_000.png
  image_001.png
  image_002.png
  CACHE STRING "List of test image files")
