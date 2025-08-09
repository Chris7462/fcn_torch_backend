# ModelGeneration.cmake
# This file handles automatic ONNX and TensorRT engine generation

# Include shared model configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake)

# Create directories for models
file(MAKE_DIRECTORY ${MODELS_DIR})

# Find Python3
find_program(PYTHON3_EXECUTABLE python3 REQUIRED)
if(NOT PYTHON3_EXECUTABLE)
  message(FATAL_ERROR "Python3 not found. Please install Python3.")
endif()

# Custom target to generate Torch script model
add_custom_command(
  OUTPUT ${MODEL_PATH}
  COMMAND ${PYTHON3_EXECUTABLE} ${EXPORT_SCRIPT_PATH} --output-dir ${MODELS_DIR}
  DEPENDS ${EXPORT_SCRIPT_PATH}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating ScriptTorch format: ${MODEL_FILE}..."
  VERBATIM
)

# Create custom targets that can be built
add_custom_target(generate_model
  DEPENDS ${MODEL_PATH}
)

# Function to add model generation dependency to a target
function(add_model_generation_dependency target_name)
  add_dependencies(${target_name} generate_model)
endfunction()

# # Install generated models
if(EXISTS ${MODELS_DIR})
  install(DIRECTORY ${MODELS_DIR}/
    DESTINATION share/${PROJECT_NAME}/models
    FILES_MATCHING PATTERN "*.pt")
endif()
