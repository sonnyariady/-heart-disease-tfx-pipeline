# Base image TensorFlow Serving
FROM tensorflow/serving:latest

# Environment variables
ENV MODEL_NAME=heart-disease-model
ENV MODEL_PATH=/models/heart-disease-model

# Copy serving model to container model directory
COPY serving_model/ ${MODEL_PATH}/

# Expose gRPC and REST API ports
EXPOSE 8500
EXPOSE 8501

# Command to run TensorFlow Serving
CMD ["tensorflow_model_server", "--port=8500", "--rest_api_port=8501", "--model_name=heart-disease-model", "--model_base_path=/models/heart-disease-model"]
