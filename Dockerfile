# Use the official PyTorch image with CUDA 12.1 as the base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables to prevent issues with Python in Docker
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Copy the project files to the working directory
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libxtst6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install required Python packages
RUN pip install --upgrade pip \
    && pip install -e . \
    && pip install --no-cache-dir torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html \
    && pip install -r requirements.txt 

# Install MLflow and Jupyter
RUN pip install mlflow jupyterlab

# Expose ports for Jupyter and MLflow
EXPOSE 8888 5000

# Start Jupyter and MLflow
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --allow-root --no-browser & mlflow server --host 0.0.0.0 --default-artifact-root ./mlruns --backend-store-uri sqlite:///mlflow.db"]

# docker build -t image .
# docker run -it -p 8888:8888 -p 5000:5000 image