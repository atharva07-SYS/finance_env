FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    gymnasium==0.29.1 \
    yfinance \
    numpy \
    pandas \
    fastapi \
    uvicorn \
    matplotlib \
    gradio==5.23.0

# Install Meta OpenEnv
RUN pip install --no-cache-dir \
    git+https://github.com/meta-pytorch/OpenEnv.git

# Copy all files
COPY . .

# Expose port
EXPOSE 7860

# Run the server
CMD ["python", "inference.py"]