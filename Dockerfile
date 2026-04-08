FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy all files first
COPY . .

# Install all dependencies
RUN pip install --no-cache-dir \
    gymnasium==0.29.1 \
    yfinance==1.2.0 \
    numpy \
    pandas \
    fastapi \
    uvicorn \
    matplotlib \
    openenv-core \
    gradio==5.23.0

# Expose port
EXPOSE 7860

# Run the server
CMD ["python", "inference.py"]