FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model-dir \
    OUTPUT_DIR=/app/output \
    TRANSFORMERS_OFFLINE=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    poppler-utils \
    libsm6 \
    libxext6 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directories
WORKDIR /app
RUN mkdir -p /app/model-dir /app/output

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies with exact versions
RUN pip3 install --no-cache-dir numpy==1.24.3 \
    && pip3 install --no-cache-dir torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py /app/

# Expose port for Gradio
EXPOSE 7860

# Add verbose logging for debugging
CMD ["python3", "-u", "app.py"]