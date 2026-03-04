FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    ccache \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir paddlepaddle-gpu==3.3.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

# NOTE: OCR models will download on first run. paddlepaddle-gpu cannot import
# during build because libcuda.so.1 is only available at runtime via --gpus.

COPY ocr_process_images.py .

RUN mkdir -p /app/images

CMD ["python3", "ocr_process_images.py"]