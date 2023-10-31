# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app
COPY biased_words_and_suggestions.csv biased_words_and_suggestions.csv
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir sentence-transformers
RUN pip uninstall nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 nvidia-nccl-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-cupti-cu12 nvidia-cublas-cu1 nvidia-cusparse-cu12 nvidia-cudnn-cu12 nvidia-cusolver-cu12


EXPOSE 8000

COPY . .

CMD ["python3", "inclusion.py"]

