# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app
COPY biased_words_and_suggestions.csv biased_words_and_suggestions.csv
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers tqdm numpy scikit-learn scipy nltk sentencepiece pillow
RUN pip install --no-cache-dir  --no-deps sentence-transformers


EXPOSE 8000

COPY . .

CMD ["python3", "inclusion.py"]

