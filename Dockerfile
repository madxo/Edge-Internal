# syntax=docker/dockerfile:1

FROM python:3.8-slim

WORKDIR /app
COPY biased_words_and_suggestions.csv biased_words_and_suggestions.csv
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*


EXPOSE 8000

COPY . .

CMD ["python3", "inclusion.py"]

