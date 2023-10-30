# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app
COPY biased_words_and_suggestions.csv biased_words_and_suggestions.csv
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE 8000

COPY . .

CMD ["python3", "inclusion.py"]

