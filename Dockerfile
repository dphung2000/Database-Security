FROM python:3.11.10-bookworm

RUN apt-get update && apt-get install -y \
build-essential \
curl \
sqlite3 \
iputils-ping \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
