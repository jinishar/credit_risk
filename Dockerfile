FROM python:3.11-slim

WORKDIR /app

# System deps: lightgbm needs libgomp at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY sql/ sql/
COPY notebooks/ notebooks/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# data/ and models/ are volume-mounted at runtime (see docker-compose.yml);
# create them here too so a plain `docker run` without volumes still works.
RUN mkdir -p data models

EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]
