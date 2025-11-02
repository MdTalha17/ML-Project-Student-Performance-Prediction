FROM python:3.11-slim-bookworm

WORKDIR /app

# install deps for catboost / xgboost build (important!)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ make awscli && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# pip with no cache & no compile artifacts
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

CMD ["python3", "app.py"]