FROM python:3.12-slim

WORKDIR /app

# libgomp1 нужен LightGBM (OpenMP runtime), без него падает с OSError при импорте
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p src/db logs

CMD ["python", "-m", "src.main"]