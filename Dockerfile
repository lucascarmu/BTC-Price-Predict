FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY app/ app/
COPY models/ensemble/ models/ensemble/
COPY data/test_dataset/ data/test_dataset/
COPY scripts/ scripts/

RUN pip install --no-cache-dir -r requirements.txt

VOLUME /app/outputs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]