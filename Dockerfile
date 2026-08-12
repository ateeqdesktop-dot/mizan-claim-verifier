FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MIZAN_MODEL_PATH=/app/models/classifier.joblib \
    MIZAN_INDEX_PATH=/app/models/retriever.joblib

COPY pyproject.toml requirements.txt README.md ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
COPY frontend ./frontend
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "mizan.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
