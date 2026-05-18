FROM python:3.12-slim
WORKDIR /app
COPY services/cxr-svc/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd -m -u 1000 appuser
COPY shared /app/shared
COPY services/cxr-svc .
USER appuser
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
