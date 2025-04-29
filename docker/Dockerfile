FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY ml_model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code (directly to /app instead of /app/app)
COPY api/api_main.py .

# Copy model files (directly to /app instead of /app/app)
COPY ml_model/saved_models/ ./ml_model/saved_models/

EXPOSE 8000

CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]