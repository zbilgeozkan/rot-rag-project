# Lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Start FastAPI when the container runs
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
