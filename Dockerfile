# Lightweight Python image with Python 3.13
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Upgrade pip first
RUN pip install --upgrade pip

# Install heavy dependencies separately to leverage caching
RUN pip install --no-cache-dir torch==2.8.0 \
    transformers==4.55.2 \
    sentence-transformers==5.1.0 \
    tokenizers==0.21.4

# Copy requirements.txt and install the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Start FastAPI when the container runs
CMD ["uvicorn", "rag.app:app", "--host", "0.0.0.0", "--port", "8000"]
