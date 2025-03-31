# Dockerfile
# Example Dockerfile for deploying the pipeline

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (adjust as necessary)
COPY . .

# By default, run the training script (or switch to app.py if deploying an API)
CMD ["python", "src/models/train_model.py"]
