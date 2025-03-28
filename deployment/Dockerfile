# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install -r deployment/requirements.txt

# Expose API port
EXPOSE 5000

# Run FastAPI/Flask app
CMD ["python", "deployment/app.py"]
