# Start from Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install pandas
RUN pip install pandas

# Copy our application
COPY app.py .

# Run the application when container starts
CMD ["python", "app.py"]