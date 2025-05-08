# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies if any are identified later (e.g., for specific libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
# If src layout is used strictly:
# COPY src/ ./src/
# COPY app.py .
# COPY scripts/ ./scripts/


# Expose port (Streamlit default is 8501)
EXPOSE 8501

# Command to run the application (update if using Flask or other entrypoint)
# For development, you might mount code as a volume in docker-compose and run directly.
# For a built image, this would be the run command.
CMD ["streamlit", "run", "app.py"] 