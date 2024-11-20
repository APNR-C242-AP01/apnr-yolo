# Use the official Python 3.10 image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and to buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Install Jupyter Notebook
RUN pip install --no-cache-dir jupyter

# Expose the default Jupyter Notebook port
EXPOSE 8888

# Set the command to run Jupyter Notebook on container start
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=${JUPYTER_TOKEN}"]

