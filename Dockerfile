FROM python:3.11-slim

# Install system libraries needed for OpenCV & MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "backend.app:create_app()", "--timeout", "180", "--workers", "1"]