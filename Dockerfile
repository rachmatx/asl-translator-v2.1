# Gunakan image Python ringan
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies sistem yang diperlukan oleh OpenCV & MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Salin requirements file dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin sisa file aplikasi
COPY . .

# Expose port yang akan digunakan (default Hugging Face = 7860)
EXPOSE 7860

# Jalankan server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
