# Stage 1: Build environment
FROM tensorflow/tensorflow:2.19.0 as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime environment
FROM tensorflow/tensorflow:2.19.0

WORKDIR /app

# Copy installed dependencies from builder
ENV PYTHONPATH=/root/.local/lib/python3.11/site-packages
COPY --from=builder /root/.local /root/.local

# Create uploads directory with proper permissions
RUN mkdir -p /app/uploads && chmod 777 /app/uploads

# Copy app code
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
