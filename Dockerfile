FROM python:3.13-slim
RUN apt-get update && apt-get install -y libxcb1 libxkbcommon0 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
