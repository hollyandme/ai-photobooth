FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p temp_images

EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
