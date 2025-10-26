FROM openjdk:17-slim

RUN apt-get update && \
    apt-get install -y python3 python3-pip curl && \
    apt-get clean

WORKDIR /app
COPY . /app
RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "main.py"]
