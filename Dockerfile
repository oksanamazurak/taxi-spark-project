FROM python:3.8-slim

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y openjdk-17-jdk openjdk-17-jre curl && \
    apt-get clean

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY . /app

CMD ["python3", "main.py"]



# FOR MACOS
# FROM python:3.8-slim

# RUN apt-get update && \
#     apt-get install -y openjdk-17-jdk curl && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
# ENV PATH="${JAVA_HOME}/bin:${PATH}"

# WORKDIR /app

# COPY requirements.txt /app/requirements.txt
# RUN pip3 install --no-cache-dir -r /app/requirements.txt

# COPY . /app

# CMD ["python3", "main.py"]